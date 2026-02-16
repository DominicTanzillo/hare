#!/usr/bin/env python3
"""Train HARE's learnable cross-attention projections on LaMP data.

Supervised retrieval objective: for each LaMP sample, the model learns to
produce attention weights that match an oracle relevance distribution
derived from ROUGE-1 overlap between profile targets and the ground truth.

Loss:
    L = -sum_j p_oracle[j] * log(mean_attention[j])

where p_oracle = softmax(ROUGE-1(profile_target[j], ground_truth) / tau).

The user state is warmed up with the first k profile items before computing
the loss, simulating the online setting where users have interaction history.

Usage:
    # Quick training (5 epochs, 100 samples)
    python experiments/train_attention.py --epochs 5 --max-samples 100

    # Full training on LaMP-4
    python experiments/train_attention.py --task lamp4 --epochs 10

    # Resume from checkpoint
    python experiments/train_attention.py --resume checkpoints/attention_weights.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans

from hare.attention.learnable_cross_attention import LearnableCrossAttention
from hare.bandits.attentive_bandit import UserState
from hare.evaluation.lamp import load_lamp
from hare.utils.embeddings import TfidfEmbedder


def compute_oracle_relevance(
    profile_targets: list[str],
    ground_truth: str,
    tau: float = 0.3,
) -> np.ndarray:
    """Compute oracle relevance distribution over profile items.

    Uses ROUGE-1 F1 between each profile target and the ground truth,
    then converts to a probability distribution via softmax with temperature.

    Parameters
    ----------
    profile_targets : list of str
        Target texts from user profile items.
    ground_truth : str
        The ground truth target for this sample.
    tau : float
        Softmax temperature. Lower = more peaked.

    Returns
    -------
    p_oracle : array of shape (n_profile,)
        Oracle probability distribution.
    """
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)

    scores = np.zeros(len(profile_targets))
    for i, target in enumerate(profile_targets):
        if target and ground_truth:
            result = scorer.score(ground_truth, target)
            scores[i] = result["rouge1"].fmeasure

    # Softmax with temperature
    logits = scores / max(tau, 1e-8)
    logits = logits - logits.max()  # numerical stability
    exp_logits = np.exp(logits)
    p_oracle = exp_logits / exp_logits.sum()

    return p_oracle


def _load_training_data(task: str, max_samples: int | None = None, seed: int = 42):
    """Load training data from LaMP or Amazon sources."""
    if task.lower().startswith("amazon"):
        from hare.evaluation.amazon import load_amazon_reviews, amazon_to_lamp_format
        category = task.split("_", 1)[1] if "_" in task else "Digital_Music"
        amazon_data = load_amazon_reviews(
            category=category,
            min_reviews_per_user=5,
            max_samples=max_samples,
            seed=seed,
        )
        return amazon_to_lamp_format(amazon_data), "amazon"
    else:
        data = load_lamp(task, split="train", max_samples=max_samples)
        return data, task


def train_attention(
    task: str = "lamp4",
    epochs: int = 10,
    lr: float = 1e-3,
    max_samples: int | None = None,
    n_warmup: int = 5,
    tau: float = 0.3,
    d_knowledge: int = 64,
    d_user: int = 32,
    n_heads: int = 4,
    d_k: int = 32,
    d_v: int = 32,
    n_clusters: int = 5,
    alpha: float = 1.5,
    seed: int = 42,
    checkpoint_dir: Path = Path("checkpoints"),
    resume: str | None = None,
) -> Path:
    """Train learnable cross-attention on LaMP or Amazon data.

    Returns path to saved checkpoint.
    """
    print("=" * 70)
    print(f"Training HARE Learnable Attention on {task.upper()}")
    print("=" * 70)

    # Load data (supports LaMP and Amazon)
    print(f"\nLoading {task} training data...")
    data, task_key = _load_training_data(task, max_samples, seed)
    print(f"  {len(data)} samples loaded")

    # Build TF-IDF embedder from all profile texts
    print("\nBuilding TF-IDF embedder...")
    all_texts = []
    for sample in data.samples:
        for item in sample.profile:
            text = item.get("text", item.get("abstract", ""))
            if text:
                all_texts.append(text)
        all_texts.append(sample.input_text)

    embedder = TfidfEmbedder(max_features=2000, output_dim=d_knowledge)
    embedder.fit(all_texts)
    print(f"  Embedder fitted: {embedder.dim}-dim output")
    actual_d = embedder.dim

    # Initialize attention module
    attention = LearnableCrossAttention(
        d_knowledge=actual_d,
        d_user=min(d_user, actual_d),
        d_k=min(d_k, actual_d),
        d_v=min(d_v, actual_d),
        n_heads=n_heads,
        n_clusters=n_clusters,
        alpha=alpha,
    )
    actual_d_user = min(d_user, actual_d)

    if resume:
        print(f"\nResuming from {resume}...")
        state = torch.load(resume, weights_only=True)
        attention.load_state_dict(state["attention_state_dict"])

    optimizer = torch.optim.Adam(attention.parameters(), lr=lr)

    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    print(f"  d_knowledge={actual_d}, d_user={actual_d_user}, "
          f"n_heads={n_heads}, d_k={min(d_k, actual_d)}, d_v={min(d_v, actual_d)}")

    best_loss = float("inf")
    for epoch in range(epochs):
        t0 = time.time()
        epoch_losses = []
        n_skipped = 0

        for i, sample in enumerate(data.samples):
            profile = sample.profile
            if len(profile) < 3:
                n_skipped += 1
                continue

            # Get profile texts and targets
            from hare.evaluation.baselines import get_task_config, _get_profile_target
            cfg = get_task_config(task_key)
            profile_texts = [
                item.get(cfg.profile_text_key, "") for item in profile
            ]
            profile_targets = [
                _get_profile_target(item, cfg) for item in profile
            ]

            # Filter out empty entries
            valid = [
                (t, tgt) for t, tgt in zip(profile_texts, profile_targets)
                if t.strip() and tgt.strip()
            ]
            if len(valid) < 3:
                n_skipped += 1
                continue

            profile_texts_valid = [v[0] for v in valid]
            profile_targets_valid = [v[1] for v in valid]

            # Embed profile and input
            try:
                profile_embs = embedder.encode(profile_texts_valid)
                input_emb = embedder.encode([sample.input_text])[0]
            except Exception:
                n_skipped += 1
                continue

            # Compute oracle relevance
            p_oracle = compute_oracle_relevance(
                profile_targets_valid, sample.target, tau=tau
            )

            # Skip if oracle is degenerate (all zeros or NaN)
            if not np.isfinite(p_oracle).all() or p_oracle.sum() < 1e-8:
                n_skipped += 1
                continue

            # Cluster profile embeddings
            n_items = profile_embs.shape[0]
            n_clust = min(n_clusters, n_items)
            if n_clust < 2:
                n_skipped += 1
                continue

            clusterer = MiniBatchKMeans(
                n_clusters=n_clust, random_state=seed, n_init=3
            )
            cluster_assignments = clusterer.fit_predict(profile_embs)

            # Warm up user state
            user = UserState(actual_d_user)
            k = min(n_warmup, n_items)
            for j in range(k):
                user.update(profile_embs[j], reward=0.8)

            # Convert to tensors
            query_t = torch.tensor(input_emb, dtype=torch.float32)
            user_t = torch.tensor(user.u, dtype=torch.float32)
            keys_t = torch.tensor(profile_embs, dtype=torch.float32)
            values_t = keys_t.clone()
            p_oracle_t = torch.tensor(p_oracle, dtype=torch.float32)

            # Forward pass (differentiable)
            _, weights = attention.forward(
                query_embedding=query_t,
                user_state=user_t,
                keys=keys_t,
                values=values_t,
                cluster_assignments=cluster_assignments,
                return_weights=True,
            )

            # Mean attention across heads
            mean_attn = weights.mean(dim=0)  # (n_items,)

            # Soft cross-entropy loss: -sum p_oracle * log(attn)
            loss = -torch.sum(p_oracle_t * torch.log(mean_attn + 1e-12))

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(attention.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

            if (i + 1) % 200 == 0:
                avg = np.mean(epoch_losses[-200:])
                print(f"    [{i+1}/{len(data)}] loss={avg:.4f}")

        elapsed = time.time() - t0
        avg_loss = np.mean(epoch_losses) if epoch_losses else float("nan")
        print(f"  Epoch {epoch+1}/{epochs}: "
              f"loss={avg_loss:.4f}, "
              f"samples={len(epoch_losses)}, "
              f"skipped={n_skipped}, "
              f"time={elapsed:.1f}s")

        # Save best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            save_path = checkpoint_dir / "attention_weights.pt"
            torch.save({
                "attention_state_dict": attention.state_dict(),
                "epoch": epoch + 1,
                "loss": avg_loss,
                "config": {
                    "d_knowledge": actual_d,
                    "d_user": actual_d_user,
                    "n_heads": n_heads,
                    "d_k": min(d_k, actual_d),
                    "d_v": min(d_v, actual_d),
                    "n_clusters": n_clusters,
                    "alpha": alpha,
                },
            }, save_path)
            print(f"    Saved best checkpoint to {save_path}")

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    return checkpoint_dir / "attention_weights.pt"


def main():
    parser = argparse.ArgumentParser(
        description="Train HARE learnable cross-attention on LaMP or Amazon data"
    )
    parser.add_argument(
        "--task", type=str, default="lamp4",
        help="Task to train on: lamp4, lamp5, lamp7, or amazon_<Category> "
             "(e.g. amazon_Digital_Music). Default: lamp4.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--n-warmup", type=int, default=5)
    parser.add_argument("--tau", type=float, default=0.3)
    parser.add_argument("--d-knowledge", type=int, default=64)
    parser.add_argument("--d-user", type=int, default=32)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-k", type=int, default=32)
    parser.add_argument("--d-v", type=int, default=32)
    parser.add_argument("--n-clusters", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path("checkpoints"),
    )
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    train_attention(
        task=args.task,
        epochs=args.epochs,
        lr=args.lr,
        max_samples=args.max_samples,
        n_warmup=args.n_warmup,
        tau=args.tau,
        d_knowledge=args.d_knowledge,
        d_user=args.d_user,
        n_heads=args.n_heads,
        d_k=args.d_k,
        d_v=args.d_v,
        n_clusters=args.n_clusters,
        alpha=args.alpha,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
