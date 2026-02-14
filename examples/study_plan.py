#!/usr/bin/env python3
"""HARE demo: Study plan distillation.

Demonstrates HARE's learning curve -- how specificity emerges over
multiple interactions as the system learns about a student's needs.

Tracks attention entropy (decreasing = more specific) and cumulative
reward (increasing = better personalization) over a session.

Usage:
    python examples/study_plan.py                    # auto-detect best backend
    python examples/study_plan.py --backend tfidf    # lightweight, no GPU
    python examples/study_plan.py --backend st       # sentence-transformers (384-d)
"""

import argparse

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from hare.bandits.attentive_bandit import HARE
from hare.bandits.linucb import LinUCB
from hare.utils.embeddings import TfidfEmbedder, SentenceTransformerEmbedder

# -- Sample course materials (knowledge pool) ---------------------------------

MATERIALS = [
    "Linear algebra fundamentals: vectors, matrices, eigenvalues, SVD decomposition",
    "Probability theory: Bayes theorem, distributions, expectations, variance",
    "Optimization: gradient descent, convex optimization, Lagrange multipliers",
    "Machine learning basics: regression, classification, bias-variance tradeoff",
    "Neural networks: backpropagation, activation functions, loss functions",
    "Deep learning: CNNs, RNNs, transformers, attention mechanisms",
    "Natural language processing: tokenization, embeddings, language models",
    "Reinforcement learning: MDPs, Q-learning, policy gradient, bandits",
    "Statistical inference: hypothesis testing, confidence intervals, p-values",
    "Information theory: entropy, mutual information, KL divergence",
    "Computer vision: image classification, object detection, segmentation",
    "Generative models: VAEs, GANs, diffusion models, flow models",
]


def build_embedder(backend: str, corpus: list[str]):
    """Build and fit an embedder based on the selected backend."""
    if backend == "st":
        return SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")
    else:
        embedder = TfidfEmbedder(max_features=300, output_dim=24)
        embedder.fit(corpus)
        return embedder


def detect_backend() -> str:
    try:
        import sentence_transformers  # noqa: F401
        return "st"
    except ImportError:
        return "tfidf"


def main():
    parser = argparse.ArgumentParser(description="HARE Study Plan Demo")
    parser.add_argument(
        "--backend", choices=["tfidf", "st"], default=None,
        help="Embedding backend: 'tfidf' or 'st' (sentence-transformers)."
    )
    args = parser.parse_args()

    backend = args.backend or detect_backend()
    print(f"Embedding backend: {backend}\n")

    # -- Embed knowledge pool -------------------------------------------------
    embedder = build_embedder(backend, MATERIALS)
    material_embs = embedder.encode(MATERIALS)
    d = material_embs.shape[1]

    print(f"Knowledge pool: {len(MATERIALS)} topics, {d}-dim embeddings\n")

    # -- Initialize HARE ------------------------------------------------------
    d_user = min(128, d)
    d_k = min(96, d)
    d_v = min(96, d)

    hare = HARE(
        d_knowledge=d,
        d_user=d_user,
        n_clusters=min(4, len(MATERIALS)),
        n_heads=4,
        d_k=d_k,
        d_v=d_v,
        alpha=3.0,
        seed=42,
    )
    hare.set_knowledge_pool(material_embs)

    # -- Simulate a student interested in NLP ---------------------------------
    nlp_topics = [
        "natural language processing and transformers",
        "attention mechanisms and language models",
        "text embeddings and tokenization",
    ]
    nlp_embs = embedder.encode(nlp_topics)
    true_preference = np.mean(nlp_embs, axis=0)

    def reward_fn(synthesis, query, user_state):
        """Reward = cosine similarity to true NLP preference."""
        s = synthesis[:d]
        sim = cosine_similarity(s.reshape(1, -1), true_preference.reshape(1, -1))[0, 0]
        return float(np.clip((sim + 1) / 2, 0, 1))

    # -- Longer session (20 queries) for clearer trends -----------------------
    queries = [
        "I need to study for my AI course",
        "What should I focus on for deep learning?",
        "Help me understand attention and transformers",
        "I want to learn about language models",
        "How do word embeddings work?",
        "Explain self-attention mechanisms",
        "What's the connection between bandits and exploration in NLP?",
        "I need to understand BERT and GPT architectures",
        "How does tokenization affect model performance?",
        "Explain the transformer encoder-decoder architecture",
        "What are positional encodings?",
        "How do masked language models work?",
        "Explain beam search and decoding strategies",
        "What is the difference between GPT and BERT?",
        "How do you fine-tune a language model?",
        "What are attention heads and what do they learn?",
        "Explain cross-attention vs self-attention",
        "How does RLHF work for language models?",
        "What are the latest advances in NLP?",
        "How to build a retrieval-augmented generation system?",
    ]
    query_embs = embedder.encode(queries)

    # -- Run HARE session -----------------------------------------------------
    print("=" * 75)
    print("HARE SESSION -- Specificity Emergence Over Interactions")
    print("=" * 75)

    hare_results = hare.simulate_session(query_embs, reward_fn, user_id="student_nlp")

    print(f"\n{'Round':<6} {'Query':<52} {'Reward':<8} {'Entropy':<9} {'Uncert':<8}")
    print("-" * 83)
    for t in range(len(queries)):
        print(
            f"{t+1:<6} {queries[t][:50]:<52} "
            f"{hare_results['rewards'][t]:<8.3f} "
            f"{hare_results['entropies'][t]:<9.3f} "
            f"{hare_results['uncertainties'][t]:<8.3f}"
        )

    print(f"\nCumulative reward: {hare_results['cumulative_reward'][-1]:.3f}")
    first5_ent = np.mean(hare_results["entropies"][:5])
    last5_ent = np.mean(hare_results["entropies"][-5:])
    print(f"Entropy (first 5 avg):  {first5_ent:.3f}")
    print(f"Entropy (last 5 avg):   {last5_ent:.3f}  {'(decreased -- more specific!)' if last5_ent < first5_ent else ''}")
    print(f"Uncertainty trend:      {hare_results['uncertainties'][0]:.3f} -> {hare_results['uncertainties'][-1]:.3f}")

    # -- Compare with LinUCB baseline -----------------------------------------
    print(f"\n{'=' * 75}")
    print("LinUCB BASELINE -- Selection Only (no synthesis)")
    print("=" * 75)

    bandit = LinUCB(n_arms=len(MATERIALS), d=d, alpha=2.0)
    linucb_rewards = []

    for t in range(len(queries)):
        arm = bandit.select_arm(query_embs[t])
        sim = cosine_similarity(
            material_embs[arm].reshape(1, -1),
            true_preference.reshape(1, -1),
        )[0, 0]
        reward = float(np.clip((sim + 1) / 2, 0, 1))
        bandit.update(arm, query_embs[t], reward)
        linucb_rewards.append(reward)
        if t < 5 or t >= len(queries) - 3:
            print(f"  Round {t+1:>2}: '{MATERIALS[arm][:55]}...' -> reward={reward:.3f}")
        elif t == 5:
            print(f"  ... ({len(queries) - 8} more rounds) ...")

    linucb_cumulative = np.cumsum(linucb_rewards)

    # -- Summary --------------------------------------------------------------
    print(f"\n{'=' * 75}")
    print("COMPARISON")
    print("=" * 75)
    print(f"  HARE cumulative reward:    {hare_results['cumulative_reward'][-1]:.3f}")
    print(f"  LinUCB cumulative reward:  {linucb_cumulative[-1]:.3f}")
    print(f"  HARE avg reward (last 5):  {np.mean(hare_results['rewards'][-5:]):.3f}")
    print(f"  LinUCB avg reward (last 5):{np.mean(linucb_rewards[-5:]):.3f}")
    print(f"\n  HARE synthesizes personalized content; LinUCB can only select existing items.")
    print(f"  HARE's entropy decreases over time = emergent specificity.")


if __name__ == "__main__":
    main()
