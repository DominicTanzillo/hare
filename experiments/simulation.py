#!/usr/bin/env python3
"""HARE vs baselines: simulated environment experiments.

Runs a controlled simulation comparing:
1. HARE          - full algorithm (user state + uncertainty-augmented attention)
2. HARE-nouser   - ablation: no user state modeling (alpha > 0, no u_t)
3. HARE-noexplore - ablation: no exploration bonus (alpha = 0, u_t present)
4. LinUCB        - contextual bandit baseline (selection only)
5. RAG-cosine    - top-k cosine retrieval, no exploration or user modeling
6. Random        - uniform random selection

Metrics tracked per method:
- Cumulative reward
- Per-round reward (learning curve)
- Personalization divergence (different users, same query)
- Attention entropy trajectory (HARE variants only)
- User uncertainty trajectory (HARE variants only)

The simulated environment has:
- A known non-linear reward function (bilinear + interaction terms)
- Distinct user types with different latent preferences
- A knowledge pool of synthetic items with cluster structure

Usage:
    python experiments/simulation.py
    python experiments/simulation.py --rounds 200 --n-users 5 --seed 42
"""

import argparse
from dataclasses import dataclass

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from hare.bandits.attentive_bandit import HARE
from hare.bandits.linucb import LinUCB


@dataclass
class SimConfig:
    """Simulation configuration."""
    n_items: int = 30           # Knowledge pool size
    d: int = 32                 # Embedding dimension
    n_clusters: int = 5         # Item clusters
    n_users: int = 3            # Distinct user types
    rounds: int = 100           # Interactions per user
    seed: int = 42
    alpha_hare: float = 2.0     # Exploration parameter for HARE
    alpha_linucb: float = 1.5   # Exploration parameter for LinUCB


def generate_environment(cfg: SimConfig):
    """Create a synthetic environment with known reward structure.

    Returns
    -------
    dict with:
        item_embeddings: (n_items, d)
        user_preferences: (n_users, d) -- hidden latent preferences
        query_sequences: (n_users, rounds, d)
        reward_fn: callable(synthesis, query, user_pref) -> float
    """
    rng = np.random.default_rng(cfg.seed)

    # Generate clustered item embeddings
    cluster_centers = rng.normal(0, 1, (cfg.n_clusters, cfg.d))
    items_per_cluster = cfg.n_items // cfg.n_clusters
    item_embeddings = []
    for center in cluster_centers:
        items = center + rng.normal(0, 0.3, (items_per_cluster, cfg.d))
        item_embeddings.append(items)
    item_embeddings = np.vstack(item_embeddings)[:cfg.n_items]
    # Normalize
    item_embeddings = item_embeddings / np.linalg.norm(item_embeddings, axis=1, keepdims=True)

    # Generate distinct user preferences (each user "cares about" different clusters)
    user_preferences = []
    for u in range(cfg.n_users):
        # Each user is a weighted combo of 1-2 cluster centers
        primary = u % cfg.n_clusters
        secondary = (u + 2) % cfg.n_clusters
        pref = 0.7 * cluster_centers[primary] + 0.3 * cluster_centers[secondary]
        pref = pref / np.linalg.norm(pref)
        user_preferences.append(pref)
    user_preferences = np.array(user_preferences)

    # Generate query sequences (slightly noisy versions of user preferences)
    query_sequences = np.zeros((cfg.n_users, cfg.rounds, cfg.d))
    for u in range(cfg.n_users):
        for t in range(cfg.rounds):
            noise = rng.normal(0, 0.2, cfg.d)
            q = user_preferences[u] + noise
            q = q / np.linalg.norm(q)
            query_sequences[u, t] = q

    # Non-linear reward function: cosine + interaction term
    # The interaction term means linear models can't capture the full reward
    interaction_matrix = rng.normal(0, 0.1, (cfg.d, cfg.d))
    interaction_matrix = (interaction_matrix + interaction_matrix.T) / 2  # Symmetric

    def reward_fn(synthesis, query, user_pref):
        """Non-linear reward: cosine(synth, pref) + bilinear interaction."""
        s = synthesis[:cfg.d]  # Use knowledge dimensions
        s_norm = np.linalg.norm(s)
        if s_norm > 1e-8:
            s = s / s_norm

        # Linear term: cosine similarity to preference
        linear = float(cosine_similarity(s.reshape(1, -1), user_pref.reshape(1, -1))[0, 0])

        # Non-linear interaction: s^T M q (captures cross-feature effects)
        interaction = float(s @ interaction_matrix @ query[:cfg.d]) * 0.3

        reward = (linear + interaction + 1) / 2  # Normalize to [0, 1]-ish
        return float(np.clip(reward, 0, 1))

    return {
        "item_embeddings": item_embeddings,
        "user_preferences": user_preferences,
        "query_sequences": query_sequences,
        "reward_fn": reward_fn,
    }


def run_hare(env, cfg: SimConfig, alpha: float, use_user_state: bool, label: str):
    """Run HARE (or ablation) on the simulated environment.

    Reward is computed from attention weights over the knowledge pool:
    the attention mechanism determines WHICH items to attend to, and
    the reward reflects the quality of that attention-weighted blend
    in the ORIGINAL item embedding space. This separates the attention
    mechanism's quality from the random projection artifacts.
    """
    d = cfg.d
    d_user = 16 if use_user_state else 1

    hare = HARE(
        d_knowledge=d,
        d_user=d_user,
        n_clusters=cfg.n_clusters,
        n_heads=4,
        d_k=min(32, d),
        d_v=min(32, d),
        alpha=alpha,
        seed=cfg.seed,
    )
    hare.set_knowledge_pool(env["item_embeddings"])

    all_rewards = np.zeros((cfg.n_users, cfg.rounds))
    all_entropies = np.zeros((cfg.n_users, cfg.rounds))
    all_uncertainties = np.zeros((cfg.n_users, cfg.rounds))

    for u in range(cfg.n_users):
        user_id = f"user_{u}"
        user_pref = env["user_preferences"][u]
        items = env["item_embeddings"]

        for t in range(cfg.rounds):
            query = env["query_sequences"][u, t]
            result = hare.recommend(query, user_id, return_details=True)

            # Compute reward from attention-weighted blend of ORIGINAL items
            weights = np.mean(result["attention_weights"], axis=0)  # avg across heads
            blend = weights @ items  # attention-weighted blend in original space
            reward = env["reward_fn"](blend, query, user_pref)

            hare.update(query, user_id, reward, synthesis=result["synthesis"])

            all_rewards[u, t] = reward
            all_entropies[u, t] = result["attention_entropy"]
            all_uncertainties[u, t] = result["user_uncertainty"]

    return {
        "label": label,
        "rewards": all_rewards,
        "mean_reward": np.mean(all_rewards, axis=0),
        "cumulative_reward": np.cumsum(np.mean(all_rewards, axis=0)),
        "mean_entropy": np.mean(all_entropies, axis=0),
        "mean_uncertainty": np.mean(all_uncertainties, axis=0),
    }


def run_linucb(env, cfg: SimConfig):
    """Run LinUCB baseline (selection only)."""
    d = cfg.d
    all_rewards = np.zeros((cfg.n_users, cfg.rounds))

    for u in range(cfg.n_users):
        bandit = LinUCB(n_arms=cfg.n_items, d=d, alpha=cfg.alpha_linucb)
        user_pref = env["user_preferences"][u]

        for t in range(cfg.rounds):
            query = env["query_sequences"][u, t]
            arm = bandit.select_arm(query)
            selected_item = env["item_embeddings"][arm]
            reward = env["reward_fn"](selected_item, query, user_pref)
            bandit.update(arm, query, reward)
            all_rewards[u, t] = reward

    return {
        "label": "LinUCB",
        "rewards": all_rewards,
        "mean_reward": np.mean(all_rewards, axis=0),
        "cumulative_reward": np.cumsum(np.mean(all_rewards, axis=0)),
    }


def run_rag_cosine(env, cfg: SimConfig, top_k: int = 3):
    """Run RAG baseline (top-k cosine retrieval, no exploration, no user model)."""
    d = cfg.d
    all_rewards = np.zeros((cfg.n_users, cfg.rounds))

    for u in range(cfg.n_users):
        user_pref = env["user_preferences"][u]

        for t in range(cfg.rounds):
            query = env["query_sequences"][u, t]
            # RAG: top-k by cosine similarity, average as "synthesis"
            sims = cosine_similarity(query.reshape(1, -1), env["item_embeddings"])[0]
            top_idx = np.argsort(sims)[-top_k:]
            synthesis = np.mean(env["item_embeddings"][top_idx], axis=0)
            reward = env["reward_fn"](synthesis, query, user_pref)
            all_rewards[u, t] = reward

    return {
        "label": "RAG-cosine",
        "rewards": all_rewards,
        "mean_reward": np.mean(all_rewards, axis=0),
        "cumulative_reward": np.cumsum(np.mean(all_rewards, axis=0)),
    }


def run_random(env, cfg: SimConfig):
    """Run random baseline."""
    rng = np.random.default_rng(cfg.seed + 999)
    d = cfg.d
    all_rewards = np.zeros((cfg.n_users, cfg.rounds))

    for u in range(cfg.n_users):
        user_pref = env["user_preferences"][u]
        for t in range(cfg.rounds):
            query = env["query_sequences"][u, t]
            arm = rng.integers(0, cfg.n_items)
            selected = env["item_embeddings"][arm]
            reward = env["reward_fn"](selected, query, user_pref)
            all_rewards[u, t] = reward

    return {
        "label": "Random",
        "rewards": all_rewards,
        "mean_reward": np.mean(all_rewards, axis=0),
        "cumulative_reward": np.cumsum(np.mean(all_rewards, axis=0)),
    }


def personalization_divergence(env, cfg: SimConfig, hare_instance: HARE):
    """Measure: do different users get different outputs for the same query?

    Returns average pairwise cosine distance of synthesis vectors
    across users for the same query.
    """
    rng = np.random.default_rng(cfg.seed)
    # Use 10 shared queries
    shared_queries = rng.normal(0, 1, (10, cfg.d))
    shared_queries = shared_queries / np.linalg.norm(shared_queries, axis=1, keepdims=True)

    distances = []
    for q in shared_queries:
        syntheses = []
        for u in range(cfg.n_users):
            z = hare_instance.recommend(q, user_id=f"user_{u}")
            syntheses.append(z)
        syntheses = np.array(syntheses)
        # Pairwise cosine similarity
        sim_matrix = cosine_similarity(syntheses)
        # Mean off-diagonal (lower = more personalized)
        n = sim_matrix.shape[0]
        off_diag = sim_matrix[np.triu_indices(n, k=1)]
        distances.append(1 - np.mean(off_diag))  # Convert similarity to distance

    return float(np.mean(distances))


def main():
    parser = argparse.ArgumentParser(description="HARE Simulation Experiments")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--n-users", type=int, default=3)
    parser.add_argument("--n-items", type=int, default=30)
    parser.add_argument("--d", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = SimConfig(
        n_items=args.n_items,
        d=args.d,
        n_users=args.n_users,
        rounds=args.rounds,
        seed=args.seed,
    )

    print(f"Simulation: {cfg.n_items} items, {cfg.d}-dim, {cfg.n_users} users, {cfg.rounds} rounds\n")

    # Generate environment
    env = generate_environment(cfg)

    # Run all methods
    print("Running experiments...")
    results = {}

    results["HARE"] = run_hare(env, cfg, alpha=cfg.alpha_hare, use_user_state=True, label="HARE")
    print(f"  HARE:           cumulative={results['HARE']['cumulative_reward'][-1]:.2f}")

    results["HARE-nouser"] = run_hare(env, cfg, alpha=cfg.alpha_hare, use_user_state=False, label="HARE-nouser")
    print(f"  HARE-nouser:    cumulative={results['HARE-nouser']['cumulative_reward'][-1]:.2f}")

    results["HARE-noexplore"] = run_hare(env, cfg, alpha=0.0, use_user_state=True, label="HARE-noexplore")
    print(f"  HARE-noexplore: cumulative={results['HARE-noexplore']['cumulative_reward'][-1]:.2f}")

    results["LinUCB"] = run_linucb(env, cfg)
    print(f"  LinUCB:         cumulative={results['LinUCB']['cumulative_reward'][-1]:.2f}")

    results["RAG"] = run_rag_cosine(env, cfg)
    print(f"  RAG-cosine:     cumulative={results['RAG']['cumulative_reward'][-1]:.2f}")

    results["Random"] = run_random(env, cfg)
    print(f"  Random:         cumulative={results['Random']['cumulative_reward'][-1]:.2f}")

    # -- Detailed results table -----------------------------------------------
    print(f"\n{'=' * 80}")
    print("CUMULATIVE REWARD COMPARISON")
    print(f"{'=' * 80}")
    print(f"\n{'Method':<20} {'Final Cumul.':<14} {'Avg Last 10':<14} {'Avg First 10':<14}")
    print("-" * 62)
    for name in ["HARE", "HARE-nouser", "HARE-noexplore", "LinUCB", "RAG", "Random"]:
        r = results[name]
        final = r["cumulative_reward"][-1]
        last10 = np.mean(r["mean_reward"][-10:])
        first10 = np.mean(r["mean_reward"][:10])
        print(f"{r['label']:<20} {final:<14.3f} {last10:<14.3f} {first10:<14.3f}")

    # -- Entropy analysis (HARE variants only) --------------------------------
    print(f"\n{'=' * 80}")
    print("SPECIFICITY EMERGENCE (Entropy)")
    print(f"{'=' * 80}")
    for name in ["HARE", "HARE-nouser", "HARE-noexplore"]:
        r = results[name]
        if "mean_entropy" in r:
            first5 = np.mean(r["mean_entropy"][:5])
            last5 = np.mean(r["mean_entropy"][-5:])
            delta = last5 - first5
            direction = "decreased (more specific)" if delta < 0 else "increased or flat"
            print(f"  {r['label']:<20} entropy: {first5:.4f} -> {last5:.4f} ({direction})")

    # -- Uncertainty analysis -------------------------------------------------
    print(f"\n{'=' * 80}")
    print("USER UNCERTAINTY REDUCTION")
    print(f"{'=' * 80}")
    for name in ["HARE", "HARE-nouser", "HARE-noexplore"]:
        r = results[name]
        if "mean_uncertainty" in r:
            first = r["mean_uncertainty"][0]
            last = r["mean_uncertainty"][-1]
            pct = (first - last) / first * 100
            print(f"  {r['label']:<20} uncertainty: {first:.2f} -> {last:.2f} ({pct:.1f}% reduction)")

    # -- Learning curve (reward trend) ----------------------------------------
    print(f"\n{'=' * 80}")
    print("LEARNING CURVES (avg reward per 10-round window)")
    print(f"{'=' * 80}")
    windows = cfg.rounds // 10
    print(f"\n{'Method':<20}", end="")
    for w in range(windows):
        print(f"{'R' + str(w*10+1) + '-' + str((w+1)*10):<10}", end="")
    print()
    print("-" * (20 + windows * 10))
    for name in ["HARE", "HARE-nouser", "HARE-noexplore", "LinUCB", "RAG", "Random"]:
        r = results[name]
        print(f"{r['label']:<20}", end="")
        for w in range(windows):
            window_avg = np.mean(r["mean_reward"][w*10:(w+1)*10])
            print(f"{window_avg:<10.3f}", end="")
        print()

    print(f"\n{'=' * 80}")
    print("ANALYSIS")
    print(f"{'=' * 80}")
    # Hypothesis checks
    hare_final = results["HARE"]["cumulative_reward"][-1]
    linucb_final = results["LinUCB"]["cumulative_reward"][-1]
    rag_final = results["RAG"]["cumulative_reward"][-1]
    nouser_final = results["HARE-nouser"]["cumulative_reward"][-1]
    noexplore_final = results["HARE-noexplore"]["cumulative_reward"][-1]

    print(f"  H1 (HARE > LinUCB):      {'YES' if hare_final > linucb_final else 'NO'} "
          f"({hare_final:.2f} vs {linucb_final:.2f})")
    print(f"  H2 (HARE > RAG):         {'YES' if hare_final > rag_final else 'NO'} "
          f"({hare_final:.2f} vs {rag_final:.2f})")
    print(f"  H3 (explore helps):      {'YES' if hare_final > noexplore_final else 'NO'} "
          f"({hare_final:.2f} vs {noexplore_final:.2f})")
    print(f"  H4 (user state matters): {'YES' if hare_final > nouser_final else 'NO'} "
          f"({hare_final:.2f} vs {nouser_final:.2f})")

    # Entropy decreasing
    hare_ent = results["HARE"]["mean_entropy"]
    ent_decreased = np.mean(hare_ent[-5:]) < np.mean(hare_ent[:5])
    print(f"  H5 (entropy decreases):  {'YES' if ent_decreased else 'NO'}")


if __name__ == "__main__":
    main()
