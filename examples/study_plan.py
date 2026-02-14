#!/usr/bin/env python3
"""HARE demo: Study plan distillation.

Demonstrates HARE's learning curve -- how specificity emerges over
multiple interactions as the system learns about a student's needs.

Tracks attention entropy (decreasing = more specific) and cumulative
reward (increasing = better personalization) over a session.
"""

import numpy as np

from hare.bandits.attentive_bandit import HARE
from hare.bandits.linucb import LinUCB
from hare.utils.embeddings import TfidfEmbedder

# ── Sample course materials (knowledge pool) ───────────────────────────

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


def main():
    # ── Embed knowledge pool ────────────────────────────────────────────
    embedder = TfidfEmbedder(max_features=300, output_dim=24)
    embedder.fit(MATERIALS)
    material_embs = embedder.encode(MATERIALS)

    d = embedder.dim
    print(f"Knowledge pool: {len(MATERIALS)} topics, {d}-dim embeddings\n")

    # ── Initialize HARE ─────────────────────────────────────────────────
    hare = HARE(
        d_knowledge=d,
        d_user=16,
        n_clusters=min(4, len(MATERIALS)),
        n_heads=2,
        d_k=24,
        d_v=24,
        alpha=1.5,
        seed=42,
    )
    hare.set_knowledge_pool(material_embs)

    # ── Simulate a student interested in NLP ────────────────────────────
    # The "true preference" is a hidden vector -- the student cares about
    # NLP-related topics. HARE doesn't know this; it must discover it.

    # Encode what the student actually wants (hidden from HARE)
    nlp_topics = [
        "natural language processing and transformers",
        "attention mechanisms and language models",
        "text embeddings and tokenization",
    ]
    nlp_embs = embedder.encode(nlp_topics)
    true_preference = np.mean(nlp_embs, axis=0)  # Average NLP embedding

    def reward_fn(synthesis, query, user_state):
        """Reward = cosine similarity to true NLP preference."""
        from sklearn.metrics.pairwise import cosine_similarity
        # Use only the knowledge dimensions of synthesis
        s = synthesis[:d]
        sim = cosine_similarity(s.reshape(1, -1), true_preference.reshape(1, -1))[0, 0]
        # Clamp to [0, 1]
        return float(np.clip((sim + 1) / 2, 0, 1))

    # ── Generate queries over a session ─────────────────────────────────
    # Student asks progressively more specific questions
    queries = [
        "I need to study for my AI course",
        "What should I focus on for deep learning?",
        "Help me understand attention and transformers",
        "I want to learn about language models",
        "How do word embeddings work?",
        "Explain self-attention mechanisms",
        "What's the connection between bandits and exploration in NLP?",
        "I need to understand BERT and GPT architectures",
    ]
    query_embs = embedder.encode(queries)

    # ── Run HARE session ────────────────────────────────────────────────
    print("=" * 65)
    print("HARE SESSION -- Specificity Emergence Over Interactions")
    print("=" * 65)

    hare_results = hare.simulate_session(query_embs, reward_fn, user_id="student_nlp")

    print(f"\n{'Round':<6} {'Query':<50} {'Reward':<8} {'Entropy':<9} {'Uncert':<8}")
    print("-" * 81)
    for t in range(len(queries)):
        print(
            f"{t+1:<6} {queries[t][:48]:<50} "
            f"{hare_results['rewards'][t]:<8.3f} "
            f"{hare_results['entropies'][t]:<9.3f} "
            f"{hare_results['uncertainties'][t]:<8.3f}"
        )

    print(f"\nCumulative reward: {hare_results['cumulative_reward'][-1]:.3f}")
    print(f"Entropy trend:     {hare_results['entropies'][0]:.3f} -> {hare_results['entropies'][-1]:.3f} (should decrease)")
    print(f"Uncertainty trend:  {hare_results['uncertainties'][0]:.3f} -> {hare_results['uncertainties'][-1]:.3f} (should decrease)")

    # ── Compare with LinUCB baseline ────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("LinUCB BASELINE -- Selection Only (no synthesis)")
    print("=" * 65)

    bandit = LinUCB(n_arms=len(MATERIALS), d=d, alpha=1.5)
    linucb_rewards = []

    for t in range(len(queries)):
        arm = bandit.select_arm(query_embs[t])
        # Reward: similarity of selected material to true preference
        from sklearn.metrics.pairwise import cosine_similarity
        sim = cosine_similarity(
            material_embs[arm].reshape(1, -1),
            true_preference.reshape(1, -1),
        )[0, 0]
        reward = float(np.clip((sim + 1) / 2, 0, 1))
        bandit.update(arm, query_embs[t], reward)
        linucb_rewards.append(reward)
        print(f"  Round {t+1}: Selected '{MATERIALS[arm][:50]}...' -> reward={reward:.3f}")

    linucb_cumulative = np.cumsum(linucb_rewards)

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("COMPARISON")
    print("=" * 65)
    print(f"  HARE cumulative reward:   {hare_results['cumulative_reward'][-1]:.3f}")
    print(f"  LinUCB cumulative reward: {linucb_cumulative[-1]:.3f}")
    print(f"  HARE final reward:        {hare_results['rewards'][-1]:.3f}")
    print(f"  LinUCB final reward:      {linucb_rewards[-1]:.3f}")
    print(f"\n  HARE synthesizes personalized content; LinUCB can only select existing items.")
    print(f"  HARE's entropy decreases over time = emergent specificity.")


if __name__ == "__main__":
    main()
