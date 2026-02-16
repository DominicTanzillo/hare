#!/usr/bin/env python3
"""HARE 5-User Divergence Demo -- the killer experiment.

Shows that 5 users with different interaction histories receive
meaningfully different outputs for the SAME query.

User profiles:
    1. Security Expert     -- 20 security-focused interactions
    2. Data Engineer       -- 20 data pipeline interactions
    3. Frontend Developer  -- 20 frontend/UI interactions
    4. Educator            -- 20 teaching/learning interactions
    5. DevOps Engineer     -- 20 infrastructure interactions

All 5 users then ask: "I need help writing tests"

HARE generates 5 different synthesis vectors -> 5 different skill blends.
RAG generates 1 blend (identical for all users).

This is the paper's Figure 2.

Usage:
    python experiments/divergence_demo.py
    python experiments/divergence_demo.py --backend tfidf
    python experiments/divergence_demo.py --output results/divergence.json
    python experiments/divergence_demo.py --learnable --attention-checkpoint checkpoints/attention_weights.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from hare.bandits.attentive_bandit import HARE
from hare.data.skills import load_builtin_skills, skill_to_text
from hare.synthesis.generator import InterpolationDecoder
from hare.utils.embeddings import TfidfEmbedder


# -- User profiles: training queries that shape each user's latent state ------

USER_PROFILES = {
    "security_expert": {
        "label": "Security Expert",
        "color": "#e74c3c",
        "training_queries": [
            "find security vulnerabilities in my code",
            "audit my dependencies for CVEs",
            "check for injection attacks and XSS",
            "scan for OWASP top 10 issues",
            "review authentication flow for weaknesses",
            "check for SQL injection in database queries",
            "verify input sanitization across endpoints",
            "analyze CORS and CSP security headers",
            "review secret management and credential storage",
            "check for insecure deserialization patterns",
            "detect cross-site scripting in templates",
            "review authorization and access control logic",
            "check for path traversal vulnerabilities",
            "audit cryptographic implementations",
            "scan for hardcoded secrets and API keys",
            "review session management security",
            "check for server-side request forgery",
            "analyze rate limiting and brute force protection",
            "review error handling for information leakage",
            "check for insecure direct object references",
        ],
    },
    "data_engineer": {
        "label": "Data Engineer",
        "color": "#3498db",
        "training_queries": [
            "build a data pipeline from S3 to Snowflake",
            "optimize my SQL queries for large datasets",
            "create a data validation schema for incoming records",
            "set up Apache Airflow DAGs for ETL",
            "design a data warehouse star schema",
            "write data quality checks for the pipeline",
            "migrate data from PostgreSQL to BigQuery",
            "build a streaming pipeline with Kafka",
            "create dbt models for analytics",
            "optimize Spark job for memory efficiency",
            "set up data lineage tracking",
            "build an incremental load pipeline",
            "create a CDC pipeline for real-time sync",
            "design a data lake partitioning strategy",
            "write Great Expectations data validation suite",
            "optimize parquet file compression",
            "build a feature store for ML models",
            "set up data catalog with metadata",
            "create data access policies and governance",
            "monitor pipeline SLAs and alert on failures",
        ],
    },
    "frontend_dev": {
        "label": "Frontend Developer",
        "color": "#2ecc71",
        "training_queries": [
            "build a responsive React component",
            "fix CSS layout issues with flexbox",
            "implement dark mode with CSS variables",
            "optimize React rendering performance",
            "add keyboard navigation accessibility",
            "build a drag and drop interface",
            "implement infinite scroll with virtualization",
            "fix TypeScript type errors in components",
            "build a form with real-time validation",
            "create smooth CSS transitions and animations",
            "implement a design system with Storybook",
            "optimize bundle size with code splitting",
            "build a progressive web app with service workers",
            "add ARIA labels for screen reader support",
            "implement state management with Redux",
            "build a reusable modal component",
            "fix cross-browser compatibility issues",
            "implement lazy loading for images",
            "build a responsive navigation menu",
            "add end-to-end tests with Playwright",
        ],
    },
    "educator": {
        "label": "Educator",
        "color": "#9b59b6",
        "training_queries": [
            "create a lesson plan for teaching Python basics",
            "design a rubric for grading programming assignments",
            "build interactive coding exercises for students",
            "create a quiz on data structures",
            "design a curriculum for web development bootcamp",
            "write learning objectives for a CS course",
            "create scaffolded programming exercises",
            "design peer review activities for code",
            "build a study guide for algorithms exam",
            "create analogies to explain recursion",
            "design a flipped classroom activity on databases",
            "write assessment criteria for group projects",
            "create debugging exercises for beginners",
            "design a capstone project specification",
            "build a reading list for software engineering",
            "create lab activities for version control",
            "design differentiated assignments for mixed levels",
            "write feedback templates for code reviews",
            "create a syllabus for intro to AI course",
            "design collaborative coding activities",
        ],
    },
    "devops_engineer": {
        "label": "DevOps Engineer",
        "color": "#f39c12",
        "training_queries": [
            "set up a CI/CD pipeline with GitHub Actions",
            "write a Dockerfile for a Python application",
            "configure Kubernetes deployment and services",
            "set up monitoring with Prometheus and Grafana",
            "automate infrastructure with Terraform",
            "configure nginx as a reverse proxy",
            "set up log aggregation with ELK stack",
            "implement blue-green deployment strategy",
            "configure auto-scaling for cloud instances",
            "set up secrets management with Vault",
            "write Ansible playbooks for server config",
            "implement infrastructure as code best practices",
            "set up container orchestration with Docker Compose",
            "configure SSL certificates with Let's Encrypt",
            "implement canary deployment with feature flags",
            "set up alerting rules for SLO violations",
            "optimize Docker image size and build time",
            "configure network policies in Kubernetes",
            "set up disaster recovery and backup strategy",
            "implement GitOps workflow with ArgoCD",
        ],
    },
}

# The query all 5 users will share
SHARED_QUERY = "I need help writing tests"


def run_divergence_experiment(
    backend: str = "tfidf",
    learnable: bool = False,
    attention_checkpoint: str | None = None,
) -> dict:
    """Run the 5-user divergence experiment.

    Parameters
    ----------
    backend : str
        Embedding backend ("tfidf" or "st").
    learnable : bool
        If True, use LearnableHARE with trained attention projections.
    attention_checkpoint : str or None
        Path to trained attention weights. Required if learnable=True.

    Returns a results dict suitable for JSON serialization and paper figures.
    """
    # Load skills and embed
    skills = load_builtin_skills()
    skill_texts = [skill_to_text(s) for s in skills]

    # Collect ALL training queries so TF-IDF vocabulary includes user-profile terms
    all_training_queries = []
    for profile in USER_PROFILES.values():
        all_training_queries.extend(profile["training_queries"])

    if backend == "st":
        from hare.utils.embeddings import SentenceTransformerEmbedder
        embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")
        skill_embeddings = embedder.encode(skill_texts)
    else:
        # Fit TF-IDF on the COMBINED corpus (skills + all training queries)
        # so the vocabulary captures domain-specific terms from user profiles
        combined_corpus = skill_texts + all_training_queries + [SHARED_QUERY]
        embedder = TfidfEmbedder(max_features=1000, output_dim=64)
        embedder.fit(combined_corpus)
        skill_embeddings = embedder.encode(skill_texts)

    d_knowledge = skill_embeddings.shape[1]
    d_user = min(64, d_knowledge)

    mode = "LearnableHARE" if learnable else "HARE (random projections)"
    print(f"Knowledge pool: {len(skills)} skills, {d_knowledge}-dim embeddings")
    print(f"User state dim: {d_user}")
    print(f"Embedding backend: {backend}")
    print(f"Mode: {mode}\n")

    # Initialize HARE
    if learnable:
        import torch
        from hare.bandits.learnable_hare import LearnableHARE

        hare = LearnableHARE(
            d_knowledge=d_knowledge,
            d_user=d_user,
            n_clusters=min(5, len(skills)),
            n_heads=4,
            d_k=min(64, d_knowledge),
            d_v=min(64, d_knowledge),
            alpha=2.0,
            seed=42,
        )
        if attention_checkpoint:
            state = torch.load(attention_checkpoint, weights_only=False)
            hare.attention.load_state_dict(
                state["attention_state_dict"], strict=False
            )
            print(f"Loaded attention weights from {attention_checkpoint}")
        hare.set_knowledge_pool(skill_embeddings)
    else:
        hare = HARE(
            d_knowledge=d_knowledge,
            d_user=d_user,
            n_clusters=min(5, len(skills)),
            n_heads=4,
            d_k=min(64, d_knowledge),
            d_v=min(64, d_knowledge),
            alpha=2.0,
            seed=42,
        )
        hare.set_knowledge_pool(skill_embeddings)

    decoder = InterpolationDecoder(top_k=3, temperature=0.3)

    # -- Phase 1: Train each user profile with domain-specific interactions ----

    print("=" * 70)
    print("PHASE 1: Training user profiles")
    print("=" * 70)

    for user_id, profile in USER_PROFILES.items():
        training_texts = profile["training_queries"]

        if backend == "st":
            training_embs = embedder.encode(training_texts)
        else:
            # For TF-IDF, we need to re-fit including training queries
            # Use transform (may have unseen terms, which is fine)
            training_embs = embedder.encode(training_texts)

        for i, emb in enumerate(training_embs):
            result = hare.recommend(emb, user_id=user_id, return_details=True)
            # High reward -- the user finds these queries highly relevant
            hare.update(emb, user_id, reward=0.9 + 0.1 * np.random.random(),
                        synthesis=result["synthesis"])

        user = hare.get_user(user_id)
        print(f"  {profile['label']:<22s}  "
              f"interactions={user.n_interactions:>2d}  "
              f"uncertainty={user.uncertainty:.3f}")

    # -- Phase 2: All users ask the same query --------------------------------

    print(f"\n{'=' * 70}")
    print(f"PHASE 2: Shared query: \"{SHARED_QUERY}\"")
    print(f"{'=' * 70}")

    if backend == "st":
        query_emb = embedder.encode([SHARED_QUERY])[0]
    else:
        query_emb = embedder.encode([SHARED_QUERY])[0]

    user_results = {}
    synthesis_vectors = []

    for user_id, profile in USER_PROFILES.items():
        result = hare.recommend(query_emb, user_id=user_id, return_details=True)
        top_indices, blend_weights = decoder.get_blend_weights(
            result["synthesis"], skill_embeddings
        )

        # Get top skill titles
        top_skills = [(skills[i]["title"], float(w))
                      for i, w in zip(top_indices, blend_weights)]

        user_results[user_id] = {
            "label": profile["label"],
            "color": profile["color"],
            "entropy": float(result["attention_entropy"]),
            "uncertainty": float(result["user_uncertainty"]),
            "n_interactions": result["n_interactions"],
            "top_skills": top_skills,
            "attention_weights": np.mean(result["attention_weights"], axis=0).tolist(),
        }
        synthesis_vectors.append(result["synthesis"])

        print(f"\n  {profile['label']}:")
        print(f"    Entropy: {result['attention_entropy']:.4f}  "
              f"Uncertainty: {result['user_uncertainty']:.4f}")
        print(f"    Top skills (by decoder blend):")
        for title, w in top_skills:
            print(f"      {w:.3f}  {title}")

    # -- Phase 3: RAG baseline (same for all users) ---------------------------

    print(f"\n{'=' * 70}")
    print("PHASE 3: RAG baseline (cosine top-3, user-independent)")
    print(f"{'=' * 70}")

    sim = cosine_similarity(query_emb.reshape(1, -1), skill_embeddings)[0]
    rag_top_k = np.argsort(sim)[-3:][::-1]
    rag_skills = [(skills[i]["title"], float(sim[i])) for i in rag_top_k]

    print(f"  RAG retrieves the SAME 3 skills for ALL users:")
    for title, s in rag_skills:
        print(f"    {s:.3f}  {title}")

    # -- Phase 4: Divergence metrics ------------------------------------------

    print(f"\n{'=' * 70}")
    print("PHASE 4: Divergence analysis")
    print(f"{'=' * 70}")

    synthesis_matrix = np.array(synthesis_vectors)
    # Pairwise cosine similarity between all user synthesis vectors
    pairwise_sim = cosine_similarity(synthesis_matrix)
    user_ids = list(USER_PROFILES.keys())
    n_users = len(user_ids)

    print(f"\n  Pairwise synthesis similarity (1.0=identical, lower=more personalized):")
    print(f"  {'':>22s}", end="")
    for uid in user_ids:
        print(f"  {USER_PROFILES[uid]['label'][:8]:>8s}", end="")
    print()

    for i, uid_i in enumerate(user_ids):
        print(f"  {USER_PROFILES[uid_i]['label']:>22s}", end="")
        for j in range(n_users):
            val = pairwise_sim[i, j]
            print(f"  {val:>8.3f}", end="")
        print()

    # Mean off-diagonal similarity
    off_diag = pairwise_sim[np.triu_indices(n_users, k=1)]
    mean_sim = float(np.mean(off_diag))
    mean_divergence = 1 - mean_sim

    print(f"\n  Mean pairwise similarity: {mean_sim:.4f}")
    print(f"  Mean divergence:         {mean_divergence:.4f}")
    print(f"  RAG divergence:          0.0000 (identical for all users)")

    # -- Per-skill attention comparison across users --------------------------

    print(f"\n  Per-skill attention weight by user (mean across heads):")
    print(f"  {'Skill':<28s}", end="")
    for uid in user_ids:
        print(f"  {USER_PROFILES[uid]['label'][:6]:>6s}", end="")
    print(f"  {'StdDev':>6s}")
    print(f"  {'-' * 28}", end="")
    for _ in user_ids:
        print(f"  {'------':>6s}", end="")
    print(f"  {'------':>6s}")

    for i, skill in enumerate(skills):
        title = skill["title"][:27]
        weights = [user_results[uid]["attention_weights"][i] for uid in user_ids]
        std = np.std(weights)
        print(f"  {title:<28s}", end="")
        for w in weights:
            print(f"  {w:>6.3f}", end="")
        marker = " <--" if std > 0.01 else ""
        print(f"  {std:>6.3f}{marker}")

    # -- Summary --------------------------------------------------------------

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"  HARE: 5 users, same query -> {n_users} different synthesis vectors")
    print(f"  RAG:  5 users, same query -> 1 identical retrieval")
    print(f"  Mean personalization divergence: {mean_divergence:.4f}")
    print(f"  This demonstrates user-conditioned generation:")
    print(f"  the attention mechanism attends to different knowledge")
    print(f"  regions based on each user's learned latent state.")

    # -- Build results dict for JSON output -----------------------------------

    results = {
        "experiment": "5-user-divergence",
        "shared_query": SHARED_QUERY,
        "n_skills": len(skills),
        "embedding_dim": d_knowledge,
        "backend": backend,
        "metrics": {
            "mean_pairwise_similarity": mean_sim,
            "mean_divergence": mean_divergence,
            "rag_divergence": 0.0,
        },
        "users": user_results,
        "rag_baseline": {
            "top_skills": rag_skills,
            "note": "Identical for all users",
        },
        "pairwise_similarity_matrix": {
            "user_ids": user_ids,
            "matrix": pairwise_sim.tolist(),
        },
        "skill_titles": [s["title"] for s in skills],
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="HARE 5-User Divergence Demo (Paper Figure 2)"
    )
    parser.add_argument(
        "--backend", choices=["tfidf", "st"], default="tfidf",
        help="Embedding backend."
    )
    parser.add_argument(
        "--learnable", action="store_true",
        help="Use LearnableHARE with trained attention projections."
    )
    parser.add_argument(
        "--attention-checkpoint", type=str, default=None,
        help="Path to trained attention weights (for --learnable mode)."
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Save results JSON to this path."
    )
    args = parser.parse_args()

    results = run_divergence_experiment(
        backend=args.backend,
        learnable=args.learnable,
        attention_checkpoint=args.attention_checkpoint,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
