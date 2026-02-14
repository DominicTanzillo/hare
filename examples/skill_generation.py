#!/usr/bin/env python3
"""HARE demo: Claude Skills synthesis.

Demonstrates how HARE synthesizes a novel Claude Skill by attending over
a pool of existing skills with uncertainty-augmented attention.

Two simulated users with different expertise levels get *different*
synthesized outputs from the same query -- showing emergent personalization.

Usage:
    python examples/skill_generation.py                    # auto-detect best backend
    python examples/skill_generation.py --backend tfidf    # lightweight, no GPU
    python examples/skill_generation.py --backend st       # sentence-transformers (384-d)
"""

import argparse
import sys

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from hare.bandits.attentive_bandit import HARE
from hare.synthesis.generator import InterpolationDecoder
from hare.utils.embeddings import TfidfEmbedder, SentenceTransformerEmbedder

# -- Sample Claude Skills (knowledge pool) ------------------------------------

SKILLS = [
    {
        "title": "Git Commit Helper",
        "description": (
            "Analyzes staged changes and generates conventional commit messages. "
            "Follows the Conventional Commits specification with type, scope, and "
            "description. Supports gitmoji prefix option."
        ),
    },
    {
        "title": "Code Reviewer",
        "description": (
            "Reviews code for bugs, security vulnerabilities, and style issues. "
            "Provides inline suggestions with severity levels. Checks for OWASP "
            "top 10 vulnerabilities and common anti-patterns."
        ),
    },
    {
        "title": "API Documentation Writer",
        "description": (
            "Generates OpenAPI/Swagger documentation from code. Extracts endpoint "
            "signatures, request/response schemas, and authentication requirements. "
            "Produces markdown or YAML output."
        ),
    },
    {
        "title": "Test Generator",
        "description": (
            "Creates unit tests for functions and classes. Covers happy path, edge "
            "cases, and error scenarios. Supports pytest, jest, and go test frameworks. "
            "Generates test fixtures and mocks."
        ),
    },
    {
        "title": "Dependency Auditor",
        "description": (
            "Scans project dependencies for known vulnerabilities and outdated packages. "
            "Cross-references CVE databases and suggests safe upgrade paths. "
            "Checks license compatibility."
        ),
    },
    {
        "title": "Performance Profiler",
        "description": (
            "Analyzes code for performance bottlenecks. Identifies O(n^2) patterns, "
            "unnecessary allocations, and N+1 query problems. Suggests optimizations "
            "with estimated impact."
        ),
    },
    {
        "title": "Database Migration Writer",
        "description": (
            "Generates database migration scripts from schema changes. Supports "
            "PostgreSQL, MySQL, and SQLite. Handles index creation, foreign keys, "
            "and data transformations with rollback support."
        ),
    },
    {
        "title": "Refactoring Assistant",
        "description": (
            "Identifies code smells and suggests refactoring patterns. Handles "
            "extract method, extract class, replace conditional with polymorphism. "
            "Preserves behavior while improving structure."
        ),
    },
]


def build_embedder(backend: str, corpus: list[str]):
    """Build and fit an embedder based on the selected backend."""
    if backend == "st":
        embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")
        # sentence-transformers doesn't need fit(), just encode
        return embedder
    else:
        embedder = TfidfEmbedder(max_features=500, output_dim=32)
        embedder.fit(corpus)
        return embedder


def detect_backend() -> str:
    """Auto-detect best available backend."""
    try:
        import sentence_transformers  # noqa: F401
        return "st"
    except ImportError:
        return "tfidf"


def main():
    parser = argparse.ArgumentParser(description="HARE Skill Generation Demo")
    parser.add_argument(
        "--backend", choices=["tfidf", "st"], default=None,
        help="Embedding backend: 'tfidf' (lightweight) or 'st' (sentence-transformers)."
    )
    args = parser.parse_args()

    backend = args.backend or detect_backend()
    print(f"Embedding backend: {backend}\n")

    # -- Embed the knowledge pool ---------------------------------------------
    skill_texts = [f"{s['title']}: {s['description']}" for s in SKILLS]
    embedder = build_embedder(backend, skill_texts)
    skill_embeddings = embedder.encode(skill_texts)
    d_knowledge = skill_embeddings.shape[1]

    print(f"Knowledge pool: {len(SKILLS)} skills, {d_knowledge}-dim embeddings\n")

    # -- Initialize HARE ------------------------------------------------------
    # Scale internal dimensions to embedding size
    # For high-dim embeddings (384), use larger internal state for stronger personalization
    d_user = min(128, d_knowledge)
    d_k = min(96, d_knowledge)
    d_v = min(96, d_knowledge)

    hare = HARE(
        d_knowledge=d_knowledge,
        d_user=d_user,
        n_clusters=min(4, len(SKILLS)),
        n_heads=4,
        d_k=d_k,
        d_v=d_v,
        alpha=3.0,
        seed=42,
    )
    hare.set_knowledge_pool(skill_embeddings)

    decoder = InterpolationDecoder(top_k=3, temperature=0.3)

    # -- Query: "I need help with code quality" -------------------------------
    query_text = "I need a skill that helps improve code quality and catch bugs"
    query_emb = embedder.encode([query_text])[0]

    # -- User A: Beginner (no history) ----------------------------------------
    print("=" * 60)
    print("USER A: New user (no interaction history)")
    print("=" * 60)

    result_a = hare.recommend(query_emb, user_id="beginner", return_details=True)
    print(f"  Attention entropy:  {result_a['attention_entropy']:.3f} (higher = more general)")
    print(f"  User uncertainty:   {result_a['user_uncertainty']:.3f}")
    print(f"  Interactions:       {result_a['n_interactions']}")

    output_a = decoder.generate(result_a["synthesis"], skill_texts, skill_embeddings)
    print(f"\n{output_a}\n")

    # -- User B: Experienced (simulate past interactions) ---------------------
    print("=" * 60)
    print("USER B: Experienced user (simulated security-focused history)")
    print("=" * 60)

    # Simulate 20 interactions that teach HARE about User B's preferences
    security_queries = [
        "find security vulnerabilities in my code",
        "audit my dependencies for CVEs",
        "check for injection attacks",
        "scan for OWASP top 10 issues",
        "review authentication flow for weaknesses",
        "check for SQL injection in database queries",
        "verify input sanitization across endpoints",
        "analyze CORS and CSP headers",
        "review secret management and credential storage",
        "check for insecure deserialization patterns",
        "detect cross-site scripting XSS in templates",
        "review authorization and access control logic",
        "check for path traversal vulnerabilities",
        "audit cryptographic implementations",
        "scan for hardcoded secrets and API keys",
        "review session management security",
        "check for server-side request forgery SSRF",
        "analyze rate limiting and brute force protection",
        "review error handling for information leakage",
        "check for insecure direct object references",
    ]
    security_query_embs = embedder.encode(security_queries)

    for sq_emb in security_query_embs:
        r = hare.recommend(sq_emb, user_id="security_expert", return_details=True)
        hare.update(sq_emb, "security_expert", reward=0.95, synthesis=r["synthesis"])

    # Now ask the SAME query as User A
    result_b = hare.recommend(query_emb, user_id="security_expert", return_details=True)
    print(f"  Attention entropy:  {result_b['attention_entropy']:.3f} (higher = more general)")
    print(f"  User uncertainty:   {result_b['user_uncertainty']:.3f}")
    print(f"  Interactions:       {result_b['n_interactions']}")

    output_b = decoder.generate(result_b["synthesis"], skill_texts, skill_embeddings)
    print(f"\n{output_b}\n")

    # -- Compare --------------------------------------------------------------
    print("=" * 60)
    print("COMPARISON: Same query, different users")
    print("=" * 60)
    print(f"  User A entropy:      {result_a['attention_entropy']:.3f} vs User B entropy:      {result_b['attention_entropy']:.3f}")
    print(f"  User A uncertainty:  {result_a['user_uncertainty']:.3f} vs User B uncertainty:  {result_b['user_uncertainty']:.3f}")

    sim = cosine_similarity(
        result_a["synthesis"].reshape(1, -1),
        result_b["synthesis"].reshape(1, -1),
    )[0, 0]
    print(f"  Synthesis similarity: {sim:.3f}")
    print(f"  (1.0 = identical, <1.0 = personalized)")

    # -- Show attention weight divergence on individual skills ----------------
    print(f"\n  Per-skill attention weight comparison (mean across heads):")
    w_a = np.mean(result_a["attention_weights"], axis=0)
    w_b = np.mean(result_b["attention_weights"], axis=0)
    for i, skill in enumerate(SKILLS):
        marker = " <--" if abs(w_a[i] - w_b[i]) > 0.01 else ""
        print(f"    {skill['title']:<28s}  A={w_a[i]:.3f}  B={w_b[i]:.3f}{marker}")

    print(f"\n  Note: In high-dim embedding space (384-d), cosine similarity stays high")
    print(f"  but the TOP-K ITEMS CHANGE -- User B's blend shifts toward security.")
    print(f"  With the fine-tuned decoder, even small attention shifts produce very")
    print(f"  different generated text. The prototype decoder shows the mechanism;")
    print(f"  the LM decoder amplifies the effect.")
    print(f"\n  -> Different users, same query, different synthesis. HARE personalizes.")


if __name__ == "__main__":
    main()
