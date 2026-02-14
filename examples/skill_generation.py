#!/usr/bin/env python3
"""HARE demo: Claude Skills synthesis.

Demonstrates how HARE synthesizes a novel Claude Skill by attending over
a pool of existing skills with uncertainty-augmented attention.

Two simulated users with different expertise levels get *different*
synthesized outputs from the same query — showing emergent personalization.
"""

import numpy as np

from hare.bandits.attentive_bandit import HARE
from hare.synthesis.generator import InterpolationDecoder
from hare.utils.embeddings import TfidfEmbedder

# ── Sample Claude Skills (knowledge pool) ──────────────────────────────

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


def main():
    # ── Embed the knowledge pool ────────────────────────────────────────
    skill_texts = [f"{s['title']}: {s['description']}" for s in SKILLS]

    embedder = TfidfEmbedder(max_features=500, output_dim=32)
    embedder.fit(skill_texts)
    skill_embeddings = embedder.encode(skill_texts)

    d_knowledge = embedder.dim
    print(f"Knowledge pool: {len(SKILLS)} skills, {d_knowledge}-dim embeddings\n")

    # ── Initialize HARE ─────────────────────────────────────────────────
    hare = HARE(
        d_knowledge=d_knowledge,
        d_user=16,
        n_clusters=min(4, len(SKILLS)),
        n_heads=2,
        d_k=32,
        d_v=32,
        alpha=1.5,
        seed=42,
    )
    hare.set_knowledge_pool(skill_embeddings)

    decoder = InterpolationDecoder(top_k=3, temperature=0.3)

    # ── Query: "I need help with code quality" ──────────────────────────
    query_text = "I need a skill that helps improve code quality and catch bugs"
    query_emb = embedder.encode([query_text])[0]

    # ── User A: Beginner (no history) ───────────────────────────────────
    print("=" * 60)
    print("USER A: New user (no interaction history)")
    print("=" * 60)

    result_a = hare.recommend(query_emb, user_id="beginner", return_details=True)
    print(f"  Attention entropy:  {result_a['attention_entropy']:.3f} (higher = more general)")
    print(f"  User uncertainty:   {result_a['user_uncertainty']:.3f}")
    print(f"  Interactions:       {result_a['n_interactions']}")

    output_a = decoder.generate(result_a["synthesis"], skill_texts, skill_embeddings)
    print(f"\n{output_a}\n")

    # ── User B: Experienced (simulate past interactions) ────────────────
    # Simulate: User B has interacted several times, preferring security-related skills
    print("=" * 60)
    print("USER B: Experienced user (simulated security-focused history)")
    print("=" * 60)

    # Simulate interactions that teach HARE about User B's preferences
    security_queries = [
        "find security vulnerabilities in my code",
        "audit my dependencies for CVEs",
        "check for injection attacks",
    ]
    security_query_embs = embedder.encode(security_queries)

    for i, sq_emb in enumerate(security_query_embs):
        r = hare.recommend(sq_emb, user_id="security_expert", return_details=True)
        # High reward for security-related synthesis
        hare.update(sq_emb, "security_expert", reward=0.9, synthesis=r["synthesis"])

    # Now ask the SAME query as User A
    result_b = hare.recommend(query_emb, user_id="security_expert", return_details=True)
    print(f"  Attention entropy:  {result_b['attention_entropy']:.3f} (higher = more general)")
    print(f"  User uncertainty:   {result_b['user_uncertainty']:.3f}")
    print(f"  Interactions:       {result_b['n_interactions']}")

    output_b = decoder.generate(result_b["synthesis"], skill_texts, skill_embeddings)
    print(f"\n{output_b}\n")

    # ── Compare ─────────────────────────────────────────────────────────
    print("=" * 60)
    print("COMPARISON: Same query, different users")
    print("=" * 60)
    print(f"  User A entropy: {result_a['attention_entropy']:.3f} vs User B entropy: {result_b['attention_entropy']:.3f}")
    print(f"  User A uncertainty: {result_a['user_uncertainty']:.3f} vs User B uncertainty: {result_b['user_uncertainty']:.3f}")

    # Cosine similarity between the two syntheses
    from sklearn.metrics.pairwise import cosine_similarity
    sim = cosine_similarity(
        result_a["synthesis"].reshape(1, -1),
        result_b["synthesis"].reshape(1, -1),
    )[0, 0]
    print(f"  Synthesis cosine similarity: {sim:.3f}")
    print(f"  (1.0 = identical outputs, <1.0 = personalization divergence)")
    print(f"\n  -> Different users, same query, different synthesis. HARE personalizes.")


if __name__ == "__main__":
    main()
