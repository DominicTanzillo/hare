# HARE: Hybrid Attention-Reinforced Exploration for Generative Recommendation

> LLMs generate the same output for the same input, regardless of who's asking. RAG improves grounding but not personalization. Recommender systems personalize but can't generate. **HARE unifies all three.**

HARE learns a latent model of each user through principled exploration, attends over a knowledge pool with uncertainty-augmented attention, and decodes the result into **novel synthesized content** tailored to what *this user* needs for *this problem*. Specificity isn't a parameter — it emerges from the system's evolving understanding of you.

## The Gap

| Approach | Models the user | Explores | Generates novel content | Personalizes |
|----------|:--------------:|:--------:|:----------------------:|:------------:|
| Standard LLM | No | No | Yes | No |
| RAG (Lewis et al., 2020) | No | No | Yes (grounded) | No |
| LinUCB (Li et al., 2010) | Partially | Yes (UCB) | No | Yes (selection) |
| Attention-based RecSys | No | No | No | Yes (selection) |
| GPT4Rec / Generative RecSys | No | No | Partially | Partially |
| **HARE (ours)** | **Yes (latent state)** | **Yes (attention-UCB)** | **Yes (synthesis)** | **Yes (emergent)** |

## Core Idea

Two users asking the same question should get *different* answers — not because of shallow profiling, but because a principled model of each user's latent needs drives different attention patterns over the knowledge pool, which produces different synthesized outputs.

```
Q = W_Q · [x_query ⊕ u_t]                    # query conditioned on user state
K, V = W_K · X_knowledge, W_V · X_knowledge   # knowledge pool
U_j = √(q^T Σ_j^{-1} q)                      # exploration bonus (high where uncertain)
A = softmax(QK^T / √d + α · U)                # uncertainty-augmented attention
z = A @ V                                      # synthesized representation
output = Decoder(z, u_t)                       # generate novel content for THIS user
r_t → update(u_t, Σ_u)                         # learn from feedback, reduce uncertainty
```

New user? High uncertainty → broad attention → exploratory, general output.
Known user? Low uncertainty → focused attention → precise, personalized output.
The algorithm discovers the right level of specificity. You don't configure it.

## Quick Start

```bash
pip install -e ".[dev]"
pytest tests/
python examples/skill_generation.py
```

## Project Structure

```
hare/
├── hare/
│   ├── bandits/          # LinUCB baseline + HARE attentive bandit
│   ├── attention/        # Uncertainty-augmented cross-attention
│   ├── synthesis/        # Generative decoder (fine-tuned transformer)
│   └── utils/            # Embeddings, user state, data loading
├── examples/             # Demo scripts (Claude Skills, study plans)
├── tests/                # Unit + integration tests
├── data/                 # Training data (Claude Skills, etc.)
└── paper/                # Research notes, derivations, paper drafts
```

## License

Apache 2.0
