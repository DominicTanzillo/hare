# HARE: Hybrid Attention-Reinforced Exploration for Generative Recommendation

> Recommender systems select from what exists. HARE generates what *should* exist.

HARE combines contextual bandits (LinUCB) with transformer attention mechanisms and generative fine-tuning to build a recommendation system that **synthesizes novel content** rather than merely ranking existing items.

## The Gap

| Approach | Selects existing items | Non-linear feature interactions | Explore/exploit | Generates new content |
|----------|:---------------------:|:-------------------------------:|:---------------:|:---------------------:|
| LinUCB (Li et al., 2010) | Yes | No | Yes (UCB) | No |
| Attention-based RecSys | Yes | Yes | No | No |
| Generative RecSys (GPT4Rec, etc.) | Yes | Yes | No | Partially |
| **HARE (ours)** | **Optional** | **Yes** | **Yes** | **Yes** |

## Key Innovation

HARE introduces **Uncertainty-Augmented Attention**: exploration bonuses from the bandit framework are injected directly into transformer attention scores, so the model attends more strongly to under-explored content regions. The attended representation is then decoded into novel synthesized content.

```
A_hare = softmax((QK^T / sqrt(d_k)) + alpha * U)
z = A_hare @ V
output = Decoder(z, x_user)
```

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
│   ├── attention/        # Multi-head cross-attention with UCB uncertainty
│   ├── synthesis/        # Generative decoder (fine-tuned transformer)
│   └── utils/            # Embeddings, data loading
├── examples/             # Demo scripts
├── tests/                # Unit + integration tests
├── data/                 # Training data (Claude Skills, etc.)
└── paper/                # Research notes and paper drafts
```

## License

MIT
