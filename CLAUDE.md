# HARE — Project Context

## What is this?
HARE (Hybrid Attention-Reinforced Exploration) is a novel recommender system that combines contextual bandits with transformer attention and generative fine-tuning. Instead of selecting from existing items, HARE synthesizes new content.

## Tech Stack
- Python 3.11+
- numpy, scikit-learn (core math, baselines)
- PyTorch + transformers (attention module, GPT-2 fine-tuning)
- sentence-transformers (embedding layer)
- pytest (testing)

## Architecture
- `hare/bandits/linucb.py` — LinUCB baseline (Li et al., 2010)
- `hare/bandits/attentive_bandit.py` — HARE: attention-UCB fusion
- `hare/attention/cross_attention.py` — Multi-head cross-attention + uncertainty injection
- `hare/synthesis/generator.py` — Fine-tuned GPT-2 decoder
- `hare/utils/embeddings.py` — Text embeddings (sentence-transformers / TF-IDF)

## Conventions
- Gitmoji commits, feature branches merged to main
- Apache 2.0 license
- Pure NumPy for bandit math, PyTorch for attention/generation
- Academic paper target: RecSys 2026 / NeurIPS 2026 workshop

## Key Documents
- `PLAN.md` — Full research plan, timeline, experimental design
- `paper/notes.md` — Mathematical derivations and literature notes
