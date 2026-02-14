# HARE Research Plan

## Thesis

Existing recommender systems — whether collaborative filtering, contextual bandits, or attention-based neural models — fundamentally **select** from a fixed item catalogue. Even "generative" approaches like GPT4Rec use generation only to produce ranked lists of existing items. We propose HARE: a system that injects bandit-style exploration directly into transformer attention, then **decodes the attended representation into genuinely novel content**. The result is a recommender that doesn't just find the best existing item — it synthesizes the ideal item that *should* exist.

---

## 1. Problem Formulation

### 1.1 Setting
At each round `t = 1, ..., T`:
- A **user context** `x_t ∈ R^d` arrives (user profile, session features, query)
- A **content pool** `C = {c_1, ..., c_N}` of existing items with features `X_items ∈ R^{N×d}` is available
- The system must produce an **output** `y_t` (which may or may not exist in C)
- The user provides a **reward** `r_t ∈ [0, 1]` (click, rating, engagement)

### 1.2 Objective
Minimize cumulative regret vs. an oracle that always produces the optimal content:
```
R(T) = Σ_{t=1}^{T} [r*(x_t) - r(x_t, y_t)]
```

### 1.3 Why This Is Hard
- **LinUCB** handles explore/exploit but can only select existing arms, with linear reward models
- **Attention-based models** capture non-linear interactions but have no exploration mechanism
- **LLM generators** can produce novel content but have no principled exploration — they exploit their training distribution

HARE bridges all three.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                     HARE Pipeline                    │
│                                                      │
│  User Context x_t ──┐                               │
│                      ▼                               │
│              ┌──────────────┐                        │
│              │  Encoder     │  (frozen or fine-tuned  │
│              │  (LM embed)  │   sentence-transformer) │
│              └──────┬───────┘                        │
│                     │                                │
│  Item Pool X_items ─┤                                │
│                     ▼                                │
│  ┌─────────────────────────────────────┐             │
│  │  Uncertainty-Augmented Attention    │             │
│  │                                     │             │
│  │  Q = W_Q · x_user                  │             │
│  │  K = W_K · X_items                 │             │
│  │  V = W_V · X_items                 │             │
│  │  U_ij = √(x_i^T Σ_j^{-1} x_i)    │  ← bandits │
│  │  A = softmax(QK^T/√d + α·U)       │             │
│  │  z = A · V                          │             │
│  └───────────────┬─────────────────────┘             │
│                  │                                   │
│                  ▼                                   │
│  ┌─────────────────────────────────────┐             │
│  │  Generative Decoder                 │             │
│  │  (fine-tuned transformer LM)        │             │
│  │                                     │             │
│  │  Conditioned on: z ⊕ x_user        │             │
│  │  Output: novel content y_t          │             │
│  └───────────────┬─────────────────────┘             │
│                  │                                   │
│                  ▼                                   │
│  y_t (synthesized recommendation)                    │
│  r_t (user feedback) ──→ update Σ_j, fine-tune      │
└─────────────────────────────────────────────────────┘
```

---

## 3. Components & Implementation Plan

### Phase 1: Foundations (Week 1-2)

**3.1 LinUCB Baseline** — `hare/bandits/linucb.py`
- Standard disjoint LinUCB (Li et al., 2010)
- Per-arm ridge regression: `A_a ∈ R^{d×d}`, `b_a ∈ R^d`, `θ_a = A_a^{-1} b_a`
- UCB selection: `a_t = argmax_a (x^T θ_a + α√(x^T A_a^{-1} x))`
- This is our primary comparison baseline

**3.2 Embedding Layer** — `hare/utils/embeddings.py`
- Sentence-transformer embeddings (all-MiniLM-L6-v2 or similar)
- TF-IDF fallback for lightweight experiments
- Handles both user context and item features

### Phase 2: Attention + Exploration (Week 2-3)

**3.3 Multi-Head Cross-Attention** — `hare/attention/cross_attention.py`
- Pure PyTorch multi-head cross-attention
- User context as Q, item pool as K/V
- Standard scaled dot-product with learned projections

**3.4 Uncertainty Injection** — integrated into attention
- Maintain covariance matrix `Σ_j` per item cluster (not per item — too expensive)
- Compute UCB-style uncertainty: `U_ij = √(x_i^T Σ_j^{-1} x_i)`
- Add to pre-softmax attention scores: `A = softmax(QK^T/√d + α·U)`
- α is a learnable exploration parameter (or annealed schedule)

### Phase 3: Generative Decoder (Week 3-5)

**3.5 Fine-Tuned Decoder** — `hare/synthesis/generator.py`
- Base model: GPT-2 small (124M) or DistilGPT-2 (82M) for tractability
- Fine-tune on domain-specific content (see §4 Training Data)
- Input: concatenation of attended representation z and user context
- Output: generated text (new skill description, study plan, etc.)
- Training objective: standard causal LM loss, conditioned on z

**3.6 Reinforcement from Feedback** — online learning loop
- User reward r_t updates:
  1. Covariance matrices Σ_j (bandit side)
  2. Decoder weights via reward-weighted loss (RL side)
- Reward-weighted loss: `L = -r_t · log P(y_t | z, x_user)`
- This creates the feedback loop: better exploration → better synthesis → better rewards → better exploration

### Phase 4: Evaluation & Paper (Week 5-7)

**3.7 Experiments** (see §5)
**3.8 Paper writing** (see §6)

---

## 4. Training Data

### 4.1 Claude Skills Dataset
Primary training domain. Claude Skills are markdown files that follow predictable structure:
- Title, description, trigger conditions
- System prompt / instructions
- Examples

**Data collection strategy:**
- Scrape/collect publicly shared Claude Skills from community repos
- Generate synthetic skills using Claude API with controlled variation
- Create user-skill interaction logs (simulated or from usage analytics)

**Why this is a good domain:**
- Structured enough to evaluate quality (does the skill follow valid format?)
- Open-ended enough to benefit from generation (infinite possible skills)
- Practical — output is immediately useful
- Novel — no existing work on "skill recommendation as generation"

### 4.2 Additional Domains (for generalizability)
- **Study plan generation**: Given course syllabi + student profile → synthesized study plan
- **Recipe generation**: Given dietary preferences + ingredient pool → new recipe (well-studied baseline domain)

### 4.3 Simulated Bandit Environment
For controlled experiments with known optimal policy:
- Synthetic context vectors, synthetic item features
- Known reward function (e.g., bilinear with non-linear interaction)
- Allows exact regret computation

---

## 5. Experimental Design

### 5.1 Baselines
1. **LinUCB** — standard contextual bandit (Li et al., 2010)
2. **NeuralUCB** — neural network reward model with UCB (Zhou et al., 2020)
3. **Attention-only** — cross-attention recommendation without exploration (no U term)
4. **GPT-2 (vanilla)** — fine-tuned generative model without bandit exploration
5. **Random** — uniform random selection/generation

### 5.2 Metrics
- **Cumulative regret** (simulated environment with known rewards)
- **Generation quality**: BLEU, ROUGE, BERTScore against held-out high-quality items
- **Diversity**: intra-list diversity of generated items across users
- **Novelty**: % of generated items not in training set (by cosine similarity threshold)
- **Human evaluation**: small-scale rating of generated skills (1-5 quality, 1-5 relevance)

### 5.3 Ablation Studies
- HARE with vs. without uncertainty injection (α = 0)
- HARE with vs. without fine-tuning (frozen GPT-2 decoder)
- Effect of number of attention heads
- Effect of α schedule (fixed, linear decay, learned)
- Covariance update frequency

### 5.4 Key Hypotheses
1. HARE achieves lower regret than LinUCB on non-linear reward functions
2. Uncertainty injection improves generation diversity without sacrificing quality
3. The generative decoder produces higher-quality output than nearest-neighbor retrieval
4. Online reward feedback improves generation quality over time

---

## 6. Paper Outline

**Title:** HARE: Hybrid Attention-Reinforced Exploration for Generative Recommendation

**Abstract:** ~200 words. Recommender systems select; HARE generates. We inject bandit exploration into transformer attention and decode into novel content.

### Sections
1. **Introduction** — The recommendation-generation gap. Why selecting isn't enough. Motivate with Claude Skills example.
2. **Related Work**
   - Contextual bandits (LinUCB, NeuralUCB, Thompson Sampling)
   - Attention-based recommendation (DIN, BERT4Rec, SASRec)
   - Generative recommendation (GPT4Rec, generative retrieval)
   - Exploration in neural models (curiosity-driven, information gain)
3. **Method**
   - Problem formulation
   - Uncertainty-augmented attention (the core contribution)
   - Generative synthesis decoder
   - Online learning with reward feedback
   - Regret analysis (theoretical contribution)
4. **Experiments**
   - Simulated bandit environment (controlled regret analysis)
   - Claude Skills generation (primary real-world domain)
   - Study plan generation (secondary domain)
   - Ablation studies
5. **Results & Discussion**
6. **Limitations & Future Work**
   - Scaling to large item pools
   - Replacing GPT-2 with larger models
   - Real user studies
7. **Conclusion**

### Target Venues
- RecSys 2026 (primary)
- NeurIPS 2026 workshop track
- AAAI 2027

---

## 7. Implementation Priorities

### Must-have (MVP for paper)
- [ ] LinUCB baseline implementation
- [ ] Cross-attention with uncertainty injection (NumPy prototype)
- [ ] HARE attentive bandit combining the above
- [ ] Fine-tuned GPT-2 decoder conditioned on attended features
- [ ] Simulated environment with regret curves
- [ ] Claude Skills dataset collection + preprocessing
- [ ] Skill generation demo with quality metrics
- [ ] Core unit tests

### Nice-to-have (strengthens paper)
- [ ] Study plan generation domain
- [ ] Thompson Sampling variant of HARE
- [ ] Theoretical regret bound proof
- [ ] Human evaluation study
- [ ] Visualization of attention weights + uncertainty

### Stretch goals
- [ ] Real-time online learning demo
- [ ] Integration with Claude API for live skill generation
- [ ] Scaling experiments with larger models

---

## 8. Technical Decisions

### Why GPT-2 for the decoder?
- Small enough to fine-tune on a single GPU
- Well-understood architecture, easy to reproduce
- Paper-friendly: reviewers can run experiments
- Can always scale up in future work (this is a feature, not a limitation)

### Why not end-to-end transformer?
- Separating the bandit exploration from the generative decoder makes the contribution clearer
- Easier to ablate: can swap in different decoders
- The attention-UCB fusion is the novel contribution; the decoder is modular

### Why cluster-level covariance instead of per-item?
- N items × d² covariance entries is prohibitive for large item pools
- Clustering items and maintaining per-cluster covariance is O(K·d²) where K << N
- Empirically, exploration at cluster level is sufficient (hypothesis to test)

### α (exploration) schedule
- Start with fixed α, validated by grid search
- Paper contribution: learned α via meta-gradient (if time permits)
- Annealing schedule α_t = α_0 / √t as fallback

---

## 9. Timeline

| Week | Milestone |
|------|-----------|
| 1 | Repo setup, LinUCB baseline, embedding utilities |
| 2 | Cross-attention module, uncertainty injection |
| 3 | HARE algorithm integration, simulated experiments |
| 4 | GPT-2 fine-tuning pipeline, Claude Skills data |
| 5 | Generation experiments, metrics, ablations |
| 6 | Paper draft (methods + experiments) |
| 7 | Paper revision, additional experiments, submission prep |
