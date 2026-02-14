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
   - Scaling to large item pools and larger decoder models
   - Real user studies with online feedback convergence
   - Domain transfer: education (curriculum synthesis), scientific literature (research brief generation), commercial applications (product design, portfolio construction, content generation)
   - The generative recommendation paradigm as a general framework beyond selection
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

## 9. Future Directions: Where Generative Recommendation Creates Value

The core thesis — *synthesize what should exist, don't just select what does* — applies far beyond Claude Skills. Every domain where recommendation currently means "pick from a catalogue" is a domain where HARE could generate something better than anything in the catalogue.

### 9.1 Academic Use Cases

**Personalized Education & Curriculum Design**
Current EdTech recommends existing courses or lessons. HARE could synthesize a custom learning module that blends the best elements of multiple resources, tuned to the student's knowledge gaps and learning style. A student struggling with Bayesian inference doesn't need "the best existing lecture" — they need a lecture that *doesn't exist yet*, one that bridges their specific prerequisite gaps.

**Scientific Literature Synthesis**
Researchers drown in papers. Current tools recommend existing papers to read. HARE could synthesize a *novel research brief* — a generated document that integrates findings across multiple papers, highlights contradictions, and identifies gaps. Not "read these 5 papers" but "here's what these 5 papers collectively mean for your question."

**Drug Compound Suggestion**
Pharmaceutical recommendation currently means screening existing compound libraries. HARE's attention-weighted synthesis over molecular feature spaces could propose *novel molecular structures* that blend properties of known effective compounds, biased toward under-explored regions of chemical space. The uncertainty injection maps directly to the explore-exploit tradeoff in drug discovery.

**Grant Proposal & Research Direction Generation**
Funding agencies and PIs need to identify promising research directions. Rather than recommending existing funded projects as templates, HARE could synthesize novel research proposals that combine ideas from successful grants in under-explored intersections.

### 9.2 Commercial Use Cases

**Dynamic Product Design / Made-to-Order Commerce**
E-commerce recommends existing products. With generative manufacturing (3D printing, on-demand production), HARE could synthesize *product specifications* — a shoe design that blends features from items the user browsed, biased toward novel aesthetics they haven't seen. The gap between "recommend a shoe" and "design a shoe for this person" is exactly the gap HARE fills.

**Content Creation at Scale (Marketing, Ad Copy, Email)**
Marketing platforms recommend existing templates or past campaigns. HARE could generate novel ad copy, email subject lines, or campaign briefs that synthesize what worked across similar audiences while exploring new creative directions. The bandit exploration maps to A/B testing — but instead of testing existing variants, HARE generates the variants to test.

**Financial Portfolio Construction**
Robo-advisors select from existing funds/ETFs. HARE could synthesize *custom portfolio allocations* that blend strategies from the existing fund universe, with uncertainty injection mapping to the risk/reward tradeoff. Not "buy this ETF" but "here's a novel weighting across these instruments that no existing fund offers."

**Personalized Meal Planning & Nutrition**
Recipe recommendation selects existing recipes. HARE could synthesize novel recipes optimized for a user's dietary constraints, flavor preferences, and nutritional goals — a meal that doesn't exist in any cookbook but is the attention-weighted blend of what the user would like, with exploration toward new cuisines.

**Enterprise Knowledge Base & Policy Generation**
Companies recommend existing internal docs. HARE could synthesize *new policy documents, runbooks, or FAQ entries* by attending over the existing knowledge base, identifying gaps, and generating content that should exist but doesn't. Exploration ensures coverage of edge cases.

### 9.3 The Common Thread

In every case, the pattern is identical:
1. A pool of existing items with rich features (the K/V in attention)
2. A user context with specific needs (the Q in attention)
3. Value in novelty — the ideal output may not exist yet
4. An explore-exploit tradeoff — generating familiar content is safe but suboptimal

This is the fundamental argument for HARE as a *general framework*, not a domain-specific tool. The paper demonstrates it on Claude Skills (structured, evaluable, practical); future work extends to these higher-impact domains.

### 9.4 Roadmap Beyond the Paper
- **Phase 1 (paper):** Claude Skills generation — proves the framework works
- **Phase 2:** Multi-domain evaluation (education + one commercial domain) — proves generalizability
- **Phase 3:** Scale decoder from GPT-2 to larger models — proves the architecture doesn't depend on model size
- **Phase 4:** Real user studies with online learning — proves the feedback loop converges in practice
- **Phase 5:** Domain-specific productization (EdTech SaaS, content generation API, portfolio tool)

---

## 10. Timeline

| Week | Milestone |
|------|-----------|
| 1 | Repo setup, LinUCB baseline, embedding utilities |
| 2 | Cross-attention module, uncertainty injection |
| 3 | HARE algorithm integration, simulated experiments |
| 4 | GPT-2 fine-tuning pipeline, Claude Skills data |
| 5 | Generation experiments, metrics, ablations |
| 6 | Paper draft (methods + experiments) |
| 7 | Paper revision, additional experiments, submission prep |
