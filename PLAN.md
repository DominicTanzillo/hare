# HARE Research Plan

## Thesis

Existing recommender systems — whether collaborative filtering, contextual bandits, or attention-based neural models — fundamentally **select** from a fixed item catalogue. Even "generative" approaches like GPT4Rec use generation only to produce ranked lists of existing items. We propose HARE: a system that injects bandit-style exploration directly into transformer attention, then **decodes the attended representation into genuinely novel content**. The result is a recommender that doesn't just find the best existing item — it synthesizes the ideal item that *should* exist.

---

## 1. Problem Formulation

### 1.1 Setting
At each round `t = 1, ..., T`:
- A **query context** `x_t ∈ R^d` arrives (the immediate question/need)
- A **user latent state** `u_t ∈ R^p` summarizes everything learned about this user from prior interactions (expertise, preferences, goals, style) — with associated uncertainty `Σ_u(t) ∈ R^{p×p}`
- A **knowledge pool** `K = {k_1, ..., k_N}` of existing content with features `X_knowledge ∈ R^{N×d}` is available (documents, skills, items, domain data)
- The system must produce an **output** `y_t` — synthesized content tailored to this user's needs (which may or may not resemble anything in K)
- The user provides a **reward** `r_t ∈ [0, 1]` (explicit rating, implicit engagement, task completion)
- The system updates `u_{t+1}` and `Σ_u(t+1)` — becoming more certain about this user's needs

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

## 1B. The Deeper Problem: Personalization Without Slop

### The Specificity Paradox

Every generative system faces the same tradeoff:

- **Too general** → Vague, impersonal, lowest-common-denominator output. This is the "slop" problem. Ask ChatGPT to write you a study plan and you get the same generic 5-step outline regardless of who you are. It's "helpful" the way a billboard is helpful — technically relevant to everyone, actually useful to no one.
- **Too specific** → Overfits to surface signals. "You clicked on a Python tutorial, here are 47 more Python tutorials." Narrow, repetitive, and often wrong about what the user actually needs (maybe they clicked because they're *done* with Python and exploring alternatives).

The fundamental question HARE answers: **How does a system figure out what a particular user needs for a particular problem — at exactly the right level of specificity — without being told?**

This is not a hyperparameter. It's not a toggle between "general" and "personal." It's an *emergent property* of principled uncertainty modeling.

### Why Current Approaches Fail

**Standard LLMs** have no user model. Two users sending the identical message get outputs drawn from the same distribution. The conversation history provides some context, but there's no mechanism to represent *what the model doesn't know about this user* — so there's no mechanism to explore it.

**RAG (Retrieval-Augmented Generation)** improves grounding but not personalization. RAG does:
```
retrieve(query) → context_chunks → generate(query, context_chunks)
```
The retrieval step depends on the **query**, not the **user**. Two users with the same question get the same retrieved documents and therefore substantially similar outputs. RAG is better-informed LLM generation, but it's still impersonal.

**Collaborative filtering** ("users like you liked X") personalizes selection but (a) can only select existing items, (b) relies on population-level similarity rather than understanding *this* user's latent state, and (c) has no mechanism for synthesis.

**The missing piece in all of these**: a representation of *what the system doesn't know about the user* and a principled mechanism to reduce that uncertainty through interaction.

### HARE's Solution: The User Uncertainty Model

HARE introduces a **user latent state** `u_t` that evolves over interactions:

```
u_t = f(x_1, r_1, x_2, r_2, ..., x_t)     # accumulated user state
Σ_u(t) = uncertainty about u_t              # what we don't know yet
```

This isn't a static profile. It's a *distribution* over what the user might need, refined by every interaction. The key variables in `u_t`:

| Dimension | What it captures | How it's learned |
|-----------|-----------------|-----------------|
| Domain expertise | How much the user already knows | Reward signals on generated complexity level |
| Specificity preference | Broad overview vs. deep dive | Reward signals on output granularity |
| Latent goals | What they're actually trying to accomplish | Exploration of uncertain goal dimensions |
| Style/format | How they prefer information delivered | Reward on format variations |

**The attention mechanism becomes user-conditioned:**
```
Q = W_Q · [x_query ⊕ u_t]              # query is BOTH the question AND the user state
K = W_K · X_knowledge                    # knowledge pool (documents, skills, items)
U_j = √([x_query ⊕ u_t]^T Σ_j^{-1} [x_query ⊕ u_t])   # uncertainty given THIS user
A = softmax(QK^T / √d + α · U)          # attend more to uncertain regions
```

This means: **two users with the same query get different attention patterns** because their user states `u_t` differ, which changes both the relevance scores (QK^T) and the uncertainty bonuses (U).

### How Specificity Emerges

The attention distribution's **entropy** naturally controls specificity:

- **New user / unfamiliar domain** → High `Σ_u` → Large uncertainty bonuses → Attention spreads across many knowledge regions → Output is broader, more exploratory. The system is "casting a wide net" because it doesn't know what this user needs yet.

- **Known user / familiar domain** → Low `Σ_u` → Small uncertainty bonuses → Attention concentrates on specific knowledge regions → Output is precise, personalized. The system has learned what this user needs.

- **Known user / NEW domain** → Low `Σ_u` on some dimensions, high on others → Mixed attention pattern → Output is personalized in style/format but exploratory in content. The system leverages what it knows (how you like information) while exploring what it doesn't (what you need in this new area).

This is the "domain specification" — it's not configured, it **emerges from the interaction history**. The algorithm itself determines the right granularity.

### HARE as RAG 2.0

Reframing HARE in the RAG paradigm makes the contribution concrete:

| Step | RAG | HARE |
|------|-----|------|
| **Representation** | Query embedding | Query ⊕ User latent state |
| **Retrieval** | Top-k by cosine similarity | Uncertainty-augmented attention over full knowledge pool |
| **What drives retrieval** | The query alone | The query + what the system doesn't know about this user |
| **Aggregation** | Concatenate chunks | Attention-weighted synthesis (soft blend, not hard selection) |
| **Generation** | LLM conditioned on retrieved chunks | Fine-tuned decoder conditioned on synthesized representation |
| **Feedback loop** | None (static) | Reward updates user model + knowledge uncertainty |
| **Personalization** | None | Emergent from user latent state |

RAG asks: "What documents are relevant to this query?"
HARE asks: "What knowledge does *this user* need for *this problem*, including knowledge the system hasn't tried presenting yet?"

### Why This Revolutionizes Conversational LLMs

If HARE's user modeling works, it implies a fundamentally different conversational paradigm:

**Current LLM conversations:**
```
User message → [static model] → Response
User message → [static model + conversation context] → Response
```
The model is the same for every user. Context helps, but there's no *learning* about the user — no reduction in uncertainty, no exploration strategy, no evolving personalization.

**HARE-augmented conversations:**
```
User message → [model + user_state(t) + uncertainty(t)] → Response
User feedback → update user_state(t+1), reduce uncertainty(t+1)
User message → [model + user_state(t+1) + uncertainty(t+1)] → Better response
```

Each turn isn't just answered — it's an opportunity to reduce uncertainty about the user. The model gets *meaningfully better at serving this specific person* over the course of the conversation, not just by accumulating context tokens but by updating a principled probabilistic model of their needs.

This is what makes HARE general: the mechanism (uncertainty-augmented attention over knowledge, conditioned on evolving user state) applies to ANY domain where an LLM serves a user. Claude Skills is the proving ground. Personalized LLM interaction is the endgame.

### The "Figure Out What's Needed" Problem

To be concrete about what "figuring out what someone needs" means algorithmically:

1. **Initial interaction**: High uncertainty across all user dimensions. HARE generates something *deliberately varied* — broad enough to be useful, but with subtle probes across the specificity spectrum. Think of it as a first message that's helpful but also *diagnostic*.

2. **User responds** (explicit feedback or implicit engagement signals): HARE observes which aspects of the output got engagement. Did they click on the detailed section? Skim the overview? Ask a follow-up about a specific subtopic? Each signal updates `u_t` and reduces `Σ_u`.

3. **Next generation**: Attention shifts. Regions of high relevance AND low remaining uncertainty get exploited (specific, personalized content). Regions of high relevance AND high remaining uncertainty get explored (probing questions, varied suggestions). Regions of low relevance get suppressed.

4. **Over multiple interactions**: The system converges on this user's true needs — not the population average, not a stereotype, but a learned model of what *this person* actually needs for *this problem*.

This is why HARE isn't a recommender system that happens to generate. It's a framework for **adaptive, personalized generation with principled exploration**.

---

## 2. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                         HARE Pipeline                             │
│                                                                   │
│  Query x_t ──────────┐                                           │
│                       ▼                                           │
│  History ──→ ┌──────────────────┐                                │
│  (r_1..r_t)  │  User State      │                                │
│              │  u_t = f(history) │  ← evolving latent model       │
│              │  Σ_u(t) = uncert. │  ← what we don't know yet     │
│              └────────┬─────────┘                                │
│                       │                                           │
│                       ▼                                           │
│              ┌──────────────┐                                    │
│              │  Encoder     │  (frozen or fine-tuned              │
│              │  (LM embed)  │   sentence-transformer)            │
│              └──────┬───────┘                                    │
│                     │                                            │
│  Knowledge Pool ────┤  (documents, skills, items, domain data)   │
│  X_knowledge        │                                            │
│                     ▼                                            │
│  ┌───────────────────────────────────────────────┐               │
│  │  Uncertainty-Augmented Attention               │               │
│  │                                                │               │
│  │  Q = W_Q · [x_query ⊕ u_t]                   │  ← user-      │
│  │  K = W_K · X_knowledge                        │    conditioned │
│  │  V = W_V · X_knowledge                        │               │
│  │  U_j = √([x⊕u]^T Σ_j^{-1} [x⊕u])           │  ← bandits   │
│  │  A = softmax(QK^T/√d + α·U)                  │               │
│  │  z = A · V     (synthesized representation)   │               │
│  └───────────────────┬───────────────────────────┘               │
│                      │                                           │
│                      ▼                                           │
│  ┌───────────────────────────────────────────────┐               │
│  │  Generative Decoder                            │               │
│  │  (fine-tuned transformer LM)                   │               │
│  │                                                │               │
│  │  Conditioned on: z ⊕ u_t                      │               │
│  │  Output: novel content y_t                     │               │
│  └───────────────────┬───────────────────────────┘               │
│                      │                                           │
│                      ▼                                           │
│  y_t (synthesized, personalized output)                          │
│  r_t (user feedback) ──→ update u_t, Σ_u, Σ_j, fine-tune       │
└──────────────────────────────────────────────────────────────────┘
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

**Concrete data sources (implemented in `hare/data/collect.py`):**
1. **fka/awesome-chatgpt-prompts** (HuggingFace) -- 1,131 act/prompt pairs, CC0 license. Maps act->title, prompt->instructions.
2. **VoltAgent/awesome-agent-skills** (GitHub) -- 300+ agent skills, MIT license. Parsed from structured markdown with YAML frontmatter.
3. **Jeffallan/claude-skills** (GitHub) -- 66 Claude skills across 12 categories. Parsed from markdown.
4. **Built-in curated corpus** -- 20 hand-written skills across 8 categories (in `hare/data/skills.py`).

All external data routed through prompt injection validation pipeline (`_validate_skill`). Expected yield: ~1,200-1,500 validated skills after filtering.

**Additional data sources (not yet implemented, for scaling):**
- danielrosehill/System-Prompt-Library (944 system prompts)
- GitHub API search for repos with "claude skill" or "system prompt" topics
- McAuley-Lab/Amazon-Reviews-2023 (cross-domain validation)
- Goodreads Book Graph (alternative recommendation domain)

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
1. **LinUCB** — standard contextual bandit, no user state evolution (Li et al., 2010)
2. **NeuralUCB** — neural network reward model with UCB (Zhou et al., 2020)
3. **RAG** — retrieve top-k by cosine similarity, generate with LM conditioned on chunks (Lewis et al., 2020). The dominant current paradigm. Query-conditioned, not user-conditioned.
4. **Attention-only** — cross-attention without uncertainty injection (α=0). Tests whether exploration adds value beyond learned attention.
5. **GPT-2 (vanilla)** — fine-tuned generative model without bandit exploration or user modeling. Same output for same input regardless of user.
6. **HARE–user** — HARE without user latent state (query-only Q). Ablation: tests whether user modeling adds value.
7. **Random** — uniform random selection/generation

### 5.2 Metrics
- **Cumulative regret** (simulated environment with known rewards)
- **Generation quality**: BLEU, ROUGE, BERTScore against held-out high-quality items
- **Personalization divergence**: Do different users with the same query get meaningfully different outputs? Measure pairwise cosine distance of outputs across users for identical queries. (RAG = ~0 divergence. HARE should show divergence that correlates with user state distance.)
- **Specificity trajectory**: Attention entropy H(A) over interaction rounds. Should decrease as user uncertainty decreases.
- **Diversity**: intra-list diversity of generated items across users
- **Novelty**: % of generated items not in training set (by cosine similarity threshold)
- **Learning curve**: Per-user reward trajectory over interactions (does the system get better at serving each individual user?)
- **Human evaluation**: small-scale rating of generated skills (1-5 quality, 1-5 relevance, 1-5 personalization, 1-5 novelty). Full protocol in `experiments/study_design.md`

### 5.2B NLP Proof of Concept Experiments

The simulation proves the math works. These experiments prove it works in NLP:

**Experiment A: Conditioned generation quality**
- Fine-tune ConditionedGPT2 on 1,000+ collected Claude Skills
- Measure: does soft prompt conditioning on HARE's z reduce perplexity vs unconditioned GPT-2?
- Expected: conditioned perplexity < unconditioned, because z carries relevant information

**Experiment B: User-conditioned generation divergence**
- Create 5 synthetic user profiles (security, data-eng, frontend, education, productivity)
- Give identical queries, generate with HARE-conditioned decoder for each user
- Measure: pairwise cosine distance of generated text embeddings across users
- Compare: RAG baseline produces near-identical outputs (divergence near 0)
- Visualize: UMAP of generated embeddings, colored by user profile

**Experiment C: The killer demo**
- 5 users, same query: "I need help writing tests"
- HARE generates 5 different skills: security testing, data validation, component testing, etc.
- RAG generates 1 skill (the same for all users)
- Include as a qualitative figure in the paper -- visually irrefutable

**Experiment D: Learning curve on real text**
- Track BERTScore of generated skills against held-out ground truth over 10 interaction rounds
- HARE's score should improve (user model refines); baselines remain flat

### 5.3 Ablation Studies
- HARE with vs. without uncertainty injection (α = 0)
- HARE with vs. without fine-tuning (frozen GPT-2 decoder)
- Effect of number of attention heads
- Effect of α schedule (fixed, linear decay, learned)
- Covariance update frequency

### 5.4 Key Hypotheses
1. **HARE > LinUCB**: Lower regret on non-linear reward functions (attention captures interactions bandits miss)
2. **HARE > RAG**: Higher user-rated relevance and quality, especially after multiple interactions (user modeling + exploration)
3. **Exploration helps**: Uncertainty injection improves generation diversity without sacrificing quality (α > 0 beats α = 0)
4. **User state matters**: HARE with user modeling beats HARE without it, and the gap *widens* over interactions (the system learns)
5. **Specificity emerges**: Attention entropy decreases over interactions (output becomes more specific as user uncertainty decreases)
6. **Synthesis > retrieval**: Generated output is rated higher than best-matching existing item (the ideal item isn't in the catalogue)

---

## 6. Paper Outline

**Title:** HARE: Hybrid Attention-Reinforced Exploration for Generative Recommendation

**Abstract:** ~200 words. LLMs generate the same output for the same input regardless of who's asking. RAG improves grounding but not personalization. Recommender systems personalize selection but can't generate. HARE unifies these: a user latent state model with uncertainty-augmented attention over a knowledge pool, decoded into novel synthesized content. The system learns what each user needs through principled exploration, with specificity emerging naturally from uncertainty reduction.

### Sections
1. **Introduction** — The personalization gap: LLMs don't model users, RAG doesn't personalize retrieval, recommenders can't generate. The specificity paradox (too general = slop, too specific = overfitting). Motivate with Claude Skills.
2. **Related Work**
   - Retrieval-augmented generation (RAG, REALM, RETRO) — and why query-conditioned retrieval is insufficient
   - Contextual bandits (LinUCB, NeuralUCB, Thompson Sampling)
   - Attention-based recommendation (DIN, BERT4Rec, SASRec)
   - Generative recommendation (GPT4Rec, generative retrieval)
   - User modeling and latent factor models (PMF, factorization machines)
   - Exploration in neural models (curiosity-driven, information gain)
3. **Method**
   - Problem formulation (user latent state, knowledge pool, evolving uncertainty)
   - User state model and Bayesian update mechanism
   - Uncertainty-augmented attention (the core contribution)
   - Generative synthesis decoder
   - Online learning: the explore → generate → feedback → learn loop
   - How specificity emerges from attention entropy
   - Regret analysis (theoretical contribution)
4. **Experiments**
   - Simulated environment (controlled regret + specificity analysis)
   - Claude Skills generation (primary real-world domain)
   - Study plan generation (secondary domain)
   - RAG comparison (same knowledge pool, with/without user modeling)
   - Ablation studies (α, user state, attention heads, feedback loop)
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
