# HARE — Mathematical Derivations & Paper Notes

## 1. Core Formulation

### Standard LinUCB (Li et al., 2010)

For each arm a, maintain:
- `A_a = D_a^T D_a + I_d` (design matrix, d×d)
- `b_a = D_a^T c_a` (reward-weighted features, d×1)
- `θ_a = A_a^{-1} b_a` (ridge estimate)

Selection rule:
```
a_t = argmax_a [x_t^T θ_a + α √(x_t^T A_a^{-1} x_t)]
```

Update on observing reward r_t for chosen arm a_t:
```
A_{a_t} ← A_{a_t} + x_t x_t^T
b_{a_t} ← b_{a_t} + r_t x_t
```

Regret bound: `R(T) = O(d √(T log T))` (Abbasi-Yadkori et al., 2011)

---

## 1B. Relationship to RAG — Why HARE Is Fundamentally Different

### RAG Pipeline (Lewis et al., 2020)
```
q = encode(query)
D = top_k(cosine(q, doc_embeddings))     # retrieval: query-dependent, user-independent
output = LLM(query, D)                    # generation: conditioned on query + docs
```

**Critical limitation**: The retrieval step depends only on the query. Two users asking the same question get the same documents and substantially similar outputs. RAG improves *grounding* but not *personalization*. It has no user model, no exploration mechanism, no learning from feedback.

### HARE Pipeline
```
q_user = encode([query ⊕ u_t])           # encode query + user latent state
A = softmax(QK^T/√d + α·U(Σ_u, Σ_j))   # attend with uncertainty bonuses
z = A · V                                 # soft synthesis (not hard top-k)
output = Decoder(z, u_t)                  # generate conditioned on synthesis + user state
u_{t+1}, Σ_{t+1} = update(u_t, Σ_t, r_t) # learn from feedback
```

**Key differences:**
1. Retrieval is **user-conditioned** — different users with the same query attend to different knowledge
2. Retrieval is **soft** — attention-weighted synthesis, not hard top-k selection
3. Retrieval is **uncertainty-aware** — explores under-observed knowledge regions
4. The system **learns** — user state and uncertainty update from feedback
5. Generation is **user-conditioned** — decoder sees user state, not just retrieved content

### The Specificity Mechanism

The attention entropy H(A) naturally controls output specificity:
```
H(A) = -Σ_j A_j log A_j
```

- New user: high Σ_u → large U terms → flatter A → higher H(A) → broader output
- Known user: low Σ_u → small U terms → peaked A → lower H(A) → specific output

This is NOT a hyperparameter. It EMERGES from interaction history. The algorithm discovers the right level of personalization for each user-problem pair.

---

## 2. HARE: Attention-UCB Fusion

### 2.1 User Latent State Model

The user state `u_t ∈ R^p` is a learned embedding that captures:
- Domain expertise (how much they know)
- Specificity preference (broad vs. deep)
- Latent goals (what they're trying to accomplish)
- Communication style (how they prefer information)

Update rule (Bayesian linear regression analogue):
```
u_{t+1} = u_t + Σ_u(t) · ∇_u log P(r_t | y_t, u_t)   # posterior update
Σ_u(t+1) = Σ_u(t) - Σ_u(t) · x_t x_t^T · Σ_u(t) / (1 + x_t^T Σ_u(t) x_t)  # uncertainty reduction
```

This is the mechanism that makes the system "figure out" what the user needs:
- Each reward signal narrows the uncertainty about the user
- The uncertainty reduction is proportional to how "informative" the interaction was
- High-reward responses in novel dimensions provide the most information

### 2.2 Uncertainty-Augmented Attention

Given:
- User-conditioned query: `q = W_Q [x_query ⊕ u_t] ∈ R^{d_k}`
- Knowledge keys: `K = W_K X_knowledge ∈ R^{N×d_k}`
- Knowledge values: `V = W_V X_knowledge ∈ R^{N×d_v}`

Standard attention:
```
A_std = softmax(qK^T / √d_k)
```

HARE attention adds exploration bonus:
```
U_j = √(q^T Σ_j^{-1} q)    for each knowledge cluster j
A_hare = softmax(qK^T / √d_k + α · U)
```

Where Σ_j is the covariance matrix for knowledge cluster j, updated online:
```
Σ_j ← Σ_j + Σ_{t: a_t ∈ cluster_j} q_t q_t^T
```

Note: q now includes user state, so Σ_j captures uncertainty about how THIS user relates to this knowledge region — not just global query-knowledge similarity.

### 2.2 Multi-Head Extension

For H heads:
```
head_h = Attention_hare(W_Q^h x, W_K^h X, W_V^h X, α_h, {Σ_j})
z = Concat(head_1, ..., head_H) W_O
```

Each head can have its own α_h, allowing different exploration-exploitation tradeoffs.

### 2.3 Generative Output

The attended representation z is fed to a decoder:
```
y = Decoder(z ⊕ x_user)
```

Where ⊕ is concatenation and Decoder is a fine-tuned language model.

---

## 3. Regret Analysis (Sketch)

### Theorem (informal)
Under assumptions:
1. Rewards are σ-sub-Gaussian conditioned on context
2. Attention weights are bounded: `max_j A_hare[j] ≤ 1`
3. The true reward function is in the RKHS of the attention kernel

HARE achieves regret:
```
R(T) = O(d √(T log T) · √H · (1 + α²/d_k))
```

### Proof sketch
1. Decompose regret into attention approximation error + exploration cost
2. The attention approximation error is bounded by the expressiveness of the attention kernel (universal approximation for softmax attention — Yun et al., 2020)
3. The exploration cost follows the standard LinUCB analysis but applied to cluster-level arms
4. The √H factor comes from combining H independent heads
5. The (1 + α²/d_k) factor is the cost of uncertainty injection — when α=0, we recover attention-only (no exploration)

**TODO:** Formalize assumptions, tighten bounds, handle non-stationarity.

---

## 4. Connection to Existing Theory

### Kernel bandits
HARE can be viewed as a kernel bandit where the kernel is defined by the attention mechanism:
```
k(x, x') = softmax(W_Q x · (W_K x')^T / √d_k)
```

This connects to KernelUCB (Valko et al., 2013) but with a learned, data-dependent kernel.

### Information-directed sampling
The uncertainty injection can also be interpreted through the lens of information-directed sampling (Russo & Van Roy, 2014): we're directing attention toward items where we have high information gain, not just high expected reward.

### Generative retrieval
HARE generalizes the generative retrieval paradigm (De Cao et al., 2021; Tay et al., 2022) by adding principled exploration. Existing generative retrieval systems have no mechanism to handle the explore-exploit tradeoff.

---

## 5. Open Questions

1. **Convergence of online covariance updates with attention**: Does the interplay between updating Σ and updating attention weights converge?
2. **Credit assignment**: When the generated output is novel (not in training set), how do we assign reward credit to specific attention weights?
3. **Scaling**: Can we approximate the per-cluster covariance efficiently for very large item pools?
4. **α learning**: Can we learn α end-to-end via meta-gradient, or does this destabilize training?

---

## 5.1 The Generative Recommendation Thesis

The fundamental claim: **every domain where recommendation means "pick from a catalogue" is a domain where the ideal output may not exist in the catalogue.**

This is why Claude Skills is the perfect proving ground:
- Structured enough to evaluate quality (valid markdown, correct format)
- Open-ended enough that generation adds real value (infinite possible skills)
- Practically useful — synthesized output is immediately deployable
- Novel application — no prior work on "skill recommendation as generation"

But the thesis generalizes. The pattern everywhere is:
1. A pool of existing items with rich features (attention K/V)
2. A user context with specific needs (attention Q)
3. The optimal output is a synthesis, not a selection
4. There's an explore-exploit tradeoff in what to synthesize

### Domain Transfer Candidates (for paper §6 and follow-up work)

**Education — Curriculum Synthesis**
Don't recommend an existing lecture. Generate a custom learning module that bridges this student's specific gaps. The attention mechanism weights relevant pedagogical content; the uncertainty bonus explores unfamiliar teaching approaches.

**Scientific Literature — Research Brief Generation**
Don't recommend papers to read. Synthesize a novel brief that integrates findings, highlights contradictions, identifies gaps. Attention over paper embeddings, generation conditioned on the researcher's question.

**Drug Discovery — Compound Proposal**
Don't screen existing compound libraries. Synthesize novel molecular feature vectors that blend properties of known compounds, biased toward under-explored chemical space. Uncertainty injection = exploring unknown regions of the molecular landscape.

**Commerce — Product Design**
Don't recommend existing products. Synthesize product specifications. With generative manufacturing (3D printing, on-demand), the gap between "recommend" and "design" collapses — HARE fills it.

**Finance — Portfolio Construction**
Don't select existing funds. Synthesize novel portfolio weightings. Uncertainty injection maps directly to risk/reward tradeoff.

**Marketing — Campaign Generation**
Don't recommend existing templates. Synthesize novel ad copy, subject lines, campaign briefs. Bandit exploration = generating the A/B variants to test, not testing existing ones.

---

## 6. Related Work Notes

### Must-cite
- Li et al. (2010) -- LinUCB, Yahoo contextual bandits
- Vaswani et al. (2017) -- Transformer attention mechanism
- Zhou et al. (2020) -- NeuralUCB
- Abbasi-Yadkori et al. (2011) -- Improved LinUCB regret bounds
- Kang & McAuley (2018) -- SASRec (self-attention for sequential rec)
- Sun et al. (2019) -- BERT4Rec
- Petrov & Macdonald (2023) -- GPT4Rec, generative recommendation
- Lewis et al. (2020) -- RAG: Retrieval-Augmented Generation
- Guu et al. (2020) -- REALM: Retrieval-augmented language model pre-training
- **Chowdhury et al. (WACV 2025)** -- Bandit-based attention in Vision Transformers. CLOSEST prior work: injects UCB into attention scores. But (a) vision only, not NLP/rec, (b) uses bandits for token pruning efficiency, not exploration in recommendation. HARE must cite and differentiate.
- **Srber et al. (NeurIPS 2024 Spotlight)** -- VPL: Variational Preference Learning. Learns user-specific latent z for RLHF. 10-25% improvement in reward prediction. Close to HARE's user modeling, but no exploration mechanism and no generative synthesis from attended knowledge.
- **Salemi et al. (ACL 2024)** -- LaMP benchmark: 7 personalized LLM tasks. THE standard benchmark for personalized generation. HARE should evaluate on LaMP tasks.
- **Zhai et al. (ICML 2024, Meta)** -- HSTU: trillion-parameter generative recommendation. 12.4% production gains. Shows industry cares. But no exploration and no user-conditioned attention.
- **Rajput et al. (NeurIPS 2023)** -- TIGER: generative retrieval with Semantic IDs. Predicts item ID sequences. Still selects existing items, does not synthesize new ones.

### Should-cite
- Riquelme et al. (2018) -- Deep Bayesian bandits
- Valko et al. (2013) -- Kernel bandits
- Russo & Van Roy (2014) -- Information-directed sampling
- De Cao et al. (2021) -- Autoregressive entity retrieval
- Tay et al. (2022) -- DSI, differentiable search index
- Borgeaud et al. (2022) -- RETRO: improving LMs with retrieved chunks
- Salakhutdinov & Mnih (2008) -- Bayesian probabilistic matrix factorization
- Rendle (2010) -- Factorization machines
- Li et al. (2024, P-RLHF) -- Personalized Language Modeling from Personalized Human Feedback. Joint user model + reward model from preference data.
- Anonymous (EMNLP 2024) -- Efficient Personalized Large Language Models. Conditions on user history, selects model components per-user.
- Zhang et al. (2024) -- PPlug: persona embeddings as plug-in for fixed LLMs. 1.4-35.8% improvement on LaMP. Static profiles, no exploration.
- Geng et al. (RecSys 2022) -- P5: Recommendation as Language Processing. Unifies rec tasks as text-to-text via T5.
- Wang et al. (AAAI 2022) -- Context Uncertainty in Contextual Bandits (REN). Representation learning + exploration in latent space, but does not modify attention mechanism.
- arXiv 2506.21931 (2025) -- ARAG: Agentic Retrieval Augmented Generation. Multi-agent personalized RAG. Up to 42% NDCG@5 improvement. Uses agents not bandits; no exploration mechanism.
- EMNLP 2024 -- Crafting Personalized Agents via RAG on Editable Memory Graphs.
- Bauer (ACM TRS, 2024) -- RecSys Evaluation Landscape survey. Urges multi-domain evaluation.

### Novel positioning (what makes HARE different from all of the above)

| Component | Closest prior work | HARE's differentiation |
|-----------|-------------------|----------------------|
| Bandit-augmented attention | Chowdhury et al. (WACV 2025) | Vision-only, token pruning. HARE: NLP/rec, exploration for synthesis |
| User latent state | VPL (NeurIPS 2024), PPlug | Static or reward-only. HARE: Bayesian online update with exploration |
| Generative synthesis | GPT4Rec, TIGER, HSTU | Generate item IDs or ranked lists. HARE: generates novel content |
| Personalized retrieval | ARAG (2025), LaMP RAG | Agent-based or static profiles. HARE: uncertainty-driven attention |
| Exploration mechanism | LinUCB, NeuralUCB, REN | Arm selection only. HARE: exploration injected into attention itself |

The three-way intersection -- (1) bandit exploration in (2) transformer attention for (3) generative recommendation -- has NO existing work. Individual pairs exist (bandits+attention in vision, bandits+recommendation without attention modification, attention+generation without exploration). HARE is the first to combine all three.

---

## 7. The Steve Jobs Principle

> "People don't know what they want until you show it to them. That's why I
> never rely on market research. Our task is to read things that are not yet
> on the page." -- Steve Jobs

This quote captures HARE's philosophical foundation. Traditional recommender
systems are market research: they ask users what they liked (ratings, clicks)
and serve more of the same. They optimize for revealed preference over a
fixed catalogue. They cannot discover latent needs the user hasn't articulated.

HARE inverts this: instead of asking "what did you like?" and matching,
it asks "what do you *need* that you haven't seen yet?" -- and synthesizes it.

The uncertainty-augmented attention is the mechanism for "reading things that
are not yet on the page." By exploring uncertain regions of the knowledge
space conditioned on an evolving user model, HARE discovers what the user
needs before the user can articulate it. This is the explore-exploit tradeoff
applied to user understanding, not just item selection.

In the paper, frame this as: **HARE optimizes for latent user need, not
revealed preference. The exploration mechanism discovers needs that
selection-based systems cannot even represent.**

---

## 8. NLP Proof of Concept -- Academic Defense Strategy

The key challenge for HARE is demonstrating that it works in real NLP
contexts, not just synthetic simulations. Here is the evidence chain needed
to make the paper defensible:

### 8.1 Three Layers of Evidence

**Layer 1: Controlled simulation (already done)**
- Synthetic environment with known reward structure
- Proves: HARE's attention-UCB mechanism outperforms LinUCB, RAG, and
  ablations on cumulative reward
- Proves: entropy decreases (specificity emerges)
- Proves: user state matters (ablation shows gap widens over time)
- Weakness: synthetic data, synthetic reward -- reviewers will ask
  "does this transfer to real text?"

**Layer 2: Text-domain experiments (critical for NLP venue)**
- Train ConditionedGPT2 on real Claude Skills data
- Evaluate generated skills with automated metrics:
  - Perplexity: does conditioning on z reduce perplexity vs unconditioned?
  - BERTScore: is the generated skill semantically close to held-out ground truth?
  - Personalization divergence: do different user states produce measurably
    different outputs (cosine distance of generated text embeddings)?
  - Structural validity: does the output follow Claude Skill format?
    (parseable markdown, has title/trigger/instructions sections)
- Compare: HARE-conditioned GPT2 vs vanilla fine-tuned GPT2 vs RAG+GPT2
- This proves the attention-UCB synthesis vector z carries meaningful
  information into the decoder -- the soft prompt prefix does useful work

**Layer 3: Human evaluation (strongest evidence)**
- A/B study (see study_design.md)
- Proves: humans perceive HARE outputs as more relevant and personalized
- Proves: perceived quality improves over interaction rounds
- Even a small pilot (N=20) with significant results is publishable

### 8.2 Automated NLP Metrics -- What to Compute

For the Claude Skills domain, compute these for each method:

| Metric | What it measures | HARE should... |
|--------|-----------------|----------------|
| Perplexity (conditioned) | How well z predicts the target text | Be lower than unconditioned |
| BERTScore (F1) | Semantic similarity to held-out skill | Be comparable or higher than RAG |
| BLEU-4 | N-gram overlap with reference | Be comparable (not key metric) |
| ROUGE-L | Longest common subsequence | Be comparable (not key metric) |
| Self-BLEU (diversity) | Diversity across outputs for same query | Be lower (more diverse per-user) |
| Skill format accuracy | % outputs with valid title/trigger/instructions | Be >= 90% |
| Personalization divergence | Pairwise cosine distance across users | Be significantly > 0 |
| Entropy trajectory | H(attention) over interaction rounds | Decrease monotonically |

### 8.3 The "Does It Actually Work in NLP?" Experiment

The most convincing single experiment for an NLP reviewer:

1. Take 5 distinct "user profiles" (defined by interaction history with
   different skill categories: security, data-eng, frontend, etc.)
2. Give all 5 users the same query: "I need help writing tests"
3. Generate outputs from HARE (conditioned on each user's state) and RAG
4. Show: HARE generates 5 meaningfully different skills, each reflecting
   the user's domain. Security user gets security-focused testing skill.
   Data-eng user gets data validation testing skill. Frontend user gets
   component testing skill.
5. RAG generates the same output for all 5 users (because the query is
   identical and RAG is user-independent).

This is a simple, visual, irrefutable demonstration of user-conditioned
generation. Include it as a qualitative figure in the paper.

### 8.4 Positioning Against Reviewer Objections

**"This is just prompt engineering"**
No. Prompt engineering conditions on explicit user instructions. HARE
conditions on a *learned latent state* that evolves from implicit feedback.
The user never tells HARE what they need -- HARE discovers it through
exploration.

**"Why not just use RAG with a user profile?"**
RAG with a user profile would concatenate profile text with the query for
retrieval. This is: (a) hard-coded, not learned; (b) uses top-k hard
retrieval, not soft attention-weighted synthesis; (c) has no exploration
mechanism to discover what the profile is missing; (d) does not update
from feedback. Ablation study shows HARE outperforms this approach.

**"The bandit formulation is not novel"**
Injecting UCB-style uncertainty into transformer attention scores IS novel.
LinUCB selects arms. NeuralUCB uses neural networks for reward estimation.
Neither modifies the attention mechanism itself. The closest work is
Chowdhury et al. (WACV 2025) who inject UCB into vision transformer
attention -- but for token pruning efficiency in image classification,
not exploration in recommendation. HARE's contribution is bringing this
to NLP/recommendation with user-conditioned exploration for generative
synthesis. Different domain, different purpose, different mechanism.

**"GPT-2 is too small to demonstrate this"**
The contribution is the attention-UCB-synthesis framework, not the decoder.
GPT-2 demonstrates the mechanism works. The architecture is decoder-agnostic.
Scaling to larger models is future work and would only strengthen results.

**"N=20 is too small for human evaluation"**
Pilot study framing. N=20 with within-subjects design (each participant
rates all methods) gives 800+ paired comparisons. Medium effects
(Cohen's d >= 0.45) are detectable at 80% power. If effects are smaller
than medium, the contribution is the framework + automated metrics, with
human eval as supporting evidence.

### 8.5 Related Work to Cite for NLP Defense

**Personalized generation (must-cite for NLP venues):**
- Salemi et al. (ACL 2024) -- LaMP benchmark. 7 personalized LLM tasks.
  RAG personalization yields +14.92%, RAG+PEFT yields +15.98%.
  HARE should evaluate on LaMP to show exploration adds value beyond these.
- Srber et al. (NeurIPS 2024 Spotlight) -- VPL: learns user-specific latent z
  for reward models. 10-25% improvement. Closest to HARE's user modeling
  but no exploration mechanism and no generative synthesis from knowledge pool.
- PPlug (arXiv 2024) -- persona embeddings plugged into frozen LLMs.
  1.4-35.8% improvement on LaMP. Static profiles, no online learning.
- P-RLHF (Li et al., 2024) -- joint user model + reward model from
  preference data. Maps user info into user-specific embeddings.

**Bandit + attention (must-cite, primary novelty claim):**
- Chowdhury et al. (WACV 2025) -- UCB in vision transformer attention.
  THE closest prior work. 30-36% training time reduction. But vision-only,
  pruning-focused. HARE: NLP, exploration-focused, for recommendation.
- Wang et al. (AAAI 2022) -- REN: exploration in latent space for recommendation.
  But does not modify attention mechanism. HARE injects exploration INTO attention.

**Generative recommendation (context papers):**
- Zhai et al. (ICML 2024, Meta) -- HSTU: trillion-parameter generative rec.
  12.4% production gains. Shows industrial relevance of generative rec.
- Rajput et al. (NeurIPS 2023) -- TIGER: generative retrieval with Semantic IDs.
  Still predicts existing item IDs, not novel content.
- Geng et al. (RecSys 2022) -- P5: recommendation as text-to-text.
- GenRec (ECIR 2024), InstructRec (2023) -- LLM-based recommendation.

**Personalized RAG (differentiation papers):**
- ARAG (arXiv 2025) -- multi-agent personalized RAG. Up to 42% NDCG@5
  improvement. Uses agent routing, not bandit exploration. No synthesis.
- EMNLP 2024 -- Personalized agents via RAG on editable memory graphs.

**Surveys to cite:**
- KDD 2024 -- "A Review of Modern Recommender Systems Using Generative Models"
- LREC-COLING 2024 -- "LLMs for Generative Recommendation: Survey"
- arXiv 2025 -- "Multi-Armed Bandits Meet Large Language Models" (survey)
- Bauer (ACM TRS, 2024) -- RecSys evaluation landscape

### 8.6 Evaluation Benchmarks Strategy

**Primary (Claude Skills) -- novel domain, shows HARE's unique value:**
- Fine-tune ConditionedGPT2 on 1,200+ collected skills
- Evaluate: BERTScore, perplexity, personalization divergence, format accuracy
- Qualitative demo: 5 users, same query, 5 different outputs

**Secondary (LaMP benchmark) -- standard benchmark, shows generalizability:**
- LaMP has 7 tasks: 3 classification, 4 generation
- Focus on generation tasks: personalized headline, email subject, review
- Standard metrics: ROUGE, MAE, accuracy
- Compare HARE against LaMP baselines (RAG, PEFT, RAG+PEFT)
- Key result to show: HARE outperforms RAG+PEFT after interaction rounds
  (LaMP baselines are static; HARE learns)

**Tertiary (Amazon Reviews subset) -- cross-domain validation:**
- Use Amazon Reviews 2023 (McAuley-Lab) for a non-skill domain
- Task: generate personalized review summaries or product descriptions
- Standard metrics: Recall@K, NDCG@K (for retrieval), BERTScore (for generation)
- Shows HARE is not Claude-Skills-specific

### 8.7 Minimum Viable NLP Experiment for Paper

If time is limited, the absolute minimum NLP experiment that makes the
paper publishable at a venue like RecSys or an NLP workshop:

1. Fine-tune ConditionedGPT2 on 500+ Claude Skills
2. Generate skills for 5 user profiles x 10 queries = 50 outputs
3. Compute: BERTScore, personalization divergence, format accuracy
4. Show learning curve: perplexity decreases over interaction rounds
5. Qualitative example (the 5-users-same-query demonstration)
6. Compare against vanilla GPT-2 and RAG baselines

This is achievable in 2-3 days of compute + analysis.

### 8.8 Target Venues (updated with landscape knowledge)

**Tier 1 (reach):**
- RecSys 2026 (primary -- generative recommendation is THE hot topic)
- NeurIPS 2026 (if we can formalize the regret bound properly)

**Tier 2 (solid fit):**
- AAAI 2027
- EMNLP 2026 (if NLP experiments are strong)
- GenAIRecP 2025/2026 Workshop (co-located with major conferences,
  specifically about generative AI + personalization in recommendation)

**Tier 3 (fallback):**
- ACL workshop on personalization
- KDD workshop on LLM+RecSys
- arXiv preprint for visibility while revising
