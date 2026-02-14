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
- Li et al. (2010) — LinUCB, Yahoo contextual bandits
- Vaswani et al. (2017) — Transformer attention mechanism
- Zhou et al. (2020) — NeuralUCB
- Abbasi-Yadkori et al. (2011) — Improved LinUCB regret bounds
- Kang & McAuley (2018) — SASRec (self-attention for sequential rec)
- Sun et al. (2019) — BERT4Rec
- Petrov & Macdonald (2023) — GPT4Rec, generative recommendation
- Lewis et al. (2020) — RAG: Retrieval-Augmented Generation (the baseline paradigm HARE generalizes)
- Guu et al. (2020) — REALM: Retrieval-augmented language model pre-training

### Should-cite
- Riquelme et al. (2018) — Deep Bayesian bandits
- Valko et al. (2013) — Kernel bandits
- Russo & Van Roy (2014) — Information-directed sampling
- De Cao et al. (2021) — Autoregressive entity retrieval
- Tay et al. (2022) — DSI, differentiable search index
- Borgeaud et al. (2022) — RETRO: improving LMs with retrieved chunks (chunk-level RAG)
- Salakhutdinov & Mnih (2008) — Bayesian probabilistic matrix factorization (user modeling under uncertainty)
- Rendle (2010) — Factorization machines (user-item latent interactions)

### Novel positioning (what makes HARE different from all of the above)
- RAG (Lewis, Guu, Borgeaud): retrieves based on query, not user state. No exploration. No feedback loop.
- Contextual bandits (Li, Zhou): explore/exploit over existing arms, no generation, linear reward models.
- Attention RecSys (Kang, Sun): non-linear interactions, no exploration, selection-only.
- Generative RecSys (Petrov): generates ranked lists of existing items, not novel content.
- User modeling (Salakhutdinov, Rendle): models user-item latent space, but for rating prediction, not generation.
- **HARE uniquely combines**: user latent state modeling + uncertainty-augmented attention + generative synthesis + online exploration. No prior work does all four.
