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

## 2. HARE: Attention-UCB Fusion

### 2.1 Uncertainty-Augmented Attention

Given:
- User context query: `q = W_Q x_user ∈ R^{d_k}`
- Item keys: `K = W_K X_items ∈ R^{N×d_k}`
- Item values: `V = W_V X_items ∈ R^{N×d_v}`

Standard attention:
```
A_std = softmax(qK^T / √d_k)
```

HARE attention adds exploration bonus:
```
U_j = √(q^T Σ_j^{-1} q)    for each item cluster j
A_hare = softmax(qK^T / √d_k + α · U)
```

Where Σ_j is the covariance matrix for cluster j, updated online:
```
Σ_j ← Σ_j + Σ_{t: a_t ∈ cluster_j} x_t x_t^T
```

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

## 6. Related Work Notes

### Must-cite
- Li et al. (2010) — LinUCB, Yahoo contextual bandits
- Vaswani et al. (2017) — Transformer attention mechanism
- Zhou et al. (2020) — NeuralUCB
- Abbasi-Yadkori et al. (2011) — Improved LinUCB regret bounds
- Zhou et al. (2020) — Neural contextual bandits
- Kang & McAuley (2018) — SASRec (self-attention for sequential rec)
- Sun et al. (2019) — BERT4Rec
- Petrov & Macdonald (2023) — GPT4Rec, generative recommendation

### Should-cite
- Riquelme et al. (2018) — Deep Bayesian bandits
- Valko et al. (2013) — Kernel bandits
- Russo & Van Roy (2014) — Information-directed sampling
- De Cao et al. (2021) — Autoregressive entity retrieval
- Tay et al. (2022) — DSI, differentiable search index
