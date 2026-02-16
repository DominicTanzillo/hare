# HARE Regret Bound: Three-Term Decomposition

## Theorem (Informal)

Under HARE with learnable cross-attention and Bayesian user state updates,
the cumulative pseudo-regret satisfies:

$$R(T) = O\left(d_k \sqrt{C \cdot T \cdot \log T}\right)$$

where $d_k$ is the attention key dimension, $C$ is the number of knowledge
clusters, and $T$ is the time horizon.

---

## Setup and Notation

| Symbol | Meaning |
|--------|---------|
| $T$ | Time horizon (number of interactions) |
| $C$ | Number of knowledge clusters |
| $d_k$ | Key/query projection dimension |
| $d_u$ | User state dimension |
| $N$ | Knowledge pool size |
| $u_t$ | User latent state at time $t$ |
| $x_t$ | Query embedding at time $t$ |
| $a_t$ | Attention distribution at time $t$ |
| $a_t^*$ | Oracle attention distribution |
| $r_t$ | Reward at time $t$ |
| $\alpha$ | Exploration parameter |
| $\hat{W}$ | Learned projection matrices |
| $W^*$ | Oracle projection matrices |
| $\Sigma_j$ | Covariance matrix for cluster $j$ |

**Reward model**: $r_t = f(a_t, u_t^*) + \eta_t$ where $f$ is the reward
function, $u_t^*$ is the true user preference, and $\eta_t$ is sub-Gaussian noise.

**Pseudo-regret**: $R(T) = \sum_{t=1}^T \left[f(a_t^*, u_t^*) - f(a_t, u_t^*)\right]$

---

## Three-Term Decomposition

$$R(T) = \underbrace{R_{\text{approx}}(T)}_{\text{attention error}} + \underbrace{R_{\text{explore}}(T)}_{\text{cluster exploration}} + \underbrace{R_{\text{user}}(T)}_{\text{user estimation}}$$

### Term 1: $R_{\text{approx}}(T)$ — Attention Approximation Error

This term captures the gap between the learned attention and the oracle attention,
holding the user state and exploration fixed.

**Source**: Finite-capacity projection matrices $\hat{W} = \{\hat{W}_{Q_x}, \hat{W}_{Q_u}, \hat{W}_K, \hat{W}_V, \hat{W}_O\}$ approximate but don't perfectly match the oracle $W^*$.

**Bound**: If the attention training loss converges to $\epsilon_{\text{train}}$:

$$R_{\text{approx}}(T) \leq L_f \cdot T \cdot \epsilon_{\text{attn}}$$

where $L_f$ is the Lipschitz constant of the reward function w.r.t. the
attention distribution, and:

$$\epsilon_{\text{attn}} = \mathbb{E}\left[\|a_t(\hat{W}) - a_t(W^*)\|_1\right] \leq g(\epsilon_{\text{train}})$$

**Key insight**: Unlike the random-projection HARE where $\epsilon_{\text{attn}}$
is a fixed constant (close to $\log N / N$ due to near-uniform attention), the
learnable version drives $\epsilon_{\text{train}} \to 0$ as training data increases.

For well-specified models with $M$ training samples:
$$\epsilon_{\text{train}} = O\left(\sqrt{\frac{p}{M}}\right)$$

where $p = n_h(d_k \cdot d_{\text{know}} + d_k \cdot d_u + d_k \cdot d_{\text{know}} + d_v \cdot d_{\text{know}}) + n_h \cdot d_v \cdot d_{\text{know}}$ is the parameter count.

In the regime where $M \gg p$, this term becomes $O(1)$ and is dominated by the
other two terms.

---

### Term 2: $R_{\text{explore}}(T)$ — Cluster-Level Exploration Regret

This term is the standard LinUCB regret applied at the cluster level. HARE's
uncertainty tracker maintains per-cluster covariance matrices $\Sigma_j$ and uses
UCB-style bonuses $U_{t,j} = \sqrt{q_t^\top \Sigma_j^{-1} q_t}$.

**Reduction to LinUCB**: At the cluster level, HARE performs a contextual bandit
over $C$ arms (clusters) with $d_k$-dimensional projected contexts. The
attention-weighted selection over items within a cluster is fixed given the
cluster-level decision.

**Standard LinUCB bound** (Li et al., 2010; Abbasi-Yadkori et al., 2011):

$$R_{\text{explore}}(T) \leq \alpha \cdot d_k \sqrt{C \cdot T \cdot \log\left(\frac{T \cdot \lambda_{\max}}{\delta}\right)} + C \cdot d_k \cdot \log T$$

where $\lambda_{\max}$ bounds the eigenvalues of $\Sigma_j$ and $\delta$ is the
confidence parameter.

**Simplified**: $R_{\text{explore}}(T) = O(d_k \sqrt{C \cdot T \cdot \log T})$

This is the dominant term and matches the minimax rate for contextual bandits
up to logarithmic factors.

---

### Term 3: $R_{\text{user}}(T)$ — User State Estimation Error

This term captures the cost of not knowing the true user preferences $u^*$
and instead using the Bayesian posterior estimate $\hat{u}_t$.

**Bayesian posterior concentration**: Under the linear Gaussian model where
rewards are $r_t = x_t^\top u^* + \eta_t$ with $\eta_t \sim \mathcal{N}(0, \sigma^2)$:

$$\|\hat{u}_t - u^*\|_2 \leq \sigma \sqrt{\frac{d_u \log(t/\delta)}{\lambda_{\min}(\Sigma_u^{-1}(t))}}$$

with probability at least $1 - \delta$, where $\Sigma_u^{-1}(t) = I + \sum_{s=1}^t x_s x_s^\top / \sigma^2$.

**Regret contribution**: By the Lipschitz assumption on $f$:

$$R_{\text{user}}(T) \leq L_u \sum_{t=1}^T \|\hat{u}_t - u^*\|_2 \leq L_u \cdot \sigma \sum_{t=1}^T \sqrt{\frac{d_u \log(t/\delta)}{\lambda_{\min}(\Sigma_u^{-1}(t))}}$$

Under standard regularity (bounded feature norms, diverse queries), $\lambda_{\min}$ grows linearly:

$$R_{\text{user}}(T) = O\left(\sigma \cdot d_u \cdot \sqrt{T \cdot \log T}\right)$$

Since typically $d_u \ll d_k \cdot \sqrt{C}$, this term is dominated by $R_{\text{explore}}$.

---

## Combined Bound

$$R(T) = O\left(\underbrace{\sqrt{p/M}}_{\text{train}} \cdot T + \underbrace{d_k \sqrt{C \cdot T \cdot \log T}}_{\text{explore}} + \underbrace{d_u \sqrt{T \cdot \log T}}_{\text{user}}\right)$$

In the **practically relevant regime** where training data $M \gg p \cdot T$
(i.e., the attention is well-trained):

$$\boxed{R(T) = O\left(d_k \sqrt{C \cdot T \cdot \log T}\right)}$$

This matches the minimax rate for $C$-armed contextual bandits with
$d_k$-dimensional contexts.

---

## Formal Assumptions

1. **Sub-Gaussian rewards**: $\eta_t$ is conditionally $\sigma$-sub-Gaussian
   given the history $\mathcal{H}_t$.

2. **Lipschitz reward function**: $|f(a, u) - f(a', u')| \leq L_f \|a - a'\|_1 + L_u \|u - u'\|_2$.

3. **Bounded embeddings**: $\|x_t\|_2 \leq B_x$, $\|u_t\|_2 \leq B_u$ for all $t$.

4. **Cluster regularity**: Items within each cluster have bounded intra-cluster
   variance: $\max_j \text{Var}(\{v_i : c_i = j\}) \leq \sigma_c^2$.

5. **Diverse queries**: The minimum eigenvalue of
   $\sum_{t=1}^T x_t x_t^\top$ grows as $\Omega(T)$ (standard in linear bandits).

6. **Attention training convergence**: The training loss converges as
   $\epsilon_{\text{train}} = O(\sqrt{p/M})$ with $M$ training samples.

---

## Comparison with Prior Work

| Method | Regret Bound | Personalization |
|--------|-------------|-----------------|
| LinUCB (Li et al., 2010) | $O(d\sqrt{KT \log T})$ | Per-arm only |
| UCB-Attention (Chowdhury et al., 2025) | Not analyzed | Vision only |
| VPL (NeurIPS 2024) | $O(\sqrt{T})$ | Yes, but RLHF-specific |
| **HARE (this work)** | $O(d_k\sqrt{CT \log T})$ | User-conditioned attention |

**Key advantage**: HARE's regret bound is in terms of $C$ (number of clusters,
typically 5-20) rather than $N$ (number of items, potentially thousands). The
attention mechanism compresses the action space from $N$ discrete items to $C$
cluster-level decisions, yielding a tighter bound when $C \ll N$.

---

## Proof Sketch for R_explore

**Step 1**: At each time $t$, HARE selects cluster $j_t$ based on:
$$j_t = \arg\max_j \left[\bar{s}_{t,j} + \alpha \cdot U_{t,j}\right]$$
where $\bar{s}_{t,j}$ is the mean attention score for cluster $j$ and
$U_{t,j} = \sqrt{q_t^\top \Sigma_j^{-1} q_t}$.

**Step 2**: This is an instance of OFUL (Optimism in the Face of Uncertainty
for Linear bandits) from Abbasi-Yadkori et al. (2011), applied independently
to each of $C$ clusters.

**Step 3**: By Theorem 2 of Abbasi-Yadkori et al., with probability $1 - \delta$:
$$\sum_{t: j_t = j} U_{t,j} \leq \sqrt{T_j \cdot d_k \cdot \log\left(\frac{1 + T_j B_x^2}{\delta}\right)}$$
where $T_j$ is the number of times cluster $j$ is selected.

**Step 4**: Summing over clusters via Cauchy-Schwarz:
$$R_{\text{explore}}(T) \leq \alpha \sqrt{C \sum_j T_j \cdot d_k \log(\cdot)} = O(\alpha \cdot d_k \sqrt{CT \log T})$$

---

## Discussion: Random vs. Learnable Projections

The three-term decomposition reveals why random projections fail:

- **Random projections**: $\epsilon_{\text{attn}} \approx O(1)$ (attention is
  near-uniform regardless of input), so $R_{\text{approx}} = O(T)$ — linear regret.
  The system never learns to focus attention, and the user state signal through
  $W_{Q_u}$ is effectively noise.

- **Learnable projections**: $\epsilon_{\text{attn}} \to 0$ as training data
  increases, reducing $R_{\text{approx}}$ to $O(1)$. The remaining regret is
  sublinear and matches the contextual bandit minimax rate.

This provides theoretical justification for the empirical observation that
the divergence demo produces 0.99999 pairwise similarity with random projections
but meaningful divergence with learned projections.
