"""Uncertainty-Augmented Multi-Head Cross-Attention.

The core mechanism of HARE: standard scaled dot-product attention with an
additive exploration bonus derived from per-cluster covariance matrices.

Two users with the same query get different attention patterns because their
user states differ, which changes both relevance scores (QK^T) and uncertainty
bonuses (U).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _softmax(x: NDArray, axis: int = -1) -> NDArray:
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


class UncertaintyTracker:
    """Maintains per-cluster covariance matrices for UCB-style exploration.

    Each knowledge cluster j has a covariance matrix Σ_j that accumulates
    observed query-reward interactions. The uncertainty for a new query q
    w.r.t. cluster j is √(q^T Σ_j^{-1} q).

    Parameters
    ----------
    n_clusters : int
        Number of knowledge clusters.
    d : int
        Dimension of the query vectors (after projection).
    """

    def __init__(self, n_clusters: int, d: int) -> None:
        self.n_clusters = n_clusters
        self.d = d
        # Per-cluster: Σ_j = I + Σ_t q_t q_t^T (regularized covariance)
        self.sigma = np.array([np.eye(d) for _ in range(n_clusters)])
        self._sigma_inv = np.array([np.eye(d) for _ in range(n_clusters)])

    def get_uncertainty(self, query: NDArray, cluster_assignments: NDArray) -> NDArray:
        """Compute UCB-style uncertainty for each item given query.

        Parameters
        ----------
        query : array of shape (d,)
            The projected query vector [x_query ⊕ u_t] after W_Q.
        cluster_assignments : array of shape (n_items,)
            Integer cluster ID for each item in the knowledge pool.

        Returns
        -------
        array of shape (n_items,)
            Uncertainty bonus U_j = √(q^T Σ_j^{-1} q) for each item's cluster.
        """
        q = query.ravel()
        n_items = len(cluster_assignments)
        uncertainties = np.empty(n_items)

        for i in range(n_items):
            j = cluster_assignments[i]
            uncertainties[i] = np.sqrt(np.abs(q @ self._sigma_inv[j] @ q))

        return uncertainties

    def update(self, query: NDArray, cluster: int) -> None:
        """Update cluster covariance after observing an interaction.

        Parameters
        ----------
        query : array of shape (d,)
            The query vector from this interaction.
        cluster : int
            Which cluster was attended to.
        """
        q = query.ravel()
        self.sigma[cluster] += np.outer(q, q)
        # Sherman-Morrison update for the inverse
        S_inv = self._sigma_inv[cluster]
        u = S_inv @ q
        self._sigma_inv[cluster] = S_inv - np.outer(u, u) / (1.0 + q @ u)


class MultiHeadCrossAttention:
    """Multi-head cross-attention with uncertainty-augmented scores.

    Implements:
        Q = W_Q · [x_query ⊕ u_t]
        K = W_K · X_knowledge
        V = W_V · X_knowledge
        U_j = √(q^T Σ_j^{-1} q)               (per-cluster uncertainty)
        A = softmax(QK^T / √d_k + α · U)       (uncertainty-augmented attention)
        z = A · V                                (synthesized representation)

    Parameters
    ----------
    d_input : int
        Dimension of input vectors (query and knowledge items).
    d_k : int
        Dimension of key/query projections per head.
    d_v : int
        Dimension of value projections per head.
    n_heads : int
        Number of attention heads.
    n_clusters : int
        Number of knowledge clusters for uncertainty tracking.
    alpha : float
        Exploration parameter. Higher = more exploration.
    seed : int | None
        Random seed for weight initialization.
    """

    def __init__(
        self,
        d_input: int,
        d_k: int = 64,
        d_v: int = 64,
        n_heads: int = 4,
        n_clusters: int = 10,
        alpha: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self.d_input = d_input
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.alpha = alpha

        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / (d_input + d_k))

        # Per-head projection matrices
        self.W_Q = rng.normal(0, scale, (n_heads, d_input, d_k))
        self.W_K = rng.normal(0, scale, (n_heads, d_input, d_k))
        self.W_V = rng.normal(0, np.sqrt(2.0 / (d_input + d_v)), (n_heads, d_input, d_v))
        self.W_O = rng.normal(0, np.sqrt(2.0 / (n_heads * d_v + d_input)), (n_heads * d_v, d_input))

        # Per-head uncertainty trackers
        self.uncertainty_trackers = [
            UncertaintyTracker(n_clusters, d_k) for _ in range(n_heads)
        ]

    def forward(
        self,
        query: NDArray,
        keys: NDArray,
        values: NDArray,
        cluster_assignments: NDArray,
        return_weights: bool = False,
    ) -> NDArray | tuple[NDArray, NDArray]:
        """Compute uncertainty-augmented multi-head cross-attention.

        Parameters
        ----------
        query : array of shape (d_input,)
            The user-conditioned query vector [x_query ⊕ u_t].
        keys : array of shape (n_items, d_input)
            Knowledge pool item features.
        values : array of shape (n_items, d_input)
            Knowledge pool item features (can be same as keys).
        cluster_assignments : array of shape (n_items,)
            Cluster ID per item.
        return_weights : bool
            If True, also return attention weights.

        Returns
        -------
        z : array of shape (d_input,)
            Synthesized representation (output-projected).
        attn_weights : array of shape (n_heads, n_items), optional
            Attention weights per head (only if return_weights=True).
        """
        q = query.ravel()
        n_items = keys.shape[0]
        head_outputs = []
        all_weights = []

        for h in range(self.n_heads):
            # Project query, keys, values
            q_h = q @ self.W_Q[h]          # (d_k,)
            K_h = keys @ self.W_K[h]       # (n_items, d_k)
            V_h = values @ self.W_V[h]     # (n_items, d_v)

            # Scaled dot-product scores
            scores = (K_h @ q_h) / np.sqrt(self.d_k)  # (n_items,)

            # Uncertainty bonus
            U = self.uncertainty_trackers[h].get_uncertainty(q_h, cluster_assignments)
            scores = scores + self.alpha * U

            # Attention weights
            weights = _softmax(scores)  # (n_items,)
            all_weights.append(weights)

            # Weighted sum of values
            head_out = weights @ V_h    # (d_v,)
            head_outputs.append(head_out)

        # Concatenate heads and project
        concat = np.concatenate(head_outputs)  # (n_heads * d_v,)
        z = concat @ self.W_O                  # (d_input,)

        if return_weights:
            return z, np.array(all_weights)
        return z

    def update_uncertainty(self, query: NDArray, head: int, cluster: int) -> None:
        """Update uncertainty tracker for a specific head and cluster."""
        q_h = query.ravel() @ self.W_Q[head]
        self.uncertainty_trackers[head].update(q_h, cluster)

    def get_attention_entropy(self, weights: NDArray) -> float:
        """Compute mean attention entropy across heads.

        Higher entropy = more spread out attention = more general output.
        Lower entropy = more concentrated = more specific output.
        """
        # weights shape: (n_heads, n_items)
        eps = 1e-12
        entropies = -np.sum(weights * np.log(weights + eps), axis=1)
        return float(np.mean(entropies))
