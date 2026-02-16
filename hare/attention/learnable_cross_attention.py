"""Learnable Uncertainty-Augmented Multi-Head Cross-Attention (PyTorch).

Fixes the core issue with the NumPy MultiHeadCrossAttention: projection matrices
are now nn.Parameters initialized with Xavier/Glorot, enabling gradient-based
training of the attention mechanism.

Key architectural changes from the original:
1. Separate query projections for content (W_Q_x) and user state (W_Q_u)
   — the user state is the personalization lever
2. Knowledge embeddings are NOT zero-padded — W_K and W_V project from
   d_knowledge directly, preserving full capacity
3. Output W_O projects back to d_knowledge space (not d_knowledge + d_user)
4. UncertaintyTracker stays in NumPy (online Bayesian, non-differentiable)
   — projected query is detached before UCB computation
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray

from hare.attention.cross_attention import UncertaintyTracker


class LearnableCrossAttention(nn.Module):
    """Learnable multi-head cross-attention with uncertainty-augmented scores.

    Implements:
        Q_h = x_query @ W_Q_x[h] + u_t @ W_Q_u[h]   # separate projections
        K_h = X_knowledge @ W_K[h]                     # no zero-padding
        V_h = X_knowledge @ W_V[h]                     # no zero-padding
        scores = K_h @ Q_h / sqrt(d_k) + alpha * U     # UCB bonus
        A_h = softmax(scores)
        z = concat(heads) @ W_O                         # output in d_knowledge space

    Parameters
    ----------
    d_knowledge : int
        Dimension of knowledge item embeddings.
    d_user : int
        Dimension of user latent state.
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
    """

    def __init__(
        self,
        d_knowledge: int,
        d_user: int,
        d_k: int = 64,
        d_v: int = 64,
        n_heads: int = 4,
        n_clusters: int = 10,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.d_knowledge = d_knowledge
        self.d_user = d_user
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.alpha = alpha

        # Per-head query projections: SEPARATE for content and user state
        # This is the personalization lever — W_Q_u lets user state influence attention
        self.W_Q_x = nn.Parameter(torch.empty(n_heads, d_knowledge, d_k))
        self.W_Q_u = nn.Parameter(torch.empty(n_heads, d_user, d_k))

        # Per-head key/value projections: from d_knowledge only (no zero-padding)
        self.W_K = nn.Parameter(torch.empty(n_heads, d_knowledge, d_k))
        self.W_V = nn.Parameter(torch.empty(n_heads, d_knowledge, d_v))

        # Output projection: back to d_knowledge space
        self.W_O = nn.Parameter(torch.empty(n_heads * d_v, d_knowledge))

        # Xavier/Glorot initialization
        self._init_weights()

        # Per-head uncertainty trackers (NumPy, non-differentiable)
        self.uncertainty_trackers = [
            UncertaintyTracker(n_clusters, d_k) for _ in range(n_heads)
        ]

    def _init_weights(self) -> None:
        """Xavier uniform initialization for all projection matrices."""
        for param_name in ["W_Q_x", "W_K", "W_V"]:
            param = getattr(self, param_name)
            for h in range(self.n_heads):
                nn.init.xavier_uniform_(param[h])

        # W_Q_u: Xavier with fan_in=d_user, fan_out=d_k
        for h in range(self.n_heads):
            nn.init.xavier_uniform_(self.W_Q_u[h])

        # W_O: Xavier uniform
        nn.init.xavier_uniform_(self.W_O)

    def forward(
        self,
        query_embedding: torch.Tensor,
        user_state: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        cluster_assignments: NDArray,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Compute uncertainty-augmented multi-head cross-attention.

        Parameters
        ----------
        query_embedding : tensor of shape (d_knowledge,)
            Content query embedding.
        user_state : tensor of shape (d_user,)
            User latent state vector.
        keys : tensor of shape (n_items, d_knowledge)
            Knowledge pool item embeddings.
        values : tensor of shape (n_items, d_knowledge)
            Knowledge pool item embeddings (can be same as keys).
        cluster_assignments : numpy array of shape (n_items,)
            Cluster ID per item (integer).
        return_weights : bool
            If True, also return attention weights.

        Returns
        -------
        z : tensor of shape (d_knowledge,)
            Synthesized representation.
        attn_weights : tensor of shape (n_heads, n_items), optional
            Attention weights per head (only if return_weights=True).
        """
        x = query_embedding.view(-1)  # (d_knowledge,)
        u = user_state.view(-1)       # (d_user,)

        head_outputs = []
        all_weights = []

        for h in range(self.n_heads):
            # Separate query projections: content + user state
            q_h = x @ self.W_Q_x[h] + u @ self.W_Q_u[h]  # (d_k,)

            # Project keys and values from knowledge (no zero-padding)
            K_h = keys @ self.W_K[h]    # (n_items, d_k)
            V_h = values @ self.W_V[h]  # (n_items, d_v)

            # Scaled dot-product scores
            scores = (K_h @ q_h) / math.sqrt(self.d_k)  # (n_items,)

            # Uncertainty bonus (non-differentiable, from NumPy tracker)
            q_h_np = q_h.detach().cpu().numpy()
            U = self.uncertainty_trackers[h].get_uncertainty(
                q_h_np, cluster_assignments
            )
            U_tensor = torch.tensor(U, dtype=scores.dtype, device=scores.device)
            scores = scores + self.alpha * U_tensor

            # Attention weights
            weights = F.softmax(scores, dim=0)  # (n_items,)
            all_weights.append(weights)

            # Weighted sum of values
            head_out = weights @ V_h  # (d_v,)
            head_outputs.append(head_out)

        # Concatenate heads and project to d_knowledge
        concat = torch.cat(head_outputs)   # (n_heads * d_v,)
        z = concat @ self.W_O              # (d_knowledge,)

        if return_weights:
            return z, torch.stack(all_weights)  # (n_heads, n_items)
        return z

    def get_mean_attention(
        self,
        query_embedding: torch.Tensor,
        user_state: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        cluster_assignments: NDArray,
    ) -> torch.Tensor:
        """Compute mean attention weights across heads (differentiable).

        Returns
        -------
        mean_weights : tensor of shape (n_items,)
            Mean attention weights across all heads.
        """
        _, weights = self.forward(
            query_embedding, user_state, keys, values,
            cluster_assignments, return_weights=True,
        )
        return weights.mean(dim=0)  # (n_items,)

    def update_uncertainty(
        self,
        query_embedding: torch.Tensor,
        user_state: torch.Tensor,
        head: int,
        cluster: int,
    ) -> None:
        """Update uncertainty tracker for a specific head and cluster."""
        with torch.no_grad():
            x = query_embedding.view(-1)
            u = user_state.view(-1)
            q_h = (x @ self.W_Q_x[head] + u @ self.W_Q_u[head]).cpu().numpy()
        self.uncertainty_trackers[head].update(q_h, cluster)

    def get_attention_entropy(self, weights: torch.Tensor) -> float:
        """Compute mean attention entropy across heads.

        Parameters
        ----------
        weights : tensor of shape (n_heads, n_items)
        """
        eps = 1e-12
        entropies = -torch.sum(weights * torch.log(weights + eps), dim=1)
        return float(entropies.mean().item())
