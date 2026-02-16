"""LearnableHARE: PyTorch-based HARE with trainable attention projections.

Mirrors the interface of the original HARE class (hare.bandits.attentive_bandit)
but uses LearnableCrossAttention with trained projection matrices.

Key differences from HARE:
1. Attention projections are nn.Parameters (trainable via backprop)
2. Knowledge embeddings are NOT zero-padded
3. Output z has shape (d_knowledge,) not (d_knowledge + d_user,)
4. UserState Bayesian updates stay in NumPy (unchanged)
5. NumPy/PyTorch boundary: user state converted to tensor for forward pass,
   z detached back to numpy for user state updates
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from sklearn.cluster import MiniBatchKMeans

from hare.attention.learnable_cross_attention import LearnableCrossAttention
from hare.bandits.attentive_bandit import UserState


class LearnableHARE(nn.Module):
    """Hybrid Attention-Reinforced Exploration with learnable projections.

    Parameters
    ----------
    d_knowledge : int
        Dimension of knowledge item embeddings.
    d_user : int
        Dimension of user latent state.
    n_clusters : int
        Number of knowledge clusters for uncertainty tracking.
    n_heads : int
        Number of attention heads.
    d_k : int
        Key/query dimension per attention head.
    d_v : int
        Value dimension per attention head.
    alpha : float
        Exploration parameter (higher = more exploration).
    seed : int | None
        Random seed for clustering.
    """

    def __init__(
        self,
        d_knowledge: int,
        d_user: int = 32,
        n_clusters: int = 10,
        n_heads: int = 4,
        d_k: int = 64,
        d_v: int = 64,
        alpha: float = 1.0,
        seed: int | None = 42,
    ) -> None:
        super().__init__()
        self.d_knowledge = d_knowledge
        self.d_user = d_user
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.seed = seed

        # Learnable attention module
        self.attention = LearnableCrossAttention(
            d_knowledge=d_knowledge,
            d_user=d_user,
            d_k=d_k,
            d_v=d_v,
            n_heads=n_heads,
            n_clusters=n_clusters,
            alpha=alpha,
        )

        # Knowledge clustering (fit when knowledge pool is set)
        self._clusterer = MiniBatchKMeans(
            n_clusters=n_clusters, random_state=seed, n_init=3
        )
        self._knowledge_pool: NDArray | None = None
        self._knowledge_tensor: torch.Tensor | None = None
        self._cluster_assignments: NDArray | None = None

        # User states (keyed by user ID, NumPy-based Bayesian update)
        self._users: dict[str, UserState] = {}

    def set_knowledge_pool(self, embeddings: NDArray) -> None:
        """Set and cluster the knowledge pool.

        Parameters
        ----------
        embeddings : array of shape (n_items, d_knowledge)
            Embeddings of all items in the knowledge pool.
            NOT zero-padded — used directly.
        """
        self._knowledge_pool = embeddings
        n_items = embeddings.shape[0]

        # Cluster knowledge items
        effective_clusters = min(self.n_clusters, n_items)
        if effective_clusters < self.n_clusters:
            self._clusterer = MiniBatchKMeans(
                n_clusters=effective_clusters, random_state=self.seed, n_init=3
            )
        self._cluster_assignments = self._clusterer.fit_predict(embeddings)

        # Store as tensor for forward pass
        self._knowledge_tensor = torch.tensor(
            embeddings, dtype=torch.float32
        )

    def get_user(self, user_id: str) -> UserState:
        """Get or create user state."""
        if user_id not in self._users:
            self._users[user_id] = UserState(self.d_user)
        return self._users[user_id]

    def recommend(
        self,
        query_embedding: NDArray,
        user_id: str,
        return_details: bool = False,
    ) -> NDArray | dict:
        """Generate a synthesized recommendation for a user.

        Parameters
        ----------
        query_embedding : array of shape (d_knowledge,)
            Embedding of the user's current query/need.
        user_id : str
            Identifier for the user.
        return_details : bool
            If True, return a dict with synthesis vector, attention weights, etc.

        Returns
        -------
        z : array of shape (d_knowledge,) or dict
            Synthesized representation in knowledge embedding space.
        """
        if self._knowledge_pool is None:
            raise RuntimeError("Call set_knowledge_pool() first.")

        user = self.get_user(user_id)

        # Convert to tensors
        query_t = torch.tensor(
            query_embedding.ravel(), dtype=torch.float32
        )
        user_t = torch.tensor(user.u, dtype=torch.float32)
        keys = self._knowledge_tensor
        values = self._knowledge_tensor

        # Forward pass through learnable attention
        with torch.no_grad():
            z_t, weights_t = self.attention.forward(
                query_embedding=query_t,
                user_state=user_t,
                keys=keys,
                values=values,
                cluster_assignments=self._cluster_assignments,
                return_weights=True,
            )

        z = z_t.numpy()
        weights = weights_t.numpy()

        if return_details:
            entropy = self.attention.get_attention_entropy(weights_t)
            return {
                "synthesis": z,
                "attention_weights": weights,
                "attention_entropy": entropy,
                "user_uncertainty": user.uncertainty,
                "n_interactions": user.n_interactions,
            }
        return z

    def update(
        self,
        query_embedding: NDArray,
        user_id: str,
        reward: float,
        synthesis: NDArray | None = None,
    ) -> None:
        """Update user state and uncertainty trackers after reward feedback.

        Parameters
        ----------
        query_embedding : array of shape (d_knowledge,)
            The query embedding from this interaction.
        user_id : str
            User identifier.
        reward : float
            Reward signal (0 to 1).
        synthesis : array or None
            The synthesized representation (unused, for API compat).
        """
        user = self.get_user(user_id)

        # Update user state with the QUERY embedding (same logic as HARE)
        user.update(query_embedding, reward)

        # Update attention uncertainty trackers
        query_t = torch.tensor(
            query_embedding.ravel(), dtype=torch.float32
        )
        user_t = torch.tensor(user.u, dtype=torch.float32)

        with torch.no_grad():
            _, weights_t = self.attention.forward(
                query_embedding=query_t,
                user_state=user_t,
                keys=self._knowledge_tensor,
                values=self._knowledge_tensor,
                cluster_assignments=self._cluster_assignments,
                return_weights=True,
            )

        # Update each head's tracker for the dominant cluster
        mean_weights = weights_t.mean(dim=0).numpy()
        dominant_item = int(np.argmax(mean_weights))
        dominant_cluster = int(self._cluster_assignments[dominant_item])
        for h in range(self.attention.n_heads):
            self.attention.update_uncertainty(
                query_t, user_t, h, dominant_cluster
            )

    def forward_attention(
        self,
        query_embedding: torch.Tensor,
        user_state: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        cluster_assignments: NDArray,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Differentiable forward pass for training.

        Unlike recommend(), this does NOT detach gradients,
        allowing backprop through the attention projections.
        """
        return self.attention.forward(
            query_embedding=query_embedding,
            user_state=user_state,
            keys=keys,
            values=values,
            cluster_assignments=cluster_assignments,
            return_weights=return_weights,
        )
