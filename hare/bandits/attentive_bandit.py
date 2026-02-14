"""HARE: Hybrid Attention-Reinforced Exploration.

The core algorithm that combines:
1. User latent state modeling (Bayesian update of user preferences)
2. Uncertainty-augmented cross-attention over a knowledge pool
3. Generative synthesis from the attended representation

Two users with the same query produce different outputs because their
latent states u_t differ → different attention patterns → different synthesis.
Specificity emerges from uncertainty reduction over interactions.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import MiniBatchKMeans

from hare.attention.cross_attention import MultiHeadCrossAttention


class UserState:
    """Bayesian model of a single user's latent preferences.

    Maintains a mean vector u_t and covariance Σ_u(t) that evolve
    as the system observes rewards from this user.

    Parameters
    ----------
    d : int
        Dimension of user latent state.
    prior_precision : float
        Precision (1/variance) of the Gaussian prior on u.
    """

    def __init__(self, d: int, prior_precision: float = 1.0) -> None:
        self.d = d
        self.u = np.zeros(d)                             # posterior mean
        self.sigma = np.eye(d) / prior_precision         # posterior covariance
        self._sigma_inv = np.eye(d) * prior_precision    # precision matrix
        self.n_interactions = 0

    def update(self, feature: NDArray, reward: float, noise_var: float = 1.0) -> None:
        """Bayesian linear regression update.

        After observing reward r for feature vector x, update posterior:
            Σ^{-1}_{t+1} = Σ^{-1}_t + x x^T / σ²
            u_{t+1} = Σ_{t+1} (Σ^{-1}_t u_t + r·x / σ²)

        Parameters
        ----------
        feature : array of shape (d,)
            Feature vector from this interaction (e.g., the attended output z).
        reward : float
            Observed reward signal.
        noise_var : float
            Observation noise variance σ².
        """
        x = feature.ravel()[:self.d]  # Truncate if needed
        if len(x) < self.d:
            x = np.pad(x, (0, self.d - len(x)))

        # Update precision matrix
        self._sigma_inv += np.outer(x, x) / noise_var

        # Update mean via precision-weighted form
        self.u = np.linalg.solve(
            self._sigma_inv,
            self._sigma_inv @ self.u + reward * x / noise_var,
        )

        # Update covariance (inverse of precision)
        self.sigma = np.linalg.inv(self._sigma_inv)
        self.n_interactions += 1

    @property
    def uncertainty(self) -> float:
        """Scalar summary of remaining uncertainty (trace of covariance)."""
        return float(np.trace(self.sigma))

    def get_exploration_bonus(self, feature: NDArray) -> float:
        """UCB-style bonus: √(x^T Σ x) — how uncertain we are in this direction."""
        x = feature.ravel()[:self.d]
        if len(x) < self.d:
            x = np.pad(x, (0, self.d - len(x)))
        return float(np.sqrt(np.abs(x @ self.sigma @ x)))


class HARE:
    """Hybrid Attention-Reinforced Exploration for Generative Recommendation.

    The full pipeline:
    1. Encode query and user state into a joint representation
    2. Attend over knowledge pool with uncertainty bonuses
    3. Synthesize output from the attended representation
    4. Update user state and uncertainty from reward feedback

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
        Random seed.
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
        self.d_knowledge = d_knowledge
        self.d_user = d_user
        self.d_input = d_knowledge + d_user  # joint query dimension
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.seed = seed

        # Attention module
        self.attention = MultiHeadCrossAttention(
            d_input=self.d_input,
            d_k=d_k,
            d_v=d_v,
            n_heads=n_heads,
            n_clusters=n_clusters,
            alpha=alpha,
            seed=seed,
        )

        # Knowledge clustering (fit when knowledge pool is set)
        self._clusterer = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, n_init=3)
        self._knowledge_pool: NDArray | None = None
        self._cluster_assignments: NDArray | None = None
        self._knowledge_padded: NDArray | None = None

        # User states (keyed by user ID)
        self._users: dict[str, UserState] = {}

    def set_knowledge_pool(self, embeddings: NDArray) -> None:
        """Set and cluster the knowledge pool.

        Parameters
        ----------
        embeddings : array of shape (n_items, d_knowledge)
            Embeddings of all items in the knowledge pool.
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

        # Pad knowledge embeddings to joint dimension (d_knowledge + d_user)
        padding = np.zeros((n_items, self.d_user))
        self._knowledge_padded = np.hstack([embeddings, padding])

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
            If True, return a dict with synthesis vector, attention weights, entropy.

        Returns
        -------
        z : array of shape (d_input,) or dict
            Synthesized representation (the "ideal item" in embedding space).
        """
        if self._knowledge_pool is None:
            raise RuntimeError("Call set_knowledge_pool() first.")

        user = self.get_user(user_id)

        # Build joint query: [x_query ⊕ u_t]
        joint_query = np.concatenate([query_embedding.ravel(), user.u])

        # Uncertainty-augmented cross-attention
        z, weights = self.attention.forward(
            query=joint_query,
            keys=self._knowledge_padded,
            values=self._knowledge_padded,
            cluster_assignments=self._cluster_assignments,
            return_weights=True,
        )

        if return_details:
            entropy = self.attention.get_attention_entropy(weights)
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
        synthesis : array of shape (d_input,) or None
            The synthesized representation from this interaction.
            If None, re-computes it.
        """
        user = self.get_user(user_id)

        if synthesis is None:
            synthesis = self.recommend(query_embedding, user_id)
            if isinstance(synthesis, dict):
                synthesis = synthesis["synthesis"]

        # Update user state with the synthesized feature as the observation
        user.update(synthesis, reward)

        # Update attention uncertainty trackers
        joint_query = np.concatenate([query_embedding.ravel(), user.u])
        # Find which cluster got highest attention (dominant cluster)
        _, weights = self.attention.forward(
            query=joint_query,
            keys=self._knowledge_padded,
            values=self._knowledge_padded,
            cluster_assignments=self._cluster_assignments,
            return_weights=True,
        )

        # Update each head's tracker for the cluster that got most attention
        mean_weights = np.mean(weights, axis=0)
        dominant_item = int(np.argmax(mean_weights))
        dominant_cluster = int(self._cluster_assignments[dominant_item])
        for h in range(self.attention.n_heads):
            self.attention.update_uncertainty(joint_query, h, dominant_cluster)

    def simulate_session(
        self,
        query_embeddings: NDArray,
        reward_fn: callable,
        user_id: str = "user_0",
    ) -> dict:
        """Run a full session of interactions and track metrics.

        Parameters
        ----------
        query_embeddings : array of shape (T, d_knowledge)
            Sequence of query embeddings.
        reward_fn : callable
            Function(synthesis, query, user_state) -> float reward.
        user_id : str
            User identifier.

        Returns
        -------
        dict with keys: rewards, entropies, uncertainties, syntheses
        """
        T = query_embeddings.shape[0]
        rewards = np.empty(T)
        entropies = np.empty(T)
        uncertainties = np.empty(T)
        syntheses = []

        for t in range(T):
            result = self.recommend(query_embeddings[t], user_id, return_details=True)

            z = result["synthesis"]
            user = self.get_user(user_id)
            reward = reward_fn(z, query_embeddings[t], user)

            self.update(query_embeddings[t], user_id, reward, synthesis=z)

            rewards[t] = reward
            entropies[t] = result["attention_entropy"]
            uncertainties[t] = result["user_uncertainty"]
            syntheses.append(z)

        return {
            "rewards": rewards,
            "entropies": entropies,
            "uncertainties": uncertainties,
            "syntheses": np.array(syntheses),
            "cumulative_reward": np.cumsum(rewards),
        }
