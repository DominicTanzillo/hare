"""LinUCB contextual bandit baseline (Li et al., 2010).

Disjoint linear model: each arm maintains its own ridge regression.
Used as the primary comparison baseline for HARE.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class LinUCB:
    """Disjoint LinUCB with per-arm ridge regression and UCB exploration.

    Parameters
    ----------
    n_arms : int
        Number of arms (items) to choose from.
    d : int
        Dimension of context vectors.
    alpha : float
        Exploration parameter controlling UCB width. Higher = more exploration.
    """

    def __init__(self, n_arms: int, d: int, alpha: float = 1.0) -> None:
        self.n_arms = n_arms
        self.d = d
        self.alpha = alpha

        # Per-arm design matrix A_a = D_a^T D_a + I (d x d)
        self.A = np.array([np.eye(d) for _ in range(n_arms)])
        # Per-arm reward-weighted feature vector b_a (d,)
        self.b = np.zeros((n_arms, d))
        # Cache inverses for efficiency
        self._A_inv = np.array([np.eye(d) for _ in range(n_arms)])

    def select_arm(self, context: NDArray[np.floating]) -> int:
        """Select arm with highest UCB score.

        Parameters
        ----------
        context : array of shape (d,)
            Context vector for this round.

        Returns
        -------
        int
            Index of selected arm.
        """
        x = context.ravel()
        ucb_scores = np.empty(self.n_arms)

        for a in range(self.n_arms):
            A_inv = self._A_inv[a]
            theta = A_inv @ self.b[a]
            # UCB = x^T theta + alpha * sqrt(x^T A^{-1} x)
            exploitation = x @ theta
            exploration = self.alpha * np.sqrt(x @ A_inv @ x)
            ucb_scores[a] = exploitation + exploration

        return int(np.argmax(ucb_scores))

    def select_arm_batch(self, context: NDArray[np.floating]) -> tuple[int, NDArray]:
        """Select arm and return all UCB scores.

        Returns
        -------
        tuple of (selected_arm, ucb_scores)
        """
        x = context.ravel()
        ucb_scores = np.empty(self.n_arms)

        for a in range(self.n_arms):
            A_inv = self._A_inv[a]
            theta = A_inv @ self.b[a]
            exploitation = x @ theta
            exploration = self.alpha * np.sqrt(x @ A_inv @ x)
            ucb_scores[a] = exploitation + exploration

        return int(np.argmax(ucb_scores)), ucb_scores

    def update(self, arm: int, context: NDArray[np.floating], reward: float) -> None:
        """Update arm's model after observing reward.

        Parameters
        ----------
        arm : int
            The arm that was played.
        context : array of shape (d,)
            Context vector for this round.
        reward : float
            Observed reward.
        """
        x = context.ravel()
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x
        # Update cached inverse via Sherman-Morrison
        A_inv = self._A_inv[arm]
        u = A_inv @ x
        self._A_inv[arm] = A_inv - np.outer(u, u) / (1.0 + x @ u)

    def get_theta(self, arm: int) -> NDArray[np.floating]:
        """Get estimated parameter vector for an arm."""
        return self._A_inv[arm] @ self.b[arm]

    def get_uncertainty(self, arm: int, context: NDArray[np.floating]) -> float:
        """Get UCB width (exploration bonus) for an arm given context."""
        x = context.ravel()
        return float(np.sqrt(x @ self._A_inv[arm] @ x))
