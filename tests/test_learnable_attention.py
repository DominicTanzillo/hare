"""Tests for learnable cross-attention and LearnableHARE components."""

import numpy as np
import pytest
import torch

from hare.attention.learnable_cross_attention import LearnableCrossAttention
from hare.bandits.learnable_hare import LearnableHARE


# ── LearnableCrossAttention ────────────────────────────────────────────


class TestLearnableCrossAttention:
    @pytest.fixture
    def attn(self):
        return LearnableCrossAttention(
            d_knowledge=16, d_user=8, d_k=8, d_v=8,
            n_heads=2, n_clusters=3, alpha=1.0,
        )

    def test_output_shape_is_d_knowledge(self, attn):
        """z should have shape (d_knowledge,), NOT (d_knowledge + d_user,)."""
        query = torch.randn(16)
        user = torch.randn(8)
        keys = torch.randn(5, 16)
        clusters = np.array([0, 0, 1, 1, 2])

        z = attn(query, user, keys, keys, clusters)
        assert z.shape == (16,), f"Expected (16,), got {z.shape}"

    def test_returns_weights(self, attn):
        query = torch.randn(16)
        user = torch.randn(8)
        keys = torch.randn(5, 16)
        clusters = np.array([0, 0, 1, 1, 2])

        z, weights = attn(query, user, keys, keys, clusters, return_weights=True)
        assert weights.shape == (2, 5)  # n_heads x n_items

    def test_attention_weights_sum_to_one(self, attn):
        """Each head's attention weights must sum to 1."""
        query = torch.randn(16)
        user = torch.randn(8)
        keys = torch.randn(5, 16)
        clusters = np.array([0, 0, 1, 1, 2])

        _, weights = attn(query, user, keys, keys, clusters, return_weights=True)
        sums = weights.sum(dim=1)
        torch.testing.assert_close(sums, torch.ones(2), atol=1e-5, rtol=0)

    def test_attention_weights_nonnegative(self, attn):
        """Softmax output must be non-negative."""
        query = torch.randn(16)
        user = torch.randn(8)
        keys = torch.randn(5, 16)
        clusters = np.array([0, 0, 1, 1, 2])

        _, weights = attn(query, user, keys, keys, clusters, return_weights=True)
        assert (weights >= 0).all()

    def test_gradient_flow_W_Q_x(self, attn):
        """Gradients should flow through W_Q_x."""
        query = torch.randn(16)
        user = torch.randn(8)
        keys = torch.randn(5, 16)
        clusters = np.array([0, 0, 1, 1, 2])

        z = attn(query, user, keys, keys, clusters)
        loss = z.sum()
        loss.backward()

        assert attn.W_Q_x.grad is not None
        assert attn.W_Q_x.grad.abs().sum() > 0

    def test_gradient_flow_W_Q_u(self, attn):
        """Gradients should flow through W_Q_u (the personalization lever)."""
        query = torch.randn(16)
        user = torch.randn(8)
        keys = torch.randn(5, 16)
        clusters = np.array([0, 0, 1, 1, 2])

        z = attn(query, user, keys, keys, clusters)
        loss = z.sum()
        loss.backward()

        assert attn.W_Q_u.grad is not None
        assert attn.W_Q_u.grad.abs().sum() > 0

    def test_gradient_flow_all_params(self, attn):
        """All projection matrices should receive gradients."""
        query = torch.randn(16)
        user = torch.randn(8)
        keys = torch.randn(5, 16)
        clusters = np.array([0, 0, 1, 1, 2])

        z = attn(query, user, keys, keys, clusters)
        loss = z.sum()
        loss.backward()

        for name, param in attn.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"

    def test_user_sensitivity_after_gradient_step(self):
        """After discriminative training, different user states should produce
        different attention weights. This is the core personalization test.

        We train the model so user_a attends to item 0 and user_b attends to
        item 4, using the SAME query. This forces W_Q_u to learn to route
        different user states to different items.
        """
        torch.manual_seed(42)
        attn = LearnableCrossAttention(
            d_knowledge=16, d_user=8, d_k=8, d_v=8,
            n_heads=2, n_clusters=3, alpha=0.0,  # disable UCB for clean test
        )

        query = torch.randn(16)
        keys = torch.randn(5, 16)
        clusters = np.array([0, 0, 1, 1, 2])

        # Two very different user states (orthogonal directions)
        user_a = torch.zeros(8)
        user_a[0] = 1.0
        user_b = torch.zeros(8)
        user_b[4] = 1.0

        # Train: user_a -> item 0, user_b -> item 4
        target_a = torch.zeros(5)
        target_a[0] = 1.0
        target_b = torch.zeros(5)
        target_b[4] = 1.0

        optimizer = torch.optim.Adam(attn.parameters(), lr=0.05)
        for _ in range(50):
            _, w_a = attn(query, user_a, keys, keys, clusters, return_weights=True)
            _, w_b = attn(query, user_b, keys, keys, clusters, return_weights=True)
            loss_a = -torch.sum(target_a * torch.log(w_a.mean(dim=0) + 1e-12))
            loss_b = -torch.sum(target_b * torch.log(w_b.mean(dim=0) + 1e-12))
            loss = loss_a + loss_b
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # After training: attention should differ between users
        with torch.no_grad():
            _, w_a_after = attn(query, user_a, keys, keys, clusters, return_weights=True)
            _, w_b_after = attn(query, user_b, keys, keys, clusters, return_weights=True)

        mean_a = w_a_after.mean(dim=0)
        mean_b = w_b_after.mean(dim=0)

        # Users should produce different attention patterns
        diff = (mean_a - mean_b).abs().sum()
        assert diff > 0.01, f"Users should diverge after training, diff={diff}"

    def test_uncertainty_tracker_integration(self, attn):
        """UncertaintyTracker should work with detached queries."""
        query = torch.randn(16)
        user = torch.randn(8)
        keys = torch.randn(5, 16)
        clusters = np.array([0, 0, 1, 1, 2])

        # Get attention (triggers uncertainty computation)
        z, weights = attn(query, user, keys, keys, clusters, return_weights=True)
        assert z.shape == (16,)

        # Update uncertainty
        attn.update_uncertainty(query, user, head=0, cluster=0)

        # Get attention again — should still work
        z2, weights2 = attn(query, user, keys, keys, clusters, return_weights=True)
        assert z2.shape == (16,)

    def test_mean_attention_differentiable(self, attn):
        """get_mean_attention should be differentiable."""
        query = torch.randn(16, requires_grad=True)
        user = torch.randn(8)
        keys = torch.randn(5, 16)
        clusters = np.array([0, 0, 1, 1, 2])

        mean_attn = attn.get_mean_attention(query, user, keys, keys, clusters)
        assert mean_attn.shape == (5,)
        loss = mean_attn.sum()
        loss.backward()
        assert query.grad is not None

    def test_attention_entropy(self, attn):
        """Entropy computation should return a finite positive value."""
        uniform = torch.ones(2, 5) / 5
        peaked = torch.tensor([[0.96, 0.01, 0.01, 0.01, 0.01],
                                [0.96, 0.01, 0.01, 0.01, 0.01]])
        e_uniform = attn.get_attention_entropy(uniform)
        e_peaked = attn.get_attention_entropy(peaked)
        assert e_uniform > e_peaked


# ── LearnableHARE ──────────────────────────────────────────────────────


class TestLearnableHARE:
    @pytest.fixture
    def learnable_hare(self):
        rng = np.random.default_rng(42)
        knowledge = rng.normal(size=(10, 16))
        h = LearnableHARE(
            d_knowledge=16,
            d_user=8,
            n_clusters=3,
            n_heads=2,
            d_k=8,
            d_v=8,
            alpha=1.0,
            seed=42,
        )
        h.set_knowledge_pool(knowledge)
        return h

    def test_recommend_output_shape(self, learnable_hare):
        """Output should be (d_knowledge,), NOT (d_knowledge + d_user,)."""
        query = np.random.randn(16)
        z = learnable_hare.recommend(query, user_id="test")
        assert isinstance(z, np.ndarray)
        assert z.shape == (16,), f"Expected (16,), got {z.shape}"

    def test_recommend_returns_details(self, learnable_hare):
        query = np.random.randn(16)
        result = learnable_hare.recommend(query, user_id="test", return_details=True)
        assert "synthesis" in result
        assert "attention_weights" in result
        assert "attention_entropy" in result
        assert "user_uncertainty" in result
        assert result["synthesis"].shape == (16,)

    def test_different_users_different_outputs(self, learnable_hare):
        """Core test: after interaction history, users should diverge."""
        query = np.random.randn(16)

        # User A: no history
        z_a = learnable_hare.recommend(query, user_id="user_a")

        # User B: has interaction history
        rng = np.random.default_rng(99)
        for _ in range(10):
            q = rng.normal(size=16)
            learnable_hare.recommend(q, user_id="user_b")
            learnable_hare.update(q, "user_b", reward=0.9)

        z_b = learnable_hare.recommend(query, user_id="user_b")

        # Outputs should differ
        assert not np.allclose(z_a, z_b, atol=1e-6)

    def test_update_reduces_uncertainty(self, learnable_hare):
        query = np.random.randn(16)
        user = learnable_hare.get_user("test")
        unc_before = user.uncertainty

        result = learnable_hare.recommend(query, "test", return_details=True)
        learnable_hare.update(query, "test", reward=1.0, synthesis=result["synthesis"])

        unc_after = user.uncertainty
        assert unc_after < unc_before

    def test_forward_attention_differentiable(self, learnable_hare):
        """forward_attention should allow gradient computation."""
        query_t = torch.randn(16)
        user_t = torch.randn(8)
        keys_t = learnable_hare._knowledge_tensor

        z = learnable_hare.forward_attention(
            query_t, user_t, keys_t, keys_t,
            learnable_hare._cluster_assignments,
        )
        loss = z.sum()
        loss.backward()

        # Should have gradients on attention parameters
        assert learnable_hare.attention.W_Q_x.grad is not None

    def test_no_zero_padding_in_knowledge(self, learnable_hare):
        """Knowledge tensor should be (n_items, d_knowledge), NOT zero-padded."""
        assert learnable_hare._knowledge_tensor.shape == (10, 16)

    def test_checkpoint_save_and_load(self, learnable_hare, tmp_path):
        """Should be able to save and load attention weights."""
        save_path = tmp_path / "test_attention.pt"

        # Save
        torch.save({
            "attention_state_dict": learnable_hare.attention.state_dict(),
            "config": {
                "d_knowledge": 16, "d_user": 8,
                "n_heads": 2, "d_k": 8, "d_v": 8,
            },
        }, save_path)

        # Load into a fresh instance
        rng = np.random.default_rng(42)
        knowledge = rng.normal(size=(10, 16))
        h2 = LearnableHARE(
            d_knowledge=16, d_user=8, n_clusters=3,
            n_heads=2, d_k=8, d_v=8,
        )
        state = torch.load(save_path, weights_only=True)
        h2.attention.load_state_dict(state["attention_state_dict"])
        h2.set_knowledge_pool(knowledge)

        # Should produce identical output
        query = np.random.randn(16)
        z1 = learnable_hare.recommend(query, user_id="ckpt_test")
        z2 = h2.recommend(query, user_id="ckpt_test")
        np.testing.assert_allclose(z1, z2, atol=1e-6)
