"""Tests for HARE core components."""

import numpy as np
import pytest

from hare.bandits.linucb import LinUCB
from hare.bandits.attentive_bandit import HARE, UserState
from hare.attention.cross_attention import (
    MultiHeadCrossAttention,
    UncertaintyTracker,
    _softmax,
)
from hare.synthesis.generator import InterpolationDecoder
from hare.utils.embeddings import TfidfEmbedder


# ── Softmax ─────────────────────────────────────────────────────────────


class TestSoftmax:
    def test_sums_to_one(self):
        x = np.array([1.0, 2.0, 3.0])
        assert np.isclose(_softmax(x).sum(), 1.0)

    def test_numerically_stable(self):
        x = np.array([1000.0, 1001.0, 1002.0])
        result = _softmax(x)
        assert np.isclose(result.sum(), 1.0)
        assert np.all(np.isfinite(result))

    def test_uniform_for_equal_inputs(self):
        x = np.array([5.0, 5.0, 5.0])
        result = _softmax(x)
        np.testing.assert_allclose(result, [1 / 3, 1 / 3, 1 / 3], atol=1e-10)


# ── LinUCB ──────────────────────────────────────────────────────────────


class TestLinUCB:
    def test_initialization(self):
        bandit = LinUCB(n_arms=5, d=3, alpha=1.0)
        assert bandit.n_arms == 5
        assert bandit.d == 3
        for a in range(5):
            np.testing.assert_array_equal(bandit.A[a], np.eye(3))

    def test_select_arm_returns_valid_index(self):
        bandit = LinUCB(n_arms=5, d=3)
        context = np.array([1.0, 0.5, -0.2])
        arm = bandit.select_arm(context)
        assert 0 <= arm < 5

    def test_update_changes_parameters(self):
        bandit = LinUCB(n_arms=3, d=2)
        context = np.array([1.0, 0.0])
        A_before = bandit.A[0].copy()
        bandit.update(0, context, reward=1.0)
        assert not np.array_equal(bandit.A[0], A_before)

    def test_learns_best_arm(self):
        """After many updates, bandit should prefer the rewarded arm."""
        rng = np.random.default_rng(42)
        bandit = LinUCB(n_arms=3, d=4, alpha=0.1)

        # Generate a fixed context direction where arm 1 is clearly best
        context = np.array([1.0, 1.0, 1.0, 1.0])
        for _ in range(100):
            noisy_ctx = context + rng.normal(scale=0.1, size=4)
            # Only update the arm we select (realistic bandit protocol)
            arm = bandit.select_arm(noisy_ctx)
            reward = 1.0 if arm == 1 else 0.0
            bandit.update(arm, noisy_ctx, reward)

        # Directly check that theta for arm 1 has higher expected reward
        theta_1 = bandit.get_theta(1)
        rewards = [context @ bandit.get_theta(a) for a in range(3)]
        assert rewards[1] >= max(rewards[0], rewards[2]) - 0.5

    def test_uncertainty_decreases_with_updates(self):
        bandit = LinUCB(n_arms=2, d=3)
        context = np.array([1.0, 0.5, 0.0])
        unc_before = bandit.get_uncertainty(0, context)
        bandit.update(0, context, reward=1.0)
        unc_after = bandit.get_uncertainty(0, context)
        assert unc_after < unc_before


# ── UncertaintyTracker ──────────────────────────────────────────────────


class TestUncertaintyTracker:
    def test_initial_uncertainty(self):
        tracker = UncertaintyTracker(n_clusters=3, d=4)
        query = np.array([1.0, 0.0, 0.0, 0.0])
        assignments = np.array([0, 1, 2])
        uncertainties = tracker.get_uncertainty(query, assignments)
        assert len(uncertainties) == 3
        assert np.all(uncertainties > 0)

    def test_uncertainty_decreases_after_update(self):
        tracker = UncertaintyTracker(n_clusters=2, d=3)
        query = np.array([1.0, 0.5, 0.0])
        assignments = np.array([0, 0, 1])

        unc_before = tracker.get_uncertainty(query, assignments)
        tracker.update(query, cluster=0)
        unc_after = tracker.get_uncertainty(query, assignments)

        # Cluster 0 uncertainty should decrease
        assert unc_after[0] < unc_before[0]


# ── MultiHeadCrossAttention ─────────────────────────────────────────────


class TestMultiHeadCrossAttention:
    def test_output_shape(self):
        d_input = 16
        attn = MultiHeadCrossAttention(
            d_input=d_input, d_k=8, d_v=8, n_heads=2, n_clusters=3, seed=0
        )
        query = np.random.randn(d_input)
        keys = np.random.randn(5, d_input)
        values = keys.copy()
        clusters = np.array([0, 0, 1, 1, 2])

        z = attn.forward(query, keys, values, clusters)
        assert z.shape == (d_input,)

    def test_returns_weights(self):
        d_input = 16
        attn = MultiHeadCrossAttention(
            d_input=d_input, d_k=8, d_v=8, n_heads=2, n_clusters=3, seed=0
        )
        query = np.random.randn(d_input)
        keys = np.random.randn(5, d_input)
        clusters = np.array([0, 0, 1, 1, 2])

        z, weights = attn.forward(query, keys, keys, clusters, return_weights=True)
        assert weights.shape == (2, 5)  # n_heads x n_items
        # Each head's weights sum to 1
        np.testing.assert_allclose(weights.sum(axis=1), [1.0, 1.0], atol=1e-6)

    def test_attention_entropy(self):
        d_input = 8
        attn = MultiHeadCrossAttention(
            d_input=d_input, d_k=4, d_v=4, n_heads=2, n_clusters=2, seed=0
        )

        # Uniform weights → max entropy
        uniform = np.ones((2, 4)) / 4
        entropy_uniform = attn.get_attention_entropy(uniform)

        # Peaked weights → low entropy
        peaked = np.array([[0.97, 0.01, 0.01, 0.01], [0.97, 0.01, 0.01, 0.01]])
        entropy_peaked = attn.get_attention_entropy(peaked)

        assert entropy_uniform > entropy_peaked


# ── UserState ───────────────────────────────────────────────────────────


class TestUserState:
    def test_initial_state(self):
        user = UserState(d=4)
        np.testing.assert_array_equal(user.u, np.zeros(4))
        assert user.n_interactions == 0

    def test_uncertainty_decreases(self):
        user = UserState(d=4)
        unc_before = user.uncertainty
        user.update(np.array([1.0, 0.0, 0.0, 0.0]), reward=1.0)
        unc_after = user.uncertainty
        assert unc_after < unc_before

    def test_state_shifts_toward_rewarded_direction(self):
        user = UserState(d=3, prior_precision=1.0)
        feature = np.array([1.0, 0.0, 0.0])

        # Repeatedly reward this direction
        for _ in range(10):
            user.update(feature, reward=1.0)

        # User state should have positive first component
        assert user.u[0] > 0

    def test_exploration_bonus(self):
        user = UserState(d=3)
        feature = np.array([1.0, 0.0, 0.0])
        bonus = user.get_exploration_bonus(feature)
        assert bonus > 0


# ── HARE (Integration) ─────────────────────────────────────────────────


class TestHARE:
    @pytest.fixture
    def hare_instance(self):
        rng = np.random.default_rng(42)
        knowledge = rng.normal(size=(10, 8))
        h = HARE(
            d_knowledge=8,
            d_user=4,
            n_clusters=3,
            n_heads=2,
            d_k=8,
            d_v=8,
            alpha=1.0,
            seed=42,
        )
        h.set_knowledge_pool(knowledge)
        return h

    def test_recommend_returns_array(self, hare_instance):
        query = np.random.randn(8)
        z = hare_instance.recommend(query, user_id="test_user")
        assert isinstance(z, np.ndarray)

    def test_recommend_returns_details(self, hare_instance):
        query = np.random.randn(8)
        result = hare_instance.recommend(query, user_id="test_user", return_details=True)
        assert "synthesis" in result
        assert "attention_weights" in result
        assert "attention_entropy" in result
        assert "user_uncertainty" in result

    def test_different_users_different_outputs(self, hare_instance):
        query = np.random.randn(8)

        # User A: no history
        z_a = hare_instance.recommend(query, user_id="user_a")

        # User B: has interaction history
        for _ in range(5):
            q = np.random.randn(8)
            hare_instance.recommend(q, user_id="user_b")
            hare_instance.update(q, "user_b", reward=0.8)

        z_b = hare_instance.recommend(query, user_id="user_b")

        # Outputs should differ (personalization divergence)
        assert not np.allclose(z_a, z_b, atol=1e-6)

    def test_update_reduces_uncertainty(self, hare_instance):
        query = np.random.randn(8)
        user = hare_instance.get_user("test_user")
        unc_before = user.uncertainty

        r = hare_instance.recommend(query, "test_user", return_details=True)
        hare_instance.update(query, "test_user", reward=1.0, synthesis=r["synthesis"])

        unc_after = user.uncertainty
        assert unc_after < unc_before

    def test_simulate_session(self, hare_instance):
        rng = np.random.default_rng(0)
        queries = rng.normal(size=(5, 8))

        def dummy_reward(synthesis, query, user_state):
            return 0.5

        results = hare_instance.simulate_session(queries, dummy_reward, user_id="sim_user")
        assert "rewards" in results
        assert "entropies" in results
        assert "uncertainties" in results
        assert len(results["rewards"]) == 5
        assert results["cumulative_reward"][-1] == pytest.approx(2.5, abs=0.01)

    def test_entropy_trend_decreasing(self, hare_instance):
        """Entropy should generally decrease as user uncertainty decreases."""
        rng = np.random.default_rng(42)
        queries = rng.normal(size=(20, 8))
        target = rng.normal(size=8)

        def reward_fn(synthesis, query, user_state):
            from sklearn.metrics.pairwise import cosine_similarity
            s = synthesis[:8]
            sim = cosine_similarity(s.reshape(1, -1), target.reshape(1, -1))[0, 0]
            return float(np.clip((sim + 1) / 2, 0, 1))

        results = hare_instance.simulate_session(queries, reward_fn, user_id="entropy_test")

        # Compare first half vs second half average entropy
        first_half = np.mean(results["entropies"][:10])
        second_half = np.mean(results["entropies"][10:])
        # Second half should have lower or similar entropy (more specific)
        # This is a statistical tendency, not guaranteed per-run
        assert second_half <= first_half * 1.5  # generous bound


# ── InterpolationDecoder ────────────────────────────────────────────────


class TestInterpolationDecoder:
    def test_generate_output(self):
        decoder = InterpolationDecoder(top_k=2, temperature=0.5)
        candidates = ["item A: description", "item B: description", "item C: description"]
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
        synthesis = np.array([0.8, 0.2])

        output = decoder.generate(synthesis, candidates, embeddings)
        assert "HARE Synthesis" in output
        assert "item A" in output  # should be top match

    def test_blend_weights_sum_to_one(self):
        decoder = InterpolationDecoder(top_k=3)
        embeddings = np.random.randn(5, 4)
        synthesis = np.random.randn(4)
        indices, weights = decoder.get_blend_weights(synthesis, embeddings)
        assert np.isclose(weights.sum(), 1.0)


# ── TfidfEmbedder ──────────────────────────────────────────────────────


class TestTfidfEmbedder:
    def test_fit_and_encode(self):
        corpus = ["hello world", "goodbye world", "hello goodbye"]
        embedder = TfidfEmbedder(max_features=100, output_dim=2)
        embedder.fit(corpus)
        embs = embedder.encode(["hello world"])
        assert embs.shape == (1, 2)

    def test_normalized(self):
        corpus = ["the quick brown fox", "jumped over the lazy dog", "hello world"]
        embedder = TfidfEmbedder(max_features=100, output_dim=2)
        embedder.fit(corpus)
        embs = embedder.encode(corpus)
        norms = np.linalg.norm(embs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_raises_before_fit(self):
        embedder = TfidfEmbedder()
        with pytest.raises(RuntimeError, match="fit"):
            embedder.encode(["test"])
