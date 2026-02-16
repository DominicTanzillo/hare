"""Tests for LongLaMP and PENS data loading and baseline compatibility."""

import os

import pytest


# =============================================================================
# LongLaMP tests (require HuggingFace datasets download)
# =============================================================================

_requires_longlamp = pytest.mark.skipif(
    os.environ.get("SKIP_LONGLAMP_TESTS", "1") == "1",
    reason="Set SKIP_LONGLAMP_TESTS=0 to run LongLaMP tests (requires HF download)",
)


class TestLongLaMPTaskConfigs:
    """Tests for LongLaMP task configs (no network needed)."""

    def test_abstract_config_exists(self):
        from hare.evaluation.baselines import get_task_config

        cfg = get_task_config("longlamp_abstract")
        assert cfg.task_name == "LongLaMP-abstract"
        assert cfg.content_label == "Title"
        assert cfg.target_label == "Abstract"
        assert cfg.profile_text_key == "text"

    def test_review_config_exists(self):
        from hare.evaluation.baselines import get_task_config

        cfg = get_task_config("longlamp_review")
        assert cfg.task_name == "LongLaMP-review"
        assert cfg.content_label == "Product"
        assert cfg.target_label == "Review"
        assert cfg.profile_text_key == "text"

    def test_pens_config_exists(self):
        from hare.evaluation.baselines import get_task_config

        cfg = get_task_config("pens")
        assert cfg.task_name == "PENS"
        assert cfg.content_label == "Article"
        assert cfg.target_label == "Headline"
        assert cfg.profile_text_key == "text"
        assert cfg.profile_target_key == "title"

    def test_strip_prefix_abstract(self):
        from hare.evaluation.baselines import _strip_prefix, get_task_config

        cfg = get_task_config("longlamp_abstract")
        text = 'Generate an abstract for the title "Neural Networks" using the following items: 1. Deep learning'
        stripped = _strip_prefix(text, cfg)
        assert "1. Deep learning" in stripped

    def test_strip_prefix_review(self):
        from hare.evaluation.baselines import _strip_prefix, get_task_config

        cfg = get_task_config("longlamp_review")
        text = 'Generate a 4.0-star review for "Sony WH-1000XM5"'
        stripped = _strip_prefix(text, cfg)
        assert "Sony" in stripped

    def test_strip_prefix_pens(self):
        from hare.evaluation.baselines import _strip_prefix, get_task_config

        cfg = get_task_config("pens")
        text = "Generate a headline for the following article: The market rose today..."
        stripped = _strip_prefix(text, cfg)
        assert stripped.strip().startswith("The market")


class TestLongLaMPNormalization:
    """Tests for profile item normalization (no network needed)."""

    def test_abstract_normalizer(self):
        from hare.evaluation.longlamp import _normalize_abstract_profile

        item = {
            "abstract": "We study neural networks...",
            "title": "Deep Learning Survey",
            "year": 2023,
            "id": "abc123",
        }
        normalized = _normalize_abstract_profile(item)
        assert normalized["text"] == "We study neural networks..."
        assert normalized["title"] == "Deep Learning Survey"
        assert normalized["abstract"] == "We study neural networks..."

    def test_review_normalizer(self):
        from hare.evaluation.longlamp import _normalize_review_profile

        item = {
            "reviewText": "Great product, highly recommend!",
            "summary": "Excellent",
            "description": "Wireless headphones",
            "overall": "5.0",
        }
        normalized = _normalize_review_profile(item)
        assert normalized["text"] == "Great product, highly recommend!"
        assert normalized["title"] == "Excellent"
        assert normalized["rating"] == "5.0"


@_requires_longlamp
class TestLongLaMPDataLoader:
    """Tests that require HuggingFace download."""

    def test_load_abstract(self):
        from hare.evaluation.longlamp import load_longlamp

        data = load_longlamp("abstract", split="val", max_samples=5)
        assert len(data) == 5
        assert data.task == "LongLaMP-abstract"

    def test_abstract_sample_structure(self):
        from hare.evaluation.longlamp import load_longlamp

        data = load_longlamp("abstract", split="val", max_samples=3)
        sample = data.samples[0]
        assert sample.input_text
        assert sample.target
        assert len(sample.profile) > 0
        # Profile items should have normalized keys
        item = sample.profile[0]
        assert "text" in item
        assert "title" in item

    def test_load_review(self):
        from hare.evaluation.longlamp import load_longlamp

        data = load_longlamp("review", split="val", max_samples=5)
        assert len(data) == 5
        assert data.task == "LongLaMP-review"

    def test_baselines_compatible(self):
        """Verify non-neural baselines work with LongLaMP format."""
        from hare.evaluation.baselines import get_all_baselines
        from hare.evaluation.longlamp import load_longlamp

        data = load_longlamp("abstract", split="val", max_samples=3)
        baselines = get_all_baselines(
            include_neural=False, task="longlamp_abstract"
        )

        for baseline in baselines:
            sample = data.samples[0]
            pred = baseline.predict(sample.input_text, sample.profile)
            assert isinstance(pred, str), f"{baseline.name} should return str"
            assert len(pred) > 0, f"{baseline.name} returned empty string"


# =============================================================================
# PENS tests (require Azure download)
# =============================================================================

_requires_pens = pytest.mark.skipif(
    os.environ.get("SKIP_PENS_TESTS", "1") == "1",
    reason="Set SKIP_PENS_TESTS=0 to run PENS tests (requires Azure download)",
)


class TestPENSHelpers:
    """Tests for PENS helper functions (no network needed)."""

    def test_build_profile_from_clicks(self):
        from hare.evaluation.pens import _build_profile_from_clicks

        news_corpus = {
            "N1": {
                "news_id": "N1",
                "headline": "Test headline",
                "body": "Test body text",
                "category": "sports",
                "topic": "",
            },
            "N2": {
                "news_id": "N2",
                "headline": "Another headline",
                "body": "Another body",
                "category": "tech",
                "topic": "",
            },
        }

        profile = _build_profile_from_clicks(["N1", "N2", "N999"], news_corpus)
        assert len(profile) == 2  # N999 not in corpus
        assert profile[0]["text"] == "Test body text"
        assert profile[0]["title"] == "Test headline"
        assert profile[1]["category"] == "tech"

    def test_build_profile_max_limit(self):
        from hare.evaluation.pens import _build_profile_from_clicks

        news_corpus = {f"N{i}": {
            "news_id": f"N{i}",
            "headline": f"H{i}",
            "body": f"B{i}",
            "category": "",
            "topic": "",
        } for i in range(100)}

        ids = [f"N{i}" for i in range(100)]
        profile = _build_profile_from_clicks(ids, news_corpus, max_profile=10)
        assert len(profile) == 10


@_requires_pens
class TestPENSDataLoader:
    """Tests that require PENS download from Azure."""

    def test_load_val(self):
        from hare.evaluation.pens import load_pens

        data = load_pens(split="val", max_samples=10)
        assert len(data) <= 10
        assert data.task == "PENS"

    def test_sample_structure(self):
        from hare.evaluation.pens import load_pens

        data = load_pens(split="val", max_samples=5)
        if len(data) > 0:
            sample = data.samples[0]
            assert sample.input_text.startswith("Generate a headline")
            assert len(sample.target) > 0
            assert len(sample.profile) >= 2
            # Profile should have LaMP-compatible keys
            item = sample.profile[0]
            assert "text" in item
            assert "title" in item

    def test_baselines_compatible(self):
        from hare.evaluation.baselines import get_all_baselines
        from hare.evaluation.pens import load_pens

        data = load_pens(split="val", max_samples=3)
        if len(data) == 0:
            pytest.skip("No PENS samples loaded")

        baselines = get_all_baselines(include_neural=False, task="pens")
        for baseline in baselines:
            sample = data.samples[0]
            pred = baseline.predict(sample.input_text, sample.profile)
            assert isinstance(pred, str), f"{baseline.name} should return str"
