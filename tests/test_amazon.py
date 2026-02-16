"""Tests for Amazon Reviews data loading and baseline compatibility."""

import json
import os

import numpy as np
import pytest

# Mark for tests requiring network access to HuggingFace
_requires_download = pytest.mark.skipif(
    os.environ.get("SKIP_AMAZON_TESTS", "1") == "1",
    reason="Set SKIP_AMAZON_TESTS=0 to run Amazon data tests (requires HF download)",
)


@_requires_download
class TestAmazonDataLoader:
    """Tests that require the actual Amazon Reviews dataset (network)."""

    def test_load_digital_music(self):
        from hare.evaluation.amazon import load_amazon_reviews

        data = load_amazon_reviews(
            category="Digital_Music",
            min_reviews_per_user=5,
            max_samples=10,
        )
        assert len(data) == 10
        assert data.category == "Digital_Music"

    def test_sample_structure(self):
        from hare.evaluation.amazon import load_amazon_reviews

        data = load_amazon_reviews(
            category="Digital_Music",
            min_reviews_per_user=5,
            max_samples=5,
        )
        sample = data[0]
        assert sample.id
        assert sample.user_id
        assert sample.input_text.startswith("Write a review")
        assert len(sample.target) > 0
        assert len(sample.profile) >= 4  # at least min_reviews - 1

    def test_profile_has_required_fields(self):
        from hare.evaluation.amazon import load_amazon_reviews

        data = load_amazon_reviews(
            category="Digital_Music",
            min_reviews_per_user=5,
            max_samples=3,
        )
        for sample in data.samples:
            for item in sample.profile:
                assert "text" in item, "Profile items must have 'text'"
                assert "title" in item, "Profile items must have 'title'"
                assert "rating" in item, "Profile items must have 'rating'"

    def test_lamp_format_conversion(self):
        from hare.evaluation.amazon import load_amazon_reviews, amazon_to_lamp_format

        data = load_amazon_reviews(
            category="Digital_Music",
            min_reviews_per_user=5,
            max_samples=5,
        )
        lamp_data = amazon_to_lamp_format(data)
        assert lamp_data.task == "Amazon-Digital_Music"
        assert len(lamp_data.samples) == 5
        assert lamp_data.samples[0].input_text == data.samples[0].input_text
        assert lamp_data.samples[0].target == data.samples[0].target

    def test_baselines_compatible(self):
        """Verify all non-neural baselines work with Amazon data format."""
        from hare.evaluation.amazon import load_amazon_reviews, amazon_to_lamp_format
        from hare.evaluation.baselines import get_all_baselines

        data = load_amazon_reviews(
            category="Digital_Music",
            min_reviews_per_user=5,
            max_samples=3,
        )
        lamp_data = amazon_to_lamp_format(data)
        baselines = get_all_baselines(
            include_neural=False, task="amazon"
        )

        for baseline in baselines:
            sample = lamp_data.samples[0]
            pred = baseline.predict(sample.input_text, sample.profile)
            assert isinstance(pred, str), f"{baseline.name} should return str"
            assert len(pred) > 0, f"{baseline.name} returned empty string"

    def test_rouge_evaluation(self):
        """End-to-end: load data, run naive baseline, compute ROUGE."""
        from hare.evaluation.amazon import load_amazon_reviews, amazon_to_lamp_format
        from hare.evaluation.baselines import TfidfRetrieval, get_task_config
        from hare.evaluation.lamp import evaluate_rouge

        data = load_amazon_reviews(
            category="Digital_Music",
            min_reviews_per_user=5,
            max_samples=5,
        )
        lamp_data = amazon_to_lamp_format(data)
        cfg = get_task_config("amazon")
        baseline = TfidfRetrieval(task_config=cfg)

        predictions = [
            baseline.predict(s.input_text, s.profile)
            for s in lamp_data.samples
        ]
        references = [s.target for s in lamp_data.samples]

        scores = evaluate_rouge(predictions, references)
        assert "rouge1" in scores
        assert "rougeL" in scores
        assert 0 <= scores["rouge1"] <= 1
        assert scores["n_samples"] == 5


class TestAmazonTaskConfig:
    """Tests for Amazon task config (no network needed)."""

    def test_amazon_config_exists(self):
        from hare.evaluation.baselines import get_task_config

        cfg = get_task_config("amazon")
        assert cfg.task_name == "Amazon Reviews"
        assert cfg.content_label == "Product"
        assert cfg.target_label == "Review"
        assert cfg.profile_text_key == "text"

    def test_strip_prefix(self):
        from hare.evaluation.baselines import _strip_prefix, get_task_config

        cfg = get_task_config("amazon")
        text = "Write a review for the following product: Sony WH-1000XM5"
        stripped = _strip_prefix(text, cfg)
        assert stripped.strip() == "Sony WH-1000XM5"
