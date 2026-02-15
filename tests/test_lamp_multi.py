"""Tests for multi-task LaMP support (LaMP-4, LaMP-5, LaMP-7)."""

import pytest

from hare.evaluation.baselines import (
    TaskConfig,
    TASK_CONFIGS,
    get_task_config,
    get_all_baselines,
    _get_profile_target,
    _strip_prefix,
    InputCopy,
    TfidfRetrieval,
    BM25Retrieval,
    RandomProfile,
    MostRecent,
)


# =============================================================================
# TaskConfig
# =============================================================================


class TestTaskConfig:
    def test_all_configs_exist(self):
        assert "lamp4" in TASK_CONFIGS
        assert "lamp5" in TASK_CONFIGS
        assert "lamp7" in TASK_CONFIGS

    def test_get_task_config_case_insensitive(self):
        assert get_task_config("lamp4") is get_task_config("LaMP-4")
        assert get_task_config("lamp5") is get_task_config("LaMP_5")
        assert get_task_config("lamp7") is get_task_config("LAMP7")

    def test_unknown_task_raises(self):
        with pytest.raises(ValueError, match="Unknown task"):
            get_task_config("lamp99")

    def test_lamp4_config(self):
        cfg = get_task_config("lamp4")
        assert cfg.task_name == "LaMP-4"
        assert cfg.content_label == "Article"
        assert cfg.target_label == "Headline"
        assert cfg.profile_text_key == "text"
        assert cfg.profile_target_key == "title"

    def test_lamp5_config(self):
        cfg = get_task_config("lamp5")
        assert cfg.task_name == "LaMP-5"
        assert cfg.content_label == "Abstract"
        assert cfg.target_label == "Title"
        assert cfg.profile_text_key == "abstract"
        assert cfg.profile_target_key == "title"

    def test_lamp7_config(self):
        cfg = get_task_config("lamp7")
        assert cfg.task_name == "LaMP-7"
        assert cfg.content_label == "Tweet"
        assert cfg.target_label == "Paraphrase"
        assert cfg.profile_text_key == "text"
        assert cfg.profile_target_key is None


# =============================================================================
# Helper functions
# =============================================================================


class TestHelpers:
    def test_get_profile_target_with_title(self):
        cfg = get_task_config("lamp4")
        item = {"text": "article body", "title": "the headline"}
        assert _get_profile_target(item, cfg) == "the headline"

    def test_get_profile_target_lamp5(self):
        cfg = get_task_config("lamp5")
        item = {"abstract": "paper body", "title": "Paper Title"}
        assert _get_profile_target(item, cfg) == "Paper Title"

    def test_get_profile_target_lamp7_falls_back_to_text(self):
        cfg = get_task_config("lamp7")
        item = {"text": "just a tweet"}
        assert _get_profile_target(item, cfg) == "just a tweet"

    def test_strip_prefix_lamp4(self):
        cfg = get_task_config("lamp4")
        text = "Generate a headline for the following article: The president visited..."
        assert _strip_prefix(text, cfg) == "The president visited..."

    def test_strip_prefix_lamp5(self):
        cfg = get_task_config("lamp5")
        text = "Generate a title for the following abstract of a paper: We propose a novel..."
        assert _strip_prefix(text, cfg) == "We propose a novel..."

    def test_strip_prefix_lamp7(self):
        cfg = get_task_config("lamp7")
        text = "Paraphrase the following tweet without any explanation before or after it: I am tired"
        assert _strip_prefix(text, cfg) == "I am tired"

    def test_strip_prefix_no_match_returns_original(self):
        cfg = get_task_config("lamp4")
        text = "Some text without a prefix"
        assert _strip_prefix(text, cfg) == "Some text without a prefix"


# =============================================================================
# Baselines with different task configs
# =============================================================================


LAMP5_PROFILE = [
    {"abstract": "We study neural network optimization...", "title": "Deep Learning Methods", "id": "1"},
    {"abstract": "Bayesian inference for recommendation systems...", "title": "Bayesian Approaches", "id": "2"},
    {"abstract": "Reinforcement learning applied to dialogue...", "title": "RL for Dialogue", "id": "3"},
]
LAMP5_INPUT = "Generate a title for the following abstract of a paper: We propose a novel neural method for text generation."

LAMP7_PROFILE = [
    {"text": "just had the best coffee ever", "id": "1"},
    {"text": "feeling tired today but pushing through", "id": "2"},
    {"text": "loving the new album from my favorite band", "id": "3"},
]
LAMP7_INPUT = "Paraphrase the following tweet without any explanation before or after it: I am absolutely exhausted today."

LAMP4_PROFILE = [
    {"text": "The economy grew by 3% this quarter...", "title": "Economy Surges", "id": "1"},
    {"text": "Scientists discover new species in the Amazon...", "title": "New Species Found", "id": "2"},
]
LAMP4_INPUT = "Generate a headline for the following article: Markets rallied today on inflation data."


class TestRandomProfileMultiTask:
    def test_lamp4(self):
        cfg = get_task_config("lamp4")
        b = RandomProfile(seed=42, task_config=cfg)
        result = b.predict(LAMP4_INPUT, LAMP4_PROFILE)
        assert result in ["Economy Surges", "New Species Found"]

    def test_lamp5(self):
        cfg = get_task_config("lamp5")
        b = RandomProfile(seed=42, task_config=cfg)
        result = b.predict(LAMP5_INPUT, LAMP5_PROFILE)
        assert result in ["Deep Learning Methods", "Bayesian Approaches", "RL for Dialogue"]

    def test_lamp7(self):
        cfg = get_task_config("lamp7")
        b = RandomProfile(seed=42, task_config=cfg)
        result = b.predict(LAMP7_INPUT, LAMP7_PROFILE)
        assert result in [item["text"] for item in LAMP7_PROFILE]


class TestMostRecentMultiTask:
    def test_lamp5(self):
        cfg = get_task_config("lamp5")
        b = MostRecent(task_config=cfg)
        assert b.predict(LAMP5_INPUT, LAMP5_PROFILE) == "RL for Dialogue"

    def test_lamp7(self):
        cfg = get_task_config("lamp7")
        b = MostRecent(task_config=cfg)
        assert b.predict(LAMP7_INPUT, LAMP7_PROFILE) == "loving the new album from my favorite band"


class TestInputCopyMultiTask:
    def test_lamp5_strips_prefix(self):
        cfg = get_task_config("lamp5")
        b = InputCopy(task_config=cfg)
        result = b.predict(LAMP5_INPUT, LAMP5_PROFILE)
        assert "Generate a title" not in result
        assert len(result) > 0

    def test_lamp7_strips_prefix(self):
        cfg = get_task_config("lamp7")
        b = InputCopy(task_config=cfg)
        result = b.predict(LAMP7_INPUT, LAMP7_PROFILE)
        assert "Paraphrase" not in result
        assert len(result) > 0


class TestTfidfRetrievalMultiTask:
    def test_lamp5_uses_abstract_key(self):
        cfg = get_task_config("lamp5")
        b = TfidfRetrieval(task_config=cfg)
        result = b.predict(LAMP5_INPUT, LAMP5_PROFILE)
        # Should return a title from the profile
        assert result in ["Deep Learning Methods", "Bayesian Approaches", "RL for Dialogue"]

    def test_lamp7_returns_tweet(self):
        cfg = get_task_config("lamp7")
        b = TfidfRetrieval(task_config=cfg)
        result = b.predict(LAMP7_INPUT, LAMP7_PROFILE)
        assert result in [item["text"] for item in LAMP7_PROFILE]


class TestBM25RetrievalMultiTask:
    def test_lamp5(self):
        cfg = get_task_config("lamp5")
        b = BM25Retrieval(task_config=cfg)
        result = b.predict(LAMP5_INPUT, LAMP5_PROFILE)
        assert result in ["Deep Learning Methods", "Bayesian Approaches", "RL for Dialogue"]

    def test_lamp7(self):
        cfg = get_task_config("lamp7")
        b = BM25Retrieval(task_config=cfg)
        result = b.predict(LAMP7_INPUT, LAMP7_PROFILE)
        assert result in [item["text"] for item in LAMP7_PROFILE]


# =============================================================================
# get_all_baselines
# =============================================================================


class TestGetAllBaselines:
    def test_non_neural_baselines_per_task(self):
        for task in ["lamp4", "lamp5", "lamp7"]:
            baselines = get_all_baselines(include_neural=False, task=task)
            assert len(baselines) == 5
            for b in baselines:
                assert hasattr(b, "task_config")

    def test_baselines_use_correct_config(self):
        baselines = get_all_baselines(include_neural=False, task="lamp5")
        for b in baselines:
            assert b.task_config.task_name == "LaMP-5"

    def test_default_is_lamp4(self):
        baselines = get_all_baselines(include_neural=False)
        for b in baselines:
            assert b.task_config.task_name == "LaMP-4"
