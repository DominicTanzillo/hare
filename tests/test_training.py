"""Tests for GPT-2 fine-tuning pipeline components."""

import numpy as np
import pytest
import torch

from hare.synthesis.training import (
    SoftPromptProjection,
    SkillDataset,
    ConditionedGPT2,
    TrainingConfig,
    prepare_skill_texts,
)


class TestSoftPromptProjection:
    def test_output_shape(self):
        proj = SoftPromptProjection(z_dim=64, d_model=128, n_prefix=8)
        z = torch.randn(2, 64)
        out = proj(z)
        assert out.shape == (2, 8, 128)

    def test_different_z_different_prefix(self):
        proj = SoftPromptProjection(z_dim=32, d_model=64, n_prefix=4)
        z1 = torch.randn(1, 32)
        z2 = torch.randn(1, 32)
        out1 = proj(z1)
        out2 = proj(z2)
        assert not torch.allclose(out1, out2)


class TestSkillDataset:
    @pytest.fixture
    def tokenizer(self):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("distilgpt2")
        tok.pad_token = tok.eos_token
        return tok

    def test_dataset_length(self, tokenizer):
        texts = ["Hello world", "Goodbye world"]
        z = np.random.randn(2, 16)
        ds = SkillDataset(texts, z, tokenizer, max_length=32)
        assert len(ds) == 2

    def test_item_keys(self, tokenizer):
        texts = ["Test text"]
        z = np.random.randn(1, 16)
        ds = SkillDataset(texts, z, tokenizer, max_length=32)
        item = ds[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "z" in item

    def test_item_shapes(self, tokenizer):
        texts = ["Test text for shape checking"]
        z = np.random.randn(1, 32)
        ds = SkillDataset(texts, z, tokenizer, max_length=64)
        item = ds[0]
        assert item["input_ids"].shape == (64,)
        assert item["attention_mask"].shape == (64,)
        assert item["z"].shape == (32,)


class TestPrepareSkillTexts:
    def test_formats_skills(self):
        skills = [
            {
                "title": "Test Skill",
                "category": "test",
                "description": "A test skill",
                "trigger": "When testing",
                "instructions": "Do the test",
            }
        ]
        texts = prepare_skill_texts(skills)
        assert len(texts) == 1
        assert "# Test Skill" in texts[0]
        assert "Category: test" in texts[0]
        assert "## When to use" in texts[0]
        assert "## Instructions" in texts[0]


class TestConditionedGPT2:
    @pytest.fixture
    def model(self):
        config = TrainingConfig(model_name="distilgpt2", n_prefix_tokens=4, device="cpu")
        return ConditionedGPT2(config, z_dim=32)

    def test_forward(self, model):
        input_ids = torch.randint(0, 1000, (2, 16))
        attention_mask = torch.ones(2, 16, dtype=torch.long)
        z = torch.randn(2, 32)

        outputs = model(input_ids, attention_mask, z)
        assert "loss" in outputs
        assert "logits" in outputs
        assert outputs["loss"].requires_grad

    def test_prefix_frozen_base(self, model):
        """Initially only prefix projection should be trainable."""
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        assert trainable < total  # Base model is frozen
        assert trainable > 0     # Prefix projection is trainable

    def test_unfreeze(self, model):
        trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model.unfreeze_base(unfreeze_layers=1)
        trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable_after > trainable_before

    def test_generate(self, model):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        tokenizer.pad_token = tokenizer.eos_token

        model.eval()
        z = torch.randn(32)
        text = model.generate(z, tokenizer, max_new_tokens=20)
        assert isinstance(text, str)
        assert len(text) > 0
