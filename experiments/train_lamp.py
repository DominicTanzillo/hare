#!/usr/bin/env python3
"""Fine-tune DistilGPT2 on LaMP generation tasks.

Supports LaMP-4 (headlines), LaMP-5 (scholarly titles), and LaMP-7 (tweets).

All three neural baselines (Vanilla GPT-2, RAG+GPT-2, HARE+GPT-2) share the
same fine-tuned base model. They differ only in how context is constructed at
inference time:
  - Vanilla: content only
  - RAG: content + TF-IDF-retrieved profile examples
  - HARE: content + user-conditioned attention-retrieved examples

This script fine-tunes DistilGPT2 on the training set using the format:
    {ContentLabel}: {content_text}\n\n{TargetLabel}: {target}<eos>

Usage:
    # Quick test (100 samples, 1 epoch)
    python experiments/train_lamp.py --task lamp4 --max-samples 100 --epochs 1

    # Full training on LaMP-5
    python experiments/train_lamp.py --task lamp5 --epochs 3 --batch-size 8

    # Resume from checkpoint
    python experiments/train_lamp.py --task lamp4 --resume checkpoints/lamp4/
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from hare.evaluation.lamp import load_lamp
from hare.evaluation.baselines import get_task_config, TaskConfig


class LaMPTrainDataset(Dataset):
    """LaMP training dataset for causal LM fine-tuning.

    Each sample is formatted as:
        {ContentLabel}: {truncated_content}\n\n{TargetLabel}: {target}<eos>

    The loss is computed only on the target tokens (teacher forcing).
    """

    def __init__(
        self,
        data,
        tokenizer,
        max_length: int = 256,
        task_config: TaskConfig | None = None,
        include_profile_examples: bool = False,
        n_examples: int = 2,
    ) -> None:
        self.samples = data.samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_config = task_config or get_task_config("lamp4")
        self.include_profile_examples = include_profile_examples
        self.n_examples = n_examples

    def __len__(self) -> int:
        return len(self.samples)

    def _format_sample(self, sample) -> tuple[str, str]:
        """Format a sample into prompt and target."""
        cfg = self.task_config
        content = re.sub(
            cfg.instruction_prefix,
            "", sample.input_text, flags=re.IGNORECASE,
        )

        if self.include_profile_examples and sample.profile:
            n = min(self.n_examples, len(sample.profile))
            examples = sample.profile[:n]
            parts = []
            for ex in examples:
                ex_text = ex.get(cfg.profile_text_key, "")[:cfg.max_example_chars - 50]
                if cfg.profile_target_key and cfg.profile_target_key in ex:
                    ex_target = ex[cfg.profile_target_key]
                else:
                    ex_target = ex.get("text", "")
                parts.append(
                    f"{cfg.content_label}: {ex_text}\n{cfg.target_label}: {ex_target}"
                )
            parts.append(
                f"{cfg.content_label}: {content[:cfg.max_input_chars - 100]}\n{cfg.target_label}:"
            )
            prompt = "\n\n".join(parts)
        else:
            prompt = f"{cfg.content_label}: {content[:cfg.max_input_chars]}\n{cfg.target_label}:"

        return prompt, sample.target

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        prompt, target = self._format_sample(sample)

        # Full text for training: prompt + target + eos
        full_text = f"{prompt} {target}{self.tokenizer.eos_token}"

        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Create labels: -100 for prompt tokens (only train on target)
        prompt_encoding = self.tokenizer(
            prompt + " ",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        prompt_len = prompt_encoding["input_ids"].shape[1]

        labels = input_ids.clone()
        labels[:prompt_len] = -100  # Mask prompt tokens
        labels[attention_mask == 0] = -100  # Mask padding

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def train_lamp(
    task: str = "lamp4",
    model_name: str = "distilgpt2",
    max_samples: int | None = None,
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    max_length: int = 256,
    output_dir: str | None = None,
    resume_from: str | None = None,
    seed: int = 42,
) -> Path:
    """Fine-tune DistilGPT2 on a LaMP generation task.

    Returns the path to the saved checkpoint.
    """
    task_key = task.lower().replace("-", "").replace("_", "")
    task_config = get_task_config(task_key)

    if output_dir is None:
        output_dir = f"checkpoints/{task_key}"

    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print(f"{task_config.task_name} Fine-tuning: DistilGPT2 for "
          f"{task_config.target_label} Generation")
    print("=" * 60)
    print(f"  Task: {task_config.task_name}")
    print(f"  Model: {model_name}")
    print(f"  Device: {device}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Max length: {max_length}")

    # Load tokenizer and model
    if resume_from:
        print(f"\nResuming from {resume_from}...")
        tokenizer = AutoTokenizer.from_pretrained(resume_from)
        model = AutoModelForCausalLM.from_pretrained(resume_from)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.to(device)

    # Load training data
    print(f"\nLoading {task_config.task_name} training set...")
    train_data = load_lamp(task_key, split="train", max_samples=max_samples)
    print(f"  {len(train_data)} training samples")

    # Load small dev set for validation
    print(f"Loading {task_config.task_name} dev set (for validation)...")
    val_data = load_lamp(task_key, split="dev", max_samples=min(50, max_samples or 50))
    print(f"  {len(val_data)} validation samples")

    # Create datasets
    train_dataset = LaMPTrainDataset(
        train_data, tokenizer, max_length, task_config=task_config,
    )
    val_dataset = LaMPTrainDataset(
        val_data, tokenizer, max_length, task_config=task_config,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
    )

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    # Training loop
    print(f"\nTraining for {epochs} epochs ({total_steps} steps)...\n")
    best_val_loss = float("inf")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        # Train
        model.train()
        total_loss = 0.0
        n_batches = 0

        for i, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            n_batches += 1

            if (i + 1) % 50 == 0:
                avg = total_loss / n_batches
                print(f"  Epoch {epoch+1} | Step {i+1}/{len(train_loader)} | Loss: {avg:.4f}")

        train_loss = total_loss / max(n_batches, 1)

        # Validate
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                val_loss += outputs.loss.item()
                val_batches += 1

        val_loss /= max(val_batches, 1)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f}")

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(out)
            tokenizer.save_pretrained(out)
            print(f"  -> Saved best model (val_loss={val_loss:.4f})")

    # Quick generation test
    print("\n" + "=" * 60)
    print("Sample generations from fine-tuned model:")
    print("=" * 60)

    model.eval()
    cfg = task_config
    for i, sample in enumerate(val_data.samples[:3]):
        content = re.sub(
            cfg.instruction_prefix,
            "", sample.input_text, flags=re.IGNORECASE,
        )
        prompt = f"{cfg.content_label}: {content[:cfg.max_input_chars]}\n{cfg.target_label}:"

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=200)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=32,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip().split("\n")[0]
        print(f"\n  Sample {i+1}:")
        print(f"    Reference: {sample.target}")
        print(f"    Generated: {text}")

    # Save training metadata
    meta = {
        "task": task_key,
        "task_name": task_config.task_name,
        "model_name": model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_length": max_length,
        "n_train_samples": len(train_data),
        "n_val_samples": len(val_data),
        "best_val_loss": best_val_loss,
        "seed": seed,
    }
    with open(out / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nModel saved to {out}")
    print(f"Best validation loss: {best_val_loss:.4f}")

    return out


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilGPT2 on LaMP generation tasks"
    )
    parser.add_argument(
        "--task", default="lamp4",
        choices=["lamp4", "lamp5", "lamp7"],
        help="LaMP task to train on (default: lamp4).",
    )
    parser.add_argument("--model", default="distilgpt2", help="Base model name")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit training samples (for quick testing)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--output-dir", default=None,
                        help="Output dir (default: checkpoints/{task})")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint dir")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_lamp(
        task=args.task,
        model_name=args.model,
        max_samples=args.max_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
        output_dir=args.output_dir,
        resume_from=args.resume,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
