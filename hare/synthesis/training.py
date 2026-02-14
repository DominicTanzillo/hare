"""GPT-2 fine-tuning pipeline for HARE's generative decoder.

Fine-tunes a causal language model (GPT-2 / DistilGPT-2) on Claude Skills,
conditioned on HARE's attended representation z concatenated with user context.

The conditioning signal is injected as a learned prefix: the synthesis vector z
is projected into the token embedding space and prepended as soft prompt tokens.
This lets the same base LM generate very different outputs depending on what
HARE's attention mechanism attended to.

Usage:
    python -m hare.synthesis.training --epochs 3 --batch-size 4
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


@dataclass
class TrainingConfig:
    """Configuration for fine-tuning."""
    model_name: str = "distilgpt2"
    max_length: int = 512
    n_prefix_tokens: int = 8       # Number of soft prompt tokens from z
    batch_size: int = 4
    learning_rate: float = 5e-5
    epochs: int = 3
    warmup_ratio: float = 0.1
    output_dir: str = "checkpoints"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


class SoftPromptProjection(nn.Module):
    """Projects HARE's synthesis vector z into soft prompt token embeddings.

    z (from attention) -> linear projection -> reshape to (n_prefix, d_model)

    These prefix embeddings are prepended to the input tokens, conditioning
    the generation on what HARE attended to.

    Parameters
    ----------
    z_dim : int
        Dimension of HARE's synthesis vector.
    d_model : int
        Hidden dimension of the language model.
    n_prefix : int
        Number of soft prompt tokens to generate.
    """

    def __init__(self, z_dim: int, d_model: int, n_prefix: int = 8) -> None:
        super().__init__()
        self.n_prefix = n_prefix
        self.d_model = d_model
        self.projection = nn.Sequential(
            nn.Linear(z_dim, d_model * n_prefix // 2),
            nn.GELU(),
            nn.Linear(d_model * n_prefix // 2, d_model * n_prefix),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Project z into prefix embeddings.

        Parameters
        ----------
        z : tensor of shape (batch, z_dim)

        Returns
        -------
        tensor of shape (batch, n_prefix, d_model)
        """
        projected = self.projection(z)  # (batch, d_model * n_prefix)
        return projected.view(-1, self.n_prefix, self.d_model)


class SkillDataset(Dataset):
    """Dataset of Claude Skills formatted for causal LM training.

    Each item is a skill formatted as text, tokenized, and paired with
    a synthesis vector z (which conditions the generation).

    Parameters
    ----------
    texts : list of str
        Skill texts formatted for generation.
    synthesis_vectors : array of shape (n_skills, z_dim)
        Precomputed HARE synthesis vectors for each skill.
    tokenizer : AutoTokenizer
        HuggingFace tokenizer.
    max_length : int
        Maximum token sequence length.
    """

    def __init__(
        self,
        texts: list[str],
        synthesis_vectors: np.ndarray,
        tokenizer,
        max_length: int = 512,
    ) -> None:
        self.texts = texts
        self.synthesis_vectors = synthesis_vectors
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        text = self.texts[idx]
        z = self.synthesis_vectors[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "z": torch.tensor(z, dtype=torch.float32),
        }


class ConditionedGPT2(nn.Module):
    """GPT-2 with soft prompt conditioning from HARE synthesis vectors.

    Architecture:
        z -> SoftPromptProjection -> prefix_embeds (n_prefix, d_model)
        input_tokens -> token_embeds (seq_len, d_model)
        [prefix_embeds; token_embeds] -> GPT-2 -> next token prediction

    The prefix tokens are not part of the vocabulary -- they're learned
    projections of HARE's synthesis vector, injecting the attended
    knowledge pool information into the generation process.
    """

    def __init__(self, config: TrainingConfig, z_dim: int) -> None:
        super().__init__()
        self.config = config

        self.base_model = AutoModelForCausalLM.from_pretrained(config.model_name)
        d_model = self.base_model.config.n_embd

        self.prefix_projection = SoftPromptProjection(
            z_dim=z_dim, d_model=d_model, n_prefix=config.n_prefix_tokens
        )

        # Freeze base model initially (only train prefix projection)
        for param in self.base_model.parameters():
            param.requires_grad = False

    def unfreeze_base(self, unfreeze_layers: int = 2) -> None:
        """Unfreeze the last N transformer layers for fine-tuning."""
        # Unfreeze LM head
        for param in self.base_model.lm_head.parameters():
            param.requires_grad = True

        # Unfreeze last N layers
        n_layers = len(self.base_model.transformer.h)
        for i in range(max(0, n_layers - unfreeze_layers), n_layers):
            for param in self.base_model.transformer.h[i].parameters():
                param.requires_grad = True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        z: torch.Tensor,
    ) -> dict:
        """Forward pass with soft prompt prefix.

        Parameters
        ----------
        input_ids : tensor of shape (batch, seq_len)
        attention_mask : tensor of shape (batch, seq_len)
        z : tensor of shape (batch, z_dim)

        Returns
        -------
        dict with 'loss' and 'logits'
        """
        batch_size = input_ids.shape[0]

        # Generate prefix embeddings from z
        prefix_embeds = self.prefix_projection(z)  # (batch, n_prefix, d_model)

        # Get token embeddings from base model
        token_embeds = self.base_model.transformer.wte(input_ids)  # (batch, seq_len, d_model)

        # Concatenate: [prefix | tokens]
        inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)

        # Extend attention mask for prefix tokens (always attended to)
        prefix_mask = torch.ones(batch_size, self.config.n_prefix_tokens,
                                 device=attention_mask.device, dtype=attention_mask.dtype)
        extended_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        # Create labels: -100 for prefix (don't compute loss on conditioning tokens)
        prefix_labels = torch.full(
            (batch_size, self.config.n_prefix_tokens), -100,
            device=input_ids.device, dtype=input_ids.dtype
        )
        labels = torch.cat([prefix_labels, input_ids], dim=1)

        # Forward through GPT-2 with embeddings input
        outputs = self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_mask,
            labels=labels,
        )

        return {"loss": outputs.loss, "logits": outputs.logits}

    @torch.no_grad()
    def generate(
        self,
        z: torch.Tensor,
        tokenizer,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Generate text conditioned on a synthesis vector.

        Parameters
        ----------
        z : tensor of shape (z_dim,) or (1, z_dim)
        tokenizer : HuggingFace tokenizer
        max_new_tokens : int
        temperature : float
        top_p : float

        Returns
        -------
        str
            Generated text.
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)

        device = next(self.parameters()).device
        z = z.to(device)

        # Create prefix embeddings
        prefix_embeds = self.prefix_projection(z)  # (1, n_prefix, d_model)

        # Start with just the prefix as context
        # Use BOS token as initial input
        bos_id = tokenizer.bos_token_id or tokenizer.eos_token_id
        input_ids = torch.tensor([[bos_id]], device=device)
        token_embeds = self.base_model.transformer.wte(input_ids)
        inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)

        # Generate autoregressively
        generated = self.base_model.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

        # Decode, skipping the BOS token
        text = tokenizer.decode(generated[0][1:], skip_special_tokens=True)
        return text


def prepare_skill_texts(skills: Sequence[dict]) -> list[str]:
    """Format skills for causal LM training."""
    texts = []
    for s in skills:
        text = (
            f"# {s['title']}\n"
            f"Category: {s['category']}\n\n"
            f"{s['description']}\n\n"
            f"## When to use\n{s['trigger']}\n\n"
            f"## Instructions\n{s['instructions']}"
        )
        texts.append(text)
    return texts


def train(
    config: TrainingConfig,
    skills: list[dict],
    synthesis_vectors: np.ndarray,
) -> ConditionedGPT2:
    """Fine-tune GPT-2 on skills conditioned on synthesis vectors.

    Parameters
    ----------
    config : TrainingConfig
    skills : list of skill dicts
    synthesis_vectors : array of shape (n_skills, z_dim)

    Returns
    -------
    ConditionedGPT2
        The fine-tuned model.
    """
    torch.manual_seed(config.seed)

    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare data
    texts = prepare_skill_texts(skills)
    z_dim = synthesis_vectors.shape[1]

    dataset = SkillDataset(texts, synthesis_vectors, tokenizer, config.max_length)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # Build model
    model = ConditionedGPT2(config, z_dim=z_dim)
    model.to(config.device)

    # Optimizer (only prefix projection is trainable initially)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
    )

    total_steps = len(dataloader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps,
    )

    # Training loop
    model.train()
    for epoch in range(config.epochs):
        total_loss = 0.0
        n_batches = 0

        # Unfreeze base model layers after first epoch
        if epoch == 1:
            model.unfreeze_base(unfreeze_layers=2)
            # Re-create optimizer with all trainable params
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=config.learning_rate * 0.1,  # Lower LR for base model
            )

        for batch in dataloader:
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            z = batch["z"].to(config.device)

            outputs = model(input_ids, attention_mask, z)
            loss = outputs["loss"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch + 1}/{config.epochs} | Loss: {avg_loss:.4f}")

    # Save checkpoint
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "hare_decoder.pt")
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Fine-tune HARE decoder on Claude Skills")
    parser.add_argument("--model", default="distilgpt2", help="Base model name")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = TrainingConfig(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_dir=args.output_dir,
    )
    if args.device:
        config.device = args.device

    # Load skills
    from hare.data.skills import load_builtin_skills, skills_to_texts
    from hare.utils.embeddings import TfidfEmbedder

    skills = load_builtin_skills()
    print(f"Training on {len(skills)} skills")

    # Generate synthesis vectors using HARE
    from hare.bandits.attentive_bandit import HARE

    texts = skills_to_texts(skills)
    embedder = TfidfEmbedder(max_features=500, output_dim=64)
    embedder.fit(texts)
    skill_embs = embedder.encode(texts)

    hare = HARE(
        d_knowledge=embedder.dim,
        d_user=32,
        n_clusters=min(5, len(skills)),
        n_heads=4,
        d_k=32,
        d_v=32,
        alpha=1.0,
        seed=42,
    )
    hare.set_knowledge_pool(skill_embs)

    # Generate a synthesis vector for each skill (using the skill itself as query)
    synthesis_vectors = []
    for emb in skill_embs:
        z = hare.recommend(emb, user_id="training")
        synthesis_vectors.append(z)
    synthesis_vectors = np.array(synthesis_vectors)

    print(f"Synthesis vectors: {synthesis_vectors.shape}")
    print(f"Training on device: {config.device}")

    model = train(config, skills, synthesis_vectors)

    # Quick generation test
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    test_z = torch.tensor(synthesis_vectors[0], dtype=torch.float32)
    generated = model.generate(test_z, tokenizer, max_new_tokens=100)
    print(f"\nSample generation (conditioned on skill 0):\n{generated[:500]}")


if __name__ == "__main__":
    main()
