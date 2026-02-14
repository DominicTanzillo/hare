"""Generative synthesis decoder for HARE.

Takes the attended representation z (from uncertainty-augmented attention)
and the user state u_t, and produces novel text content.

Two modes:
1. Prototype: Weighted interpolation of existing text features → nearest-neighbor
   blending. Fast, no GPU required, good for paper experiments.
2. LM-based: Fine-tuned GPT-2 conditioned on z ⊕ u_t → true generation.
   Requires GPU, produces genuinely novel text.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics.pairwise import cosine_similarity


class Decoder(Protocol):
    """Protocol for synthesis decoders."""

    def generate(self, synthesis: NDArray, candidates: list[str], candidate_embeddings: NDArray) -> str: ...


class InterpolationDecoder:
    """Prototype decoder: weighted blend of existing content.

    Uses cosine similarity between the synthesized representation and
    candidate embeddings to produce a weighted interpolation of candidate texts.
    The output is the top-k most relevant candidates, blended by attention weight.

    This is NOT true generation — it's a principled mashup. But it demonstrates
    that the attention mechanism is producing meaningful synthesis vectors,
    and it runs without a GPU.

    Parameters
    ----------
    top_k : int
        Number of candidates to blend.
    temperature : float
        Softmax temperature for blending weights. Lower = more peaked.
    """

    def __init__(self, top_k: int = 3, temperature: float = 0.5) -> None:
        self.top_k = top_k
        self.temperature = temperature

    @staticmethod
    def _align_dims(synthesis: NDArray, candidate_embeddings: NDArray) -> NDArray:
        """Truncate synthesis to match candidate embedding dimension."""
        d_cand = candidate_embeddings.shape[1]
        return synthesis.ravel()[:d_cand]

    def generate(
        self,
        synthesis: NDArray,
        candidates: list[str],
        candidate_embeddings: NDArray,
    ) -> str:
        """Generate text by blending top-k most similar candidates.

        Parameters
        ----------
        synthesis : array of shape (d,) or (d_input,)
            The synthesized representation from HARE's attention.
            Will be truncated to match candidate_embeddings dimension.
        candidates : list of str
            The text content of each item in the knowledge pool.
        candidate_embeddings : array of shape (n_items, d_knowledge)
            Embeddings of the candidates.

        Returns
        -------
        str
            Blended output text.
        """
        s = self._align_dims(synthesis, candidate_embeddings)
        # Compute similarity between synthesis and all candidates
        sim = cosine_similarity(s.reshape(1, -1), candidate_embeddings)[0]

        # Top-k selection
        k = min(self.top_k, len(candidates))
        top_indices = np.argsort(sim)[-k:][::-1]

        # Softmax weights over top-k similarities
        top_sims = sim[top_indices] / self.temperature
        weights = np.exp(top_sims - np.max(top_sims))
        weights = weights / weights.sum()

        # Build blended output
        sections = []
        for idx, w in zip(top_indices, weights):
            sections.append(f"[weight={w:.2f}] {candidates[idx]}")

        header = (
            f"=== HARE Synthesis (blended from top-{k} items) ===\n"
            f"Blend weights: {dict(zip(top_indices.tolist(), weights.round(3).tolist()))}\n"
            f"{'=' * 50}\n\n"
        )

        return header + "\n\n---\n\n".join(sections)

    def get_blend_weights(
        self,
        synthesis: NDArray,
        candidate_embeddings: NDArray,
    ) -> tuple[NDArray, NDArray]:
        """Get blend weights without generating text.

        Returns
        -------
        tuple of (top_indices, weights)
        """
        s = self._align_dims(synthesis, candidate_embeddings)
        sim = cosine_similarity(s.reshape(1, -1), candidate_embeddings)[0]
        k = min(self.top_k, len(candidate_embeddings))
        top_indices = np.argsort(sim)[-k:][::-1]
        top_sims = sim[top_indices] / self.temperature
        weights = np.exp(top_sims - np.max(top_sims))
        weights = weights / weights.sum()
        return top_indices, weights


class PromptConditionedDecoder:
    """LM-based decoder: generates novel text via a prompted language model.

    Constructs a prompt from the top-k attended items and user context,
    then generates via a causal LM. This produces genuinely novel content
    rather than blending existing text.

    Parameters
    ----------
    model_name : str
        HuggingFace model name (e.g. "gpt2", "distilgpt2").
    max_new_tokens : int
        Maximum tokens to generate.
    top_k_context : int
        Number of knowledge items to include as context.
    temperature : float
        Generation temperature.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        max_new_tokens: int = 256,
        top_k_context: int = 3,
        temperature: float = 0.7,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.top_k_context = top_k_context
        self.temperature = temperature
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        if self._model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

    def generate(
        self,
        synthesis: NDArray,
        candidates: list[str],
        candidate_embeddings: NDArray,
    ) -> str:
        """Generate novel text conditioned on attended knowledge items.

        Parameters
        ----------
        synthesis : array of shape (d,)
            Synthesized representation from HARE.
        candidates : list of str
            Text content of knowledge pool items.
        candidate_embeddings : array of shape (n_items, d)
            Embeddings of candidates.

        Returns
        -------
        str
            Generated text.
        """
        import torch

        self._load()

        # Find top-k items by cosine similarity to synthesis
        d_cand = candidate_embeddings.shape[1]
        s = synthesis.ravel()[:d_cand]
        sim = cosine_similarity(s.reshape(1, -1), candidate_embeddings)[0]
        k = min(self.top_k_context, len(candidates))
        top_indices = np.argsort(sim)[-k:][::-1]

        # Build prompt with attended context
        context_items = [candidates[i] for i in top_indices]
        prompt = self._build_prompt(context_items)

        # Generate
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        # Decode only the generated part
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(generated_ids, skip_special_tokens=True)

    def _build_prompt(self, context_items: list[str]) -> str:
        """Build a generation prompt from attended context items."""
        context = "\n\n---\n\n".join(context_items)
        return (
            f"Below are relevant reference items:\n\n{context}\n\n"
            f"---\n\n"
            f"Based on the above references, synthesize a new, improved version "
            f"that combines the best elements:\n\n"
        )
