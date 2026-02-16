"""Three-tier baseline suite for LaMP evaluation.

Supports LaMP-4 (headlines), LaMP-5 (scholarly titles), and LaMP-7 (tweets).

Tier 1 -- Naive baselines:
    - RandomProfile: randomly select a target from user's profile
    - MostRecent: use the most recent profile target
    - InputCopy: extract first sentence of input as output

Tier 2 -- Classical ML:
    - TfidfRetrieval: TF-IDF cosine similarity retrieval from profile
    - BM25Retrieval: BM25-based retrieval from profile

Tier 3 -- Neural / Deep Learning:
    - VanillaGPT2: fine-tuned DistilGPT2 (no personalization)
    - RAGGPT2: retrieve top-k from profile, condition GPT2 generation
    - HareGPT2: HARE user modeling + uncertainty attention + conditioned GPT2

Each baseline implements the same interface:
    model.predict(input_text, profile) -> str
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =============================================================================
# Task configuration
# =============================================================================

@dataclass
class TaskConfig:
    """Task-specific configuration for LaMP baselines."""
    task_name: str              # "LaMP-4", "LaMP-5", "LaMP-7"
    instruction_prefix: str     # regex pattern to strip from input
    content_label: str          # "Article", "Abstract", "Tweet"
    target_label: str           # "Headline", "Title", "Paraphrase"
    profile_text_key: str       # "text" or "abstract"
    profile_target_key: str | None  # "title" or None (LaMP-7)
    max_input_chars: int = 400
    max_example_chars: int = 200


TASK_CONFIGS: dict[str, TaskConfig] = {
    "lamp4": TaskConfig(
        task_name="LaMP-4",
        instruction_prefix=r"^Generate a headline for the following article:\s*",
        content_label="Article",
        target_label="Headline",
        profile_text_key="text",
        profile_target_key="title",
    ),
    "lamp5": TaskConfig(
        task_name="LaMP-5",
        instruction_prefix=r"^Generate a title for the following abstract of a paper:\s*",
        content_label="Abstract",
        target_label="Title",
        profile_text_key="abstract",
        profile_target_key="title",
    ),
    "lamp7": TaskConfig(
        task_name="LaMP-7",
        instruction_prefix=(
            r"^Paraphrase the following tweet without any explanation"
            r" before or after it:\s*"
        ),
        content_label="Tweet",
        target_label="Paraphrase",
        profile_text_key="text",
        profile_target_key=None,
        max_input_chars=280,
        max_example_chars=140,
    ),
    "amazon": TaskConfig(
        task_name="Amazon Reviews",
        instruction_prefix=r"^Write a review for the following product:\s*",
        content_label="Product",
        target_label="Review",
        profile_text_key="text",
        profile_target_key="text",
        max_input_chars=300,
        max_example_chars=300,
    ),
}


def get_task_config(task: str) -> TaskConfig:
    """Get task configuration by name. Accepts 'lamp4', 'LaMP-4', etc."""
    key = task.lower().replace("-", "").replace("_", "")
    if key not in TASK_CONFIGS:
        raise ValueError(f"Unknown task: {task}. Choose from: {list(TASK_CONFIGS.keys())}")
    return TASK_CONFIGS[key]


def _get_profile_target(item: dict, cfg: TaskConfig) -> str:
    """Extract the target text from a profile item using task config."""
    if cfg.profile_target_key and cfg.profile_target_key in item:
        return item[cfg.profile_target_key]
    return item.get("text", "")


def _strip_prefix(input_text: str, cfg: TaskConfig) -> str:
    """Strip the task instruction prefix from input text."""
    return re.sub(cfg.instruction_prefix, "", input_text, flags=re.IGNORECASE)


class Baseline(Protocol):
    """Protocol for all baselines."""
    name: str

    def predict(self, input_text: str, profile: list[dict]) -> str: ...


# =============================================================================
# Tier 1: Naive Baselines
# =============================================================================

class RandomProfile:
    """Randomly select a target from the user's profile."""
    name = "Random (profile)"

    def __init__(self, seed: int = 42, task_config: TaskConfig | None = None) -> None:
        self.rng = np.random.default_rng(seed)
        self.task_config = task_config or TASK_CONFIGS["lamp4"]

    def predict(self, input_text: str, profile: list[dict]) -> str:
        if not profile:
            return ""
        item = profile[self.rng.integers(0, len(profile))]
        return _get_profile_target(item, self.task_config)


class MostRecent:
    """Use the last profile item's target (assumes temporal ordering)."""
    name = "Most Recent"

    def __init__(self, task_config: TaskConfig | None = None) -> None:
        self.task_config = task_config or TASK_CONFIGS["lamp4"]

    def predict(self, input_text: str, profile: list[dict]) -> str:
        if not profile:
            return ""
        item = profile[-1]
        return _get_profile_target(item, self.task_config)


class InputCopy:
    """Extract the first sentence of the input as the output."""
    name = "Input Copy"

    _SENTENCE_RE = re.compile(r"(.+?[.!?])\s", re.DOTALL)

    def __init__(self, task_config: TaskConfig | None = None) -> None:
        self.task_config = task_config or TASK_CONFIGS["lamp4"]

    def predict(self, input_text: str, profile: list[dict]) -> str:
        text = _strip_prefix(input_text, self.task_config)
        match = self._SENTENCE_RE.match(text)
        if match:
            sentence = match.group(1).strip()
            words = sentence.split()
            return " ".join(words[:15])
        return " ".join(text.split()[:10])


# =============================================================================
# Tier 2: Classical ML Baselines
# =============================================================================

class TfidfRetrieval:
    """Retrieve the most similar profile target by TF-IDF cosine similarity.

    For each test input, compute TF-IDF similarity to all profile texts,
    then return the target of the most similar profile item.
    """
    name = "TF-IDF Retrieval"

    def __init__(self, task_config: TaskConfig | None = None) -> None:
        self.task_config = task_config or TASK_CONFIGS["lamp4"]

    def predict(self, input_text: str, profile: list[dict]) -> str:
        if not profile:
            return ""

        cfg = self.task_config
        content = _strip_prefix(input_text, cfg)

        profile_texts = [item.get(cfg.profile_text_key, "") for item in profile]
        all_texts = profile_texts + [content]

        try:
            vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
            tfidf = vectorizer.fit_transform(all_texts)
        except ValueError:
            return _get_profile_target(profile[0], cfg)

        input_vec = tfidf[-1:]
        profile_vecs = tfidf[:-1]
        sims = cosine_similarity(input_vec, profile_vecs)[0]

        best_idx = int(np.argmax(sims))
        return _get_profile_target(profile[best_idx], cfg)


class BM25Retrieval:
    """BM25-based retrieval from profile.

    Simple BM25 implementation using TF-IDF with sublinear TF and
    IDF weighting, retrieving the profile target most relevant
    to the input.
    """
    name = "BM25 Retrieval"

    def __init__(self, task_config: TaskConfig | None = None) -> None:
        self.task_config = task_config or TASK_CONFIGS["lamp4"]

    def predict(self, input_text: str, profile: list[dict]) -> str:
        if not profile:
            return ""

        cfg = self.task_config
        content = _strip_prefix(input_text, cfg)

        profile_texts = [item.get(cfg.profile_text_key, "") for item in profile]
        all_texts = profile_texts + [content]

        try:
            vectorizer = TfidfVectorizer(
                max_features=5000, stop_words="english",
                sublinear_tf=True,
            )
            tfidf = vectorizer.fit_transform(all_texts)
        except ValueError:
            return _get_profile_target(profile[0], cfg)

        input_vec = tfidf[-1:]
        profile_vecs = tfidf[:-1]
        sims = cosine_similarity(input_vec, profile_vecs)[0]

        best_idx = int(np.argmax(sims))
        return _get_profile_target(profile[best_idx], cfg)


# =============================================================================
# Tier 3: Neural / Deep Learning Baselines
# =============================================================================

class VanillaGPT2:
    """Fine-tuned DistilGPT2 for text generation (no personalization).

    Same model for all users. Input: content text. Output: target text.
    No user profile is used.
    """
    name = "Vanilla GPT-2"

    def __init__(
        self,
        model_name: str = "distilgpt2",
        max_new_tokens: int = 32,
        checkpoint: str | None = None,
        task_config: TaskConfig | None = None,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.checkpoint = checkpoint
        self.task_config = task_config or TASK_CONFIGS["lamp4"]
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer
        load_from = self.checkpoint or self.model_name
        self._tokenizer = AutoTokenizer.from_pretrained(load_from)
        self._model = AutoModelForCausalLM.from_pretrained(load_from)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model.eval()

    def predict(self, input_text: str, profile: list[dict]) -> str:
        import torch
        self._load()

        cfg = self.task_config
        content = _strip_prefix(input_text, cfg)
        prompt = f"{cfg.content_label}: {content[:cfg.max_input_chars]}\n\n{cfg.target_label}:"

        inputs = self._tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=480
        )
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        text = self._tokenizer.decode(generated, skip_special_tokens=True)
        return text.strip().split("\n")[0].strip()


class RAGGPT2:
    """RAG baseline: retrieve top-k profile items, condition GPT-2 generation.

    Retrieves the k most similar profile items (by TF-IDF cosine),
    includes their targets as few-shot examples, then generates.
    This is the standard RAG approach -- user-independent retrieval.
    """
    name = "RAG + GPT-2"

    def __init__(
        self,
        model_name: str = "distilgpt2",
        max_new_tokens: int = 32,
        top_k: int = 3,
        checkpoint: str | None = None,
        task_config: TaskConfig | None = None,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.top_k = top_k
        self.checkpoint = checkpoint
        self.task_config = task_config or TASK_CONFIGS["lamp4"]
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer
        load_from = self.checkpoint or self.model_name
        self._tokenizer = AutoTokenizer.from_pretrained(load_from)
        self._model = AutoModelForCausalLM.from_pretrained(load_from)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model.eval()

    def _retrieve_examples(self, content: str, profile: list[dict]) -> list[dict]:
        """Retrieve top-k most similar profile items."""
        if not profile:
            return []

        cfg = self.task_config
        profile_texts = [item.get(cfg.profile_text_key, "") for item in profile]
        all_texts = profile_texts + [content]

        try:
            vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
            tfidf = vectorizer.fit_transform(all_texts)
        except ValueError:
            return profile[:self.top_k]

        sims = cosine_similarity(tfidf[-1:], tfidf[:-1])[0]
        top_k = min(self.top_k, len(profile))
        top_indices = np.argsort(sims)[-top_k:][::-1]
        return [profile[i] for i in top_indices]

    def predict(self, input_text: str, profile: list[dict]) -> str:
        import torch
        self._load()

        cfg = self.task_config
        content = _strip_prefix(input_text, cfg)

        examples = self._retrieve_examples(content, profile)

        prompt_parts = []
        for ex in examples:
            ex_text = ex.get(cfg.profile_text_key, "")[:cfg.max_example_chars]
            ex_target = _get_profile_target(ex, cfg)
            prompt_parts.append(
                f"{cfg.content_label}: {ex_text}\n{cfg.target_label}: {ex_target}"
            )

        prompt_parts.append(f"{cfg.content_label}: {content[:cfg.max_input_chars - 100]}\n{cfg.target_label}:")
        prompt = "\n\n".join(prompt_parts)

        inputs = self._tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=480
        )
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        text = self._tokenizer.decode(generated, skip_special_tokens=True)
        return text.strip().split("\n")[0].strip()


class HareGPT2:
    """HARE-conditioned GPT-2: full system with user modeling.

    For each user:
    1. Build user state from their profile interactions (via HARE)
    2. Compute uncertainty-augmented attention over the profile knowledge pool
    3. Retrieve top-k items based on HARE's attention weights (user-conditioned)
    4. Generate with GPT-2 conditioned on the attention-selected examples

    Unlike RAG, the retrieval is USER-CONDITIONED: different users
    with the same query get different retrieved examples because
    their user states differ.

    When attention_checkpoint is provided, uses LearnableHARE with trained
    projection matrices instead of the random-projection HARE.
    """
    name = "HARE + GPT-2"

    def __init__(
        self,
        model_name: str = "distilgpt2",
        max_new_tokens: int = 32,
        top_k: int = 3,
        n_warmup: int = 5,
        checkpoint: str | None = None,
        task_config: TaskConfig | None = None,
        attention_checkpoint: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.top_k = top_k
        self.n_warmup = n_warmup
        self.checkpoint = checkpoint
        self.task_config = task_config or TASK_CONFIGS["lamp4"]
        self.attention_checkpoint = attention_checkpoint
        self._model = None
        self._tokenizer = None
        self._embedder = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer
        load_from = self.checkpoint or self.model_name
        self._tokenizer = AutoTokenizer.from_pretrained(load_from)
        self._model = AutoModelForCausalLM.from_pretrained(load_from)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model.eval()

    def _get_embedder(self):
        if self._embedder is None:
            from hare.utils.embeddings import TfidfEmbedder
            self._embedder = TfidfEmbedder(max_features=2000, output_dim=64)
        return self._embedder

    def _build_learnable_hare(self, profile_embs):
        """Build LearnableHARE with trained attention weights.

        Reads d_knowledge from the checkpoint config so weight shapes always
        match.  If the per-sample embeddings have fewer dimensions (small
        profiles → fewer SVD components), they are zero-padded to match.

        Returns
        -------
        hare : LearnableHARE
        profile_embs : ndarray (possibly padded)
        ckpt_d : int  — checkpoint's d_knowledge (callers pad query_emb too)
        """
        import torch as _torch
        from hare.bandits.learnable_hare import LearnableHARE

        checkpoint = _torch.load(self.attention_checkpoint, weights_only=False)
        cfg = checkpoint.get("config", {})

        actual_d = profile_embs.shape[1]
        ckpt_d = cfg.get("d_knowledge", actual_d)
        n_profile = profile_embs.shape[0]

        # Zero-pad embeddings when the per-sample SVD produced fewer dims
        if actual_d < ckpt_d:
            profile_embs = np.pad(
                profile_embs, ((0, 0), (0, ckpt_d - actual_d)), mode="constant"
            )

        hare = LearnableHARE(
            d_knowledge=ckpt_d,
            d_user=cfg.get("d_user", min(32, ckpt_d)),
            n_clusters=min(cfg.get("n_clusters", 3), n_profile),
            n_heads=cfg.get("n_heads", 2),
            d_k=cfg.get("d_k", min(32, ckpt_d)),
            d_v=cfg.get("d_v", min(32, ckpt_d)),
            alpha=cfg.get("alpha", 1.5),
            seed=42,
        )
        # Load trained attention weights
        state_dict = checkpoint["attention_state_dict"]
        hare.attention.load_state_dict(state_dict, strict=False)
        hare.set_knowledge_pool(profile_embs)
        return hare, profile_embs, ckpt_d

    def predict(self, input_text: str, profile: list[dict]) -> str:
        import torch
        from hare.bandits.attentive_bandit import HARE

        self._load()

        cfg = self.task_config
        content = _strip_prefix(input_text, cfg)

        if not profile or len(profile) < 2:
            fallback = RAGGPT2(
                self.model_name, self.max_new_tokens, self.top_k,
                task_config=cfg,
            )
            return fallback.predict(input_text, profile)

        # Build profile texts for embedding
        profile_texts = []
        for item in profile:
            text = item.get(cfg.profile_text_key, "")
            title = item.get("title", "")
            profile_texts.append(f"{title}: {text}" if title else text)

        # Fit embedder on this user's profile + the query
        embedder = self._get_embedder()
        all_texts = profile_texts + [content]
        embedder.fit(all_texts)
        profile_embs = embedder.encode(profile_texts)
        query_emb = embedder.encode([content])[0]
        d = profile_embs.shape[1]

        # Initialize HARE for this user's profile
        if self.attention_checkpoint:
            hare, profile_embs, ckpt_d = self._build_learnable_hare(profile_embs)
            # Pad query embedding to match checkpoint dimension
            if d < ckpt_d:
                query_emb = np.pad(query_emb, (0, ckpt_d - d))
                d = ckpt_d
        else:
            hare = HARE(
                d_knowledge=d,
                d_user=min(32, d),
                n_clusters=min(3, len(profile)),
                n_heads=2,
                d_k=min(32, d),
                d_v=min(32, d),
                alpha=1.5,
                seed=42,
            )
            hare.set_knowledge_pool(profile_embs)

        # Warm up user state with profile interactions
        n_warmup = min(self.n_warmup, len(profile))
        for i in range(n_warmup):
            result = hare.recommend(
                profile_embs[i], user_id="user", return_details=True
            )
            hare.update(profile_embs[i], "user", reward=0.8, synthesis=result["synthesis"])

        # Now recommend for the actual query (user-conditioned)
        result = hare.recommend(query_emb, user_id="user", return_details=True)

        # Get top-k by HARE attention weights (user-conditioned retrieval)
        mean_weights = np.mean(result["attention_weights"], axis=0)
        top_k = min(self.top_k, len(profile))
        top_indices = np.argsort(mean_weights)[-top_k:][::-1]
        examples = [profile[i] for i in top_indices]

        # Generate with retrieved context
        prompt_parts = []
        for ex in examples:
            ex_text = ex.get(cfg.profile_text_key, "")[:cfg.max_example_chars]
            ex_target = _get_profile_target(ex, cfg)
            prompt_parts.append(
                f"{cfg.content_label}: {ex_text}\n{cfg.target_label}: {ex_target}"
            )

        prompt_parts.append(
            f"{cfg.content_label}: {content[:cfg.max_input_chars - 100]}\n{cfg.target_label}:"
        )
        prompt = "\n\n".join(prompt_parts)

        inputs = self._tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=480
        )
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        text = self._tokenizer.decode(generated, skip_special_tokens=True)
        return text.strip().split("\n")[0].strip()


# =============================================================================
# Convenience: run all baselines
# =============================================================================

def get_all_baselines(
    include_neural: bool = True,
    checkpoint: str | None = None,
    task: str = "lamp4",
    attention_checkpoint: str | None = None,
) -> list:
    """Get instances of all baselines for a given LaMP task.

    Parameters
    ----------
    include_neural : bool
        If False, skip neural baselines (faster, no GPU needed).
    checkpoint : str or None
        Path to a fine-tuned model checkpoint. If provided, all neural
        baselines use this checkpoint instead of the pretrained model.
    task : str
        LaMP task name (e.g. "lamp4", "lamp5", "lamp7").
    attention_checkpoint : str or None
        Path to trained attention weights. If provided, HareGPT2 uses
        LearnableHARE with these weights instead of random projections.

    Returns
    -------
    list of baseline instances
    """
    cfg = get_task_config(task)
    baselines = [
        # Tier 1: Naive
        RandomProfile(task_config=cfg),
        MostRecent(task_config=cfg),
        InputCopy(task_config=cfg),
        # Tier 2: Classical ML
        TfidfRetrieval(task_config=cfg),
        BM25Retrieval(task_config=cfg),
    ]
    if include_neural:
        baselines.extend([
            # Tier 3: Neural
            VanillaGPT2(checkpoint=checkpoint, task_config=cfg),
            RAGGPT2(checkpoint=checkpoint, task_config=cfg),
            HareGPT2(
                checkpoint=checkpoint, task_config=cfg,
                attention_checkpoint=attention_checkpoint,
            ),
        ])
    return baselines
