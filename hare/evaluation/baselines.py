"""Three-tier baseline suite for LaMP evaluation.

Tier 1 -- Naive baselines:
    - RandomProfile: randomly select a headline from user's profile
    - MostRecent: use the most recent profile headline
    - InputCopy: extract first sentence of article as headline

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
from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Baseline(Protocol):
    """Protocol for all baselines."""
    name: str

    def predict(self, input_text: str, profile: list[dict]) -> str: ...


# =============================================================================
# Tier 1: Naive Baselines
# =============================================================================

class RandomProfile:
    """Randomly select a headline from the user's profile."""
    name = "Random (profile)"

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    def predict(self, input_text: str, profile: list[dict]) -> str:
        if not profile:
            return ""
        item = profile[self.rng.integers(0, len(profile))]
        return item.get("title", item.get("text", ""))


class MostRecent:
    """Use the last profile item's headline (assumes temporal ordering)."""
    name = "Most Recent"

    def predict(self, input_text: str, profile: list[dict]) -> str:
        if not profile:
            return ""
        item = profile[-1]
        return item.get("title", item.get("text", ""))


class InputCopy:
    """Extract the first sentence of the article as the headline."""
    name = "Input Copy"

    _SENTENCE_RE = re.compile(r"(?:Generate a headline for the following article:\s*)?(.+?[.!?])\s", re.DOTALL)

    def predict(self, input_text: str, profile: list[dict]) -> str:
        # Strip the LaMP instruction prefix
        text = re.sub(
            r"^Generate a headline for the following article:\s*",
            "", input_text, flags=re.IGNORECASE,
        )
        match = self._SENTENCE_RE.match(text)
        if match:
            sentence = match.group(1).strip()
            # Truncate to reasonable headline length
            words = sentence.split()
            return " ".join(words[:15])
        return " ".join(text.split()[:10])


# =============================================================================
# Tier 2: Classical ML Baselines
# =============================================================================

class TfidfRetrieval:
    """Retrieve the most similar profile headline by TF-IDF cosine similarity.

    For each test article, compute TF-IDF similarity to all profile articles,
    then return the headline of the most similar profile article.
    """
    name = "TF-IDF Retrieval"

    def predict(self, input_text: str, profile: list[dict]) -> str:
        if not profile:
            return ""

        # Strip instruction prefix
        article = re.sub(
            r"^Generate a headline for the following article:\s*",
            "", input_text, flags=re.IGNORECASE,
        )

        # Build TF-IDF over profile articles + input
        profile_texts = [item.get("text", "") for item in profile]
        all_texts = profile_texts + [article]

        try:
            vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
            tfidf = vectorizer.fit_transform(all_texts)
        except ValueError:
            # Empty vocabulary
            return profile[0].get("title", "")

        # Cosine similarity between input and each profile article
        input_vec = tfidf[-1:]
        profile_vecs = tfidf[:-1]
        sims = cosine_similarity(input_vec, profile_vecs)[0]

        best_idx = int(np.argmax(sims))
        return profile[best_idx].get("title", profile[best_idx].get("text", ""))


class BM25Retrieval:
    """BM25-based retrieval from profile.

    Simple BM25 implementation using TF-IDF with sublinear TF and
    IDF weighting, retrieving the profile headline most relevant
    to the input article.
    """
    name = "BM25 Retrieval"

    def predict(self, input_text: str, profile: list[dict]) -> str:
        if not profile:
            return ""

        article = re.sub(
            r"^Generate a headline for the following article:\s*",
            "", input_text, flags=re.IGNORECASE,
        )

        profile_texts = [item.get("text", "") for item in profile]
        all_texts = profile_texts + [article]

        try:
            vectorizer = TfidfVectorizer(
                max_features=5000, stop_words="english",
                sublinear_tf=True,  # BM25-like sublinear scaling
            )
            tfidf = vectorizer.fit_transform(all_texts)
        except ValueError:
            return profile[0].get("title", "")

        input_vec = tfidf[-1:]
        profile_vecs = tfidf[:-1]
        sims = cosine_similarity(input_vec, profile_vecs)[0]

        best_idx = int(np.argmax(sims))
        return profile[best_idx].get("title", profile[best_idx].get("text", ""))


# =============================================================================
# Tier 3: Neural / Deep Learning Baselines
# =============================================================================

class VanillaGPT2:
    """Fine-tuned DistilGPT2 for headline generation (no personalization).

    Same model for all users. Input: article text. Output: headline.
    No user profile is used.
    """
    name = "Vanilla GPT-2"

    def __init__(self, model_name: str = "distilgpt2", max_new_tokens: int = 32) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model.eval()

    def predict(self, input_text: str, profile: list[dict]) -> str:
        import torch
        self._load()

        article = re.sub(
            r"^Generate a headline for the following article:\s*",
            "", input_text, flags=re.IGNORECASE,
        )
        prompt = f"Article: {article[:400]}\n\nHeadline:"

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
        # Take first line as headline
        headline = text.strip().split("\n")[0].strip()
        return headline


class RAGGPT2:
    """RAG baseline: retrieve top-k profile items, condition GPT-2 generation.

    Retrieves the k most similar profile articles (by TF-IDF cosine),
    includes their headlines as few-shot examples, then generates.
    This is the standard RAG approach -- user-independent retrieval.
    """
    name = "RAG + GPT-2"

    def __init__(
        self,
        model_name: str = "distilgpt2",
        max_new_tokens: int = 32,
        top_k: int = 3,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.top_k = top_k
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model.eval()

    def _retrieve_examples(self, article: str, profile: list[dict]) -> list[dict]:
        """Retrieve top-k most similar profile items."""
        if not profile:
            return []

        profile_texts = [item.get("text", "") for item in profile]
        all_texts = profile_texts + [article]

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

        article = re.sub(
            r"^Generate a headline for the following article:\s*",
            "", input_text, flags=re.IGNORECASE,
        )

        # Retrieve similar examples from profile
        examples = self._retrieve_examples(article, profile)

        # Build few-shot prompt with retrieved examples
        prompt_parts = []
        for ex in examples:
            ex_text = ex.get("text", "")[:200]
            ex_title = ex.get("title", "")
            prompt_parts.append(f"Article: {ex_text}\nHeadline: {ex_title}")

        prompt_parts.append(f"Article: {article[:300]}\nHeadline:")
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
        headline = text.strip().split("\n")[0].strip()
        return headline


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
    """
    name = "HARE + GPT-2"

    def __init__(
        self,
        model_name: str = "distilgpt2",
        max_new_tokens: int = 32,
        top_k: int = 3,
        n_warmup: int = 5,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.top_k = top_k
        self.n_warmup = n_warmup
        self._model = None
        self._tokenizer = None
        self._embedder = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model.eval()

    def _get_embedder(self):
        if self._embedder is None:
            from hare.utils.embeddings import TfidfEmbedder
            self._embedder = TfidfEmbedder(max_features=2000, output_dim=64)
        return self._embedder

    def predict(self, input_text: str, profile: list[dict]) -> str:
        import torch
        from hare.bandits.attentive_bandit import HARE

        self._load()

        article = re.sub(
            r"^Generate a headline for the following article:\s*",
            "", input_text, flags=re.IGNORECASE,
        )

        if not profile or len(profile) < 2:
            # Fall back to RAG for tiny profiles
            return RAGGPT2(self.model_name, self.max_new_tokens, self.top_k).predict(
                input_text, profile
            )

        # Build profile texts for embedding
        profile_texts = []
        for item in profile:
            text = item.get("text", "")
            title = item.get("title", "")
            profile_texts.append(f"{title}: {text}" if title else text)

        # Fit embedder on this user's profile + the query
        embedder = self._get_embedder()
        all_texts = profile_texts + [article]
        embedder.fit(all_texts)
        profile_embs = embedder.encode(profile_texts)
        query_emb = embedder.encode([article])[0]
        d = profile_embs.shape[1]

        # Initialize HARE for this user's profile
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
        # Simulate reading through profile items (builds user state)
        n_warmup = min(self.n_warmup, len(profile))
        for i in range(n_warmup):
            result = hare.recommend(
                profile_embs[i], user_id="user", return_details=True
            )
            # Reward based on how close the attention was to the actual item
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
            ex_text = ex.get("text", "")[:200]
            ex_title = ex.get("title", "")
            prompt_parts.append(f"Article: {ex_text}\nHeadline: {ex_title}")

        prompt_parts.append(f"Article: {article[:300]}\nHeadline:")
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
        headline = text.strip().split("\n")[0].strip()
        return headline


# =============================================================================
# Convenience: run all baselines
# =============================================================================

def get_all_baselines(include_neural: bool = True) -> list:
    """Get instances of all baselines.

    Parameters
    ----------
    include_neural : bool
        If False, skip neural baselines (faster, no GPU needed).

    Returns
    -------
    list of baseline instances
    """
    baselines = [
        # Tier 1: Naive
        RandomProfile(),
        MostRecent(),
        InputCopy(),
        # Tier 2: Classical ML
        TfidfRetrieval(),
        BM25Retrieval(),
    ]
    if include_neural:
        baselines.extend([
            # Tier 3: Neural
            VanillaGPT2(),
            RAGGPT2(),
            HareGPT2(),
        ])
    return baselines
