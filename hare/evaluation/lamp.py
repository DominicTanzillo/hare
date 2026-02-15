"""LaMP benchmark data loading and evaluation for HARE.

LaMP (Language Model Personalization) provides 7 tasks for evaluating
personalized text generation. We support three generation tasks:

- LaMP-4: Personalized News Headline Generation
  Input: article text
  Output: headline
  Profile: list of (article_text, headline) pairs from user's history
  Metric: ROUGE-1, ROUGE-L

- LaMP-5: Personalized Scholarly Title Generation
  Input: paper abstract
  Output: paper title
  Profile: list of (abstract, title) pairs from author's history
  Metric: ROUGE-1, ROUGE-L

- LaMP-7: Personalized Tweet Paraphrasing
  Input: tweet
  Output: paraphrased tweet in user's style
  Profile: list of past tweets
  Metric: ROUGE-1, ROUGE-L

Data source: https://lamp-benchmark.github.io/
Reference: Salemi et al., "LaMP: When Large Language Models Meet
           Personalization", ACL 2024.

Usage:
    from hare.evaluation.lamp import load_lamp, evaluate_rouge
    data = load_lamp("lamp4", split="dev", max_samples=100)
    scores = evaluate_rouge(predictions, references)
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

LAMP_BASE_URL = "https://ciir.cs.umass.edu/downloads/LaMP"
CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "lamp"


@dataclass
class LaMPSample:
    """A single LaMP evaluation sample."""
    id: str
    input_text: str
    target: str
    profile: list[dict]  # e.g. {"text": ..., "title": ...} for LaMP-4


@dataclass
class LaMPDataset:
    """A loaded LaMP dataset split."""
    task: str
    split: str
    samples: list[LaMPSample]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> LaMPSample:
        return self.samples[idx]

    def profile_texts(self, idx: int) -> list[str]:
        """Get profile texts for a sample (for embedding)."""
        sample = self.samples[idx]
        texts = []
        for item in sample.profile:
            if "title" in item and "text" in item:
                texts.append(f"{item['title']}: {item['text']}")
            elif "title" in item and "abstract" in item:
                texts.append(f"{item['title']}: {item['abstract']}")
            elif "abstract" in item:
                texts.append(item["abstract"])
            elif "text" in item:
                texts.append(item["text"])
        return texts

    def profile_targets(self, idx: int) -> list[str]:
        """Get profile target texts (titles/headlines) for a sample."""
        sample = self.samples[idx]
        targets = []
        for item in sample.profile:
            if "title" in item:
                targets.append(item["title"])
            elif "text" in item:
                targets.append(item["text"])
        return targets


def _download_file(url: str, cache_path: Path) -> Path:
    """Download a file if not cached."""
    if cache_path.exists():
        return cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {url}...")
    urllib.request.urlretrieve(url, cache_path)
    return cache_path


def load_lamp4(
    split: str = "dev",
    max_samples: int | None = None,
) -> LaMPDataset:
    """Load LaMP-4 (Personalized News Headline Generation).

    Parameters
    ----------
    split : str
        One of "train", "dev", "test".
    max_samples : int or None
        Limit number of samples (for quick experiments).

    Returns
    -------
    LaMPDataset
    """
    split_map = {"train": "train", "dev": "dev", "val": "dev", "test": "test"}
    s = split_map.get(split, split)

    questions_url = f"{LAMP_BASE_URL}/LaMP_4/{s}/{s}_questions.json"
    questions_path = CACHE_DIR / "LaMP_4" / f"{s}_questions.json"
    _download_file(questions_url, questions_path)

    with open(questions_path, encoding="utf-8") as f:
        questions = json.load(f)

    # Load outputs (not available for test split)
    outputs_map: dict[str, str] = {}
    if s != "test":
        outputs_url = f"{LAMP_BASE_URL}/LaMP_4/{s}/{s}_outputs.json"
        outputs_path = CACHE_DIR / "LaMP_4" / f"{s}_outputs.json"
        _download_file(outputs_url, outputs_path)

        with open(outputs_path, encoding="utf-8") as f:
            outputs_data = json.load(f)
        for gold in outputs_data.get("golds", []):
            outputs_map[gold["id"]] = gold["output"]

    samples = []
    for q in questions:
        sample_id = str(q["id"])
        target = outputs_map.get(sample_id, "")
        samples.append(LaMPSample(
            id=sample_id,
            input_text=q["input"],
            target=target,
            profile=q.get("profile", []),
        ))
        if max_samples and len(samples) >= max_samples:
            break

    print(f"  Loaded LaMP-4 {s}: {len(samples)} samples, "
          f"{sum(len(s.profile) for s in samples) / max(len(samples), 1):.1f} avg profile size")
    return LaMPDataset(task="LaMP-4", split=s, samples=samples)


def load_lamp5(
    split: str = "dev",
    max_samples: int | None = None,
) -> LaMPDataset:
    """Load LaMP-5 (Personalized Scholarly Title Generation).

    Parameters
    ----------
    split : str
        One of "train", "dev", "test".
    max_samples : int or None
        Limit number of samples.

    Returns
    -------
    LaMPDataset
    """
    split_map = {"train": "train", "dev": "dev", "val": "dev", "test": "test"}
    s = split_map.get(split, split)

    questions_url = f"{LAMP_BASE_URL}/LaMP_5/{s}/{s}_questions.json"
    questions_path = CACHE_DIR / "LaMP_5" / f"{s}_questions.json"
    _download_file(questions_url, questions_path)

    with open(questions_path, encoding="utf-8") as f:
        questions = json.load(f)

    outputs_map: dict[str, str] = {}
    if s != "test":
        outputs_url = f"{LAMP_BASE_URL}/LaMP_5/{s}/{s}_outputs.json"
        outputs_path = CACHE_DIR / "LaMP_5" / f"{s}_outputs.json"
        _download_file(outputs_url, outputs_path)

        with open(outputs_path, encoding="utf-8") as f:
            outputs_data = json.load(f)
        for gold in outputs_data.get("golds", []):
            outputs_map[gold["id"]] = gold["output"]

    samples = []
    for q in questions:
        sample_id = str(q["id"])
        target = outputs_map.get(sample_id, "")
        samples.append(LaMPSample(
            id=sample_id,
            input_text=q["input"],
            target=target,
            profile=q.get("profile", []),
        ))
        if max_samples and len(samples) >= max_samples:
            break

    print(f"  Loaded LaMP-5 {s}: {len(samples)} samples, "
          f"{sum(len(s.profile) for s in samples) / max(len(samples), 1):.1f} avg profile size")
    return LaMPDataset(task="LaMP-5", split=s, samples=samples)


def load_lamp7(
    split: str = "dev",
    max_samples: int | None = None,
) -> LaMPDataset:
    """Load LaMP-7 (Personalized Tweet Paraphrasing).

    Parameters
    ----------
    split : str
        One of "train", "dev", "test".
    max_samples : int or None
        Limit number of samples.

    Returns
    -------
    LaMPDataset
    """
    split_map = {"train": "train", "dev": "dev", "val": "dev", "test": "test"}
    s = split_map.get(split, split)

    questions_url = f"{LAMP_BASE_URL}/LaMP_7/{s}/{s}_questions.json"
    questions_path = CACHE_DIR / "LaMP_7" / f"{s}_questions.json"
    _download_file(questions_url, questions_path)

    with open(questions_path, encoding="utf-8") as f:
        questions = json.load(f)

    outputs_map: dict[str, str] = {}
    if s != "test":
        outputs_url = f"{LAMP_BASE_URL}/LaMP_7/{s}/{s}_outputs.json"
        outputs_path = CACHE_DIR / "LaMP_7" / f"{s}_outputs.json"
        _download_file(outputs_url, outputs_path)

        with open(outputs_path, encoding="utf-8") as f:
            outputs_data = json.load(f)
        for gold in outputs_data.get("golds", []):
            outputs_map[gold["id"]] = gold["output"]

    samples = []
    for q in questions:
        sample_id = str(q["id"])
        target = outputs_map.get(sample_id, "")
        samples.append(LaMPSample(
            id=sample_id,
            input_text=q["input"],
            target=target,
            profile=q.get("profile", []),
        ))
        if max_samples and len(samples) >= max_samples:
            break

    print(f"  Loaded LaMP-7 {s}: {len(samples)} samples")
    return LaMPDataset(task="LaMP-7", split=s, samples=samples)


def load_lamp(
    task: str,
    split: str = "dev",
    max_samples: int | None = None,
) -> LaMPDataset:
    """Load a LaMP dataset by task name.

    Parameters
    ----------
    task : str
        One of "lamp4", "lamp5", "lamp7" (case-insensitive, dashes/underscores ok).
    split : str
        One of "train", "dev", "test".
    max_samples : int or None
        Limit number of samples.

    Returns
    -------
    LaMPDataset
    """
    key = task.lower().replace("-", "").replace("_", "")
    loaders = {
        "lamp4": load_lamp4,
        "lamp5": load_lamp5,
        "lamp7": load_lamp7,
    }
    if key not in loaders:
        raise ValueError(f"Unknown task: {task}. Choose from: {list(loaders.keys())}")
    return loaders[key](split=split, max_samples=max_samples)


def evaluate_rouge(
    predictions: Sequence[str],
    references: Sequence[str],
) -> dict[str, float]:
    """Compute ROUGE-1 and ROUGE-L F1 scores.

    Parameters
    ----------
    predictions : list of str
        Generated outputs.
    references : list of str
        Ground truth outputs.

    Returns
    -------
    dict with 'rouge1' and 'rougeL' F1 scores.
    """
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

    r1_scores = []
    rl_scores = []

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        r1_scores.append(scores["rouge1"].fmeasure)
        rl_scores.append(scores["rougeL"].fmeasure)

    return {
        "rouge1": sum(r1_scores) / max(len(r1_scores), 1),
        "rougeL": sum(rl_scores) / max(len(rl_scores), 1),
        "n_samples": len(predictions),
    }
