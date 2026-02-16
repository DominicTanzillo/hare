"""LongLaMP benchmark data loading for HARE.

LongLaMP extends LaMP to long-form personalized text generation.
We support two freely-available generation tasks:

- LongLaMP-2: Personalized Abstract Generation
  Input: paper title + keywords
  Output: full abstract
  Profile: list of author's prior papers (abstract, title, year)
  Metric: ROUGE-1, ROUGE-L

- LongLaMP-3: Personalized Product Review Writing
  Input: product info + star rating
  Output: full review
  Profile: list of user's prior reviews (reviewText, description, summary)
  Metric: ROUGE-1, ROUGE-L

Data source: https://huggingface.co/datasets/LongLaMP/LongLaMP
Reference: Kumar et al., "LongLaMP: A Benchmark for Personalized
           Long-form Text Generation", arXiv:2407.11016, 2024.

Usage:
    from hare.evaluation.longlamp import load_longlamp
    data = load_longlamp("abstract", split="val", max_samples=100)
    data = load_longlamp("review", split="val", max_samples=100)
"""

from __future__ import annotations

from hare.evaluation.lamp import LaMPDataset, LaMPSample


# HuggingFace dataset config names
_TASK_CONFIGS = {
    "abstract": "abstract_generation_user",
    "review": "product_review_user",
    "topic": "topic_writing_user",
}

# User-ID field per task
_USER_ID_FIELDS = {
    "abstract": "name",
    "review": "reviewerId",
    "topic": "author",
}


def _normalize_abstract_profile(item: dict) -> dict:
    """Normalize LongLaMP-2 profile item to LaMP-compatible format.

    LongLaMP-2 profile items have: abstract, id, title, year.
    We map to: text (=abstract), title, abstract (kept for compat).
    """
    return {
        "text": item.get("abstract", ""),
        "title": item.get("title", ""),
        "abstract": item.get("abstract", ""),
        "year": item.get("year", ""),
        "id": item.get("id", ""),
    }


def _normalize_review_profile(item: dict) -> dict:
    """Normalize LongLaMP-3 profile item to LaMP-compatible format.

    LongLaMP-3 profile items have: description, overall, reviewText, summary.
    We map to: text (=reviewText), title (=summary), rating.
    """
    return {
        "text": item.get("reviewText", ""),
        "title": item.get("summary", ""),
        "description": item.get("description", ""),
        "rating": item.get("overall", ""),
    }


def _normalize_topic_profile(item: dict) -> dict:
    """Normalize LongLaMP-4 profile item to LaMP-compatible format."""
    return {
        "text": item.get("text", item.get("content", "")),
        "title": item.get("title", item.get("subreddit", "")),
    }


# Profile normalizer per task (defined after functions)
_PROFILE_NORMALIZERS = {
    "abstract": _normalize_abstract_profile,
    "review": _normalize_review_profile,
    "topic": _normalize_topic_profile,
}


def load_longlamp(
    task: str = "abstract",
    split: str = "val",
    max_samples: int | None = None,
    setting: str = "user",
) -> LaMPDataset:
    """Load a LongLaMP dataset split.

    Parameters
    ----------
    task : str
        One of "abstract" (LongLaMP-2), "review" (LongLaMP-3),
        "topic" (LongLaMP-4).
    split : str
        One of "train", "val", "test".
    max_samples : int or None
        Limit number of samples for quick experiments.
    setting : str
        Split setting: "user" (disjoint users) or "temporal".

    Returns
    -------
    LaMPDataset
        Compatible with all existing baselines.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "LongLaMP requires the 'datasets' package. "
            "Install with: pip install datasets"
        )

    task_key = task.lower()
    if task_key not in _TASK_CONFIGS:
        raise ValueError(
            f"Unknown LongLaMP task: {task}. "
            f"Choose from: {list(_TASK_CONFIGS.keys())}"
        )

    config_name = _TASK_CONFIGS[task_key]
    if setting == "temporal":
        config_name = config_name.replace("_user", "_temporal")

    split_map = {"dev": "val", "validation": "val"}
    ds_split = split_map.get(split, split)

    user_id_field = _USER_ID_FIELDS[task_key]
    normalizer = _PROFILE_NORMALIZERS[task_key]

    print(f"  Loading LongLaMP {task_key} ({config_name}, {ds_split})...")
    ds = load_dataset("LongLaMP/LongLaMP", name=config_name, split=ds_split)

    samples = []
    for i, row in enumerate(ds):
        if max_samples and len(samples) >= max_samples:
            break

        user_id = str(row.get(user_id_field, f"user_{i}"))
        input_text = row["input"]
        target = row["output"]
        raw_profile = row.get("profile", [])

        # Normalize profile items for baseline compatibility
        profile = [normalizer(item) for item in raw_profile]

        samples.append(LaMPSample(
            id=f"longlamp_{task_key}_{i}",
            input_text=input_text,
            target=target,
            profile=profile,
        ))

    task_name = f"LongLaMP-{task_key}"
    avg_profile = (
        sum(len(s.profile) for s in samples) / max(len(samples), 1)
    )
    print(
        f"  Loaded {task_name}: {len(samples)} samples, "
        f"{avg_profile:.1f} avg profile size"
    )

    return LaMPDataset(task=task_name, split=ds_split, samples=samples)
