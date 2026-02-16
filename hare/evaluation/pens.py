"""PENS benchmark data loading for HARE.

PENS (PErsonalized News headlineS) evaluates personalized headline generation
using click-behavior profiles from Microsoft News (445K+ real users).

Unlike LaMP-4 (which uses past article-headline pairs as profiles), PENS
profiles are click histories: lists of news articles the user previously
read. This tests whether HARE can build user states from behavioral signals
rather than explicit input-output pairs.

Task: Given a news article + user's reading history, generate a personalized
      headline that matches the user's interests and style.

Data source: https://msnews.github.io/pens.html
Reference: Ao et al., "PENS: A Dataset and Generic Framework for
           Personalized News Headline Generation", ACL 2021.

Usage:
    from hare.evaluation.pens import load_pens
    data = load_pens(split="val", max_samples=100)
"""

from __future__ import annotations

import csv
import tarfile
import urllib.request
from pathlib import Path

from hare.evaluation.lamp import LaMPDataset, LaMPSample

PENS_URL = "https://mind201910small.blob.core.windows.net/release/PENS.tar.gz"
CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "pens"

_DOWNLOAD_INSTRUCTIONS = """
PENS dataset not found. The original Azure download URL is no longer public.

To obtain the dataset, use one of these methods:

1. Kaggle: https://www.kaggle.com/datasets/divyapatel4/microsoft-pens-personalized-news-headlines
   Download and extract to: {cache_dir}/extracted/

2. If you have the PENS.tar.gz file, place it at: {cache_dir}/PENS.tar.gz

Required files in extracted directory:
  - news.tsv (news article corpus)
  - train.tsv (training impression logs)
  - valid.tsv (validation impression logs)
  - personalized_test.tsv (test set with human-written headlines)
"""


def _download_pens(cache_dir: Path = CACHE_DIR) -> Path:
    """Download and extract PENS dataset if not cached."""
    extract_dir = cache_dir / "extracted"
    if extract_dir.exists() and any(extract_dir.rglob("*.tsv")):
        return extract_dir

    tarball = cache_dir / "PENS.tar.gz"
    if not tarball.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Attempting download from {PENS_URL}...")
        try:
            urllib.request.urlretrieve(PENS_URL, tarball)
        except Exception as e:
            raise FileNotFoundError(
                _DOWNLOAD_INSTRUCTIONS.format(cache_dir=cache_dir)
            ) from e

    print("  Extracting PENS archive...")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball, "r:gz") as tar:
        tar.extractall(extract_dir, filter="data")

    return extract_dir


def _find_tsv_files(extract_dir: Path) -> dict[str, Path]:
    """Locate TSV files in the extracted archive."""
    files = {}
    for tsv in extract_dir.rglob("*.tsv"):
        name = tsv.stem.lower()
        if "news" in name:
            files["news"] = tsv
        elif "train" in name:
            files["train"] = tsv
        elif "valid" in name:
            files["valid"] = tsv
        elif "personalized_test" in name or "test" in name:
            files["test"] = tsv
    return files


def _load_news_corpus(news_path: Path) -> dict[str, dict]:
    """Load news articles from news.tsv.

    Returns dict mapping news_id -> {headline, body, category, topic}.
    """
    news = {}
    with open(news_path, encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 5:
                continue
            news_id = row[0].strip()
            if not news_id or news_id.lower() == "news_id":
                continue
            news[news_id] = {
                "news_id": news_id,
                "category": row[1].strip() if len(row) > 1 else "",
                "topic": row[2].strip() if len(row) > 2 else "",
                "headline": row[3].strip() if len(row) > 3 else "",
                "body": row[4].strip() if len(row) > 4 else "",
            }
    return news


def _build_profile_from_clicks(
    click_ids: list[str],
    news_corpus: dict[str, dict],
    max_profile: int = 50,
) -> list[dict]:
    """Convert click history IDs into profile items for baselines.

    Each profile item has 'text' (article body) and 'title' (headline),
    matching the LaMP-4 profile format so all baselines work.
    """
    profile = []
    for nid in click_ids[:max_profile]:
        nid = nid.strip()
        if nid in news_corpus:
            article = news_corpus[nid]
            profile.append({
                "text": article["body"],
                "title": article["headline"],
                "category": article["category"],
            })
    return profile


def load_pens(
    split: str = "val",
    max_samples: int | None = None,
    cache_dir: Path = CACHE_DIR,
) -> LaMPDataset:
    """Load the PENS dataset.

    Parameters
    ----------
    split : str
        One of "train", "val", "test".
        - train/val: impression logs with click histories + positive articles.
          Target = original headline of the positive article.
        - test: personalized_test.tsv with human-written personalized headlines.
    max_samples : int or None
        Limit number of samples.
    cache_dir : Path
        Where to cache downloaded data.

    Returns
    -------
    LaMPDataset
        Compatible with all existing baselines. Profile items have 'text'
        (article body) and 'title' (headline) from the user's click history.
    """
    extract_dir = _download_pens(cache_dir)
    files = _find_tsv_files(extract_dir)

    if "news" not in files:
        raise FileNotFoundError(
            f"news.tsv not found in {extract_dir}. "
            "Archive may have unexpected structure."
        )

    print("  Loading PENS news corpus...")
    news_corpus = _load_news_corpus(files["news"])
    print(f"  {len(news_corpus)} news articles loaded")

    split_map = {"val": "valid", "dev": "valid"}
    split_key = split_map.get(split, split)

    if split_key == "test":
        return _load_pens_test(files, news_corpus, max_samples)
    else:
        return _load_pens_behavior(
            files, news_corpus, split_key, max_samples
        )


def _load_pens_behavior(
    files: dict[str, Path],
    news_corpus: dict[str, dict],
    split: str,
    max_samples: int | None,
) -> LaMPDataset:
    """Load train/val split from behavior logs.

    Each impression that has a positive click becomes a sample:
    - Profile: articles from user's click history
    - Input: body of the clicked article
    - Target: headline of the clicked article
    """
    if split not in files:
        raise FileNotFoundError(
            f"{split}.tsv not found. Available: {list(files.keys())}"
        )

    print(f"  Loading PENS {split} impressions...")
    samples = []

    with open(files[split], encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        for row_idx, row in enumerate(reader):
            if max_samples and len(samples) >= max_samples:
                break
            if len(row) < 5:
                continue

            user_id = row[0].strip()
            if not user_id or user_id.lower() == "userid":
                continue

            # Click history: space-separated news IDs
            click_ids = row[1].strip().split() if row[1].strip() else []
            # Positive articles in this impression: space-separated
            pos_ids = row[4].strip().split() if row[4].strip() else []

            if not pos_ids or not click_ids:
                continue

            profile = _build_profile_from_clicks(click_ids, news_corpus)
            if len(profile) < 2:
                continue

            # Use the first positive article as the target
            target_id = pos_ids[0].strip()
            if target_id not in news_corpus:
                continue

            target_article = news_corpus[target_id]
            body = target_article["body"]
            headline = target_article["headline"]

            if not body.strip() or not headline.strip():
                continue

            input_text = (
                f"Generate a headline for the following article: {body}"
            )

            samples.append(LaMPSample(
                id=f"pens_{split}_{row_idx}",
                input_text=input_text,
                target=headline,
                profile=profile,
            ))

    avg_profile = (
        sum(len(s.profile) for s in samples) / max(len(samples), 1)
    )
    print(
        f"  Loaded PENS {split}: {len(samples)} samples, "
        f"{avg_profile:.1f} avg profile size"
    )
    return LaMPDataset(task="PENS", split=split, samples=samples)


def _load_pens_test(
    files: dict[str, Path],
    news_corpus: dict[str, dict],
    max_samples: int | None,
) -> LaMPDataset:
    """Load the personalized test set with human-written headlines.

    Each test row has a user with click history, target article(s),
    and human-written personalized headlines (separated by ;;).
    """
    if "test" not in files:
        raise FileNotFoundError(
            "personalized_test.tsv not found in extracted files."
        )

    print("  Loading PENS personalized test set...")
    samples = []

    with open(files["test"], encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        for row_idx, row in enumerate(reader):
            if max_samples and len(samples) >= max_samples:
                break
            if len(row) < 4:
                continue

            user_id = row[0].strip()
            if not user_id or user_id.lower() == "userid":
                continue

            # Test file uses commas as separator (not spaces)
            click_ids = row[1].strip().split(",") if row[1].strip() else []
            pos_ids = row[2].strip().split(",") if row[2].strip() else []
            rewrite_titles = row[3].strip().split(";;") if row[3].strip() else []

            if not pos_ids or not click_ids or not rewrite_titles:
                continue

            profile = _build_profile_from_clicks(click_ids, news_corpus)
            if len(profile) < 2:
                continue

            # Create one sample per (article, personalized_headline) pair
            for i, target_id in enumerate(pos_ids):
                if max_samples and len(samples) >= max_samples:
                    break

                target_id = target_id.strip()
                if target_id not in news_corpus:
                    continue

                target_article = news_corpus[target_id]
                body = target_article["body"]

                # Use the corresponding personalized headline
                if i < len(rewrite_titles):
                    headline = rewrite_titles[i].strip()
                else:
                    continue

                if not body.strip() or not headline:
                    continue

                input_text = (
                    f"Generate a headline for the following article: {body}"
                )

                samples.append(LaMPSample(
                    id=f"pens_test_{user_id}_{i}",
                    input_text=input_text,
                    target=headline,
                    profile=profile,
                ))

    avg_profile = (
        sum(len(s.profile) for s in samples) / max(len(samples), 1)
    )
    print(
        f"  Loaded PENS test: {len(samples)} samples, "
        f"{avg_profile:.1f} avg profile size"
    )
    return LaMPDataset(task="PENS", split="test", samples=samples)
