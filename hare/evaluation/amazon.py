"""Amazon Reviews 2023 data loading for HARE evaluation.

Loads the Amazon Reviews 2023 dataset (McAuley Lab, UCSD) from HuggingFace
and formats it for personalized review generation evaluation.

Task: Given a product and a user's review history, generate a review
in the user's writing style.

Input:  Product title (+ optional description)
Profile: User's past reviews [{text, title, rating, asin}, ...]
Target: The user's actual review text for the input product

This is a standard personalized generation task on the most widely-cited
recommendation dataset in the field.

Reference:
    Hou et al., "Bridging Language and Items for Retrieval and Recommendation",
    arXiv 2403.03952, 2024. (Amazon Reviews 2023)

Data source: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023

Usage:
    from hare.evaluation.amazon import load_amazon_reviews
    data = load_amazon_reviews(category="Digital_Music", max_samples=100)
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "amazon"

# Available categories (small/medium size, suitable for experiments)
CATEGORIES = {
    "Digital_Music": "Digital Music reviews (~130K reviews, ~1.5K active users)",
    "Musical_Instruments": "Musical Instruments (~300K reviews)",
    "CDs_and_Vinyl": "CDs and Vinyl (~1.8M reviews)",
    "Video_Games": "Video Games (~900K reviews)",
    "Software": "Software (~80K reviews)",
}


@dataclass
class AmazonSample:
    """A single Amazon evaluation sample."""
    id: str
    user_id: str
    input_text: str         # "Write a review for: [product title]"
    target: str             # actual review text
    target_rating: float    # actual rating (1-5)
    product_asin: str
    product_title: str
    profile: list[dict]     # user's past reviews


@dataclass
class AmazonDataset:
    """A loaded Amazon Reviews dataset."""
    category: str
    samples: list[AmazonSample]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> AmazonSample:
        return self.samples[idx]


def _download_hf_file(repo_path: str, cache_path: Path) -> Path:
    """Download a file from HuggingFace dataset repo if not cached."""
    if cache_path.exists():
        return cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download
    downloaded = hf_hub_download(
        repo_id="McAuley-Lab/Amazon-Reviews-2023",
        filename=repo_path,
        repo_type="dataset",
    )
    # Copy to our cache location for persistence
    import shutil
    shutil.copy2(downloaded, cache_path)
    return cache_path


def load_amazon_reviews(
    category: str = "Digital_Music",
    min_reviews_per_user: int = 5,
    max_samples: int | None = None,
    max_profile_size: int = 50,
    seed: int = 42,
) -> AmazonDataset:
    """Load Amazon Reviews for personalized review generation evaluation.

    For each qualifying user (>=min_reviews_per_user reviews):
    - Hold out the most recent review as the test target
    - Use all prior reviews as the user's profile
    - Input is the product title of the target review

    Parameters
    ----------
    category : str
        Amazon product category (e.g. "Digital_Music", "Video_Games").
    min_reviews_per_user : int
        Minimum reviews a user must have to be included.
    max_samples : int or None
        Limit number of evaluation samples.
    max_profile_size : int
        Maximum profile items per user (most recent kept).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    AmazonDataset
    """
    import numpy as np

    rng = np.random.default_rng(seed)

    # Download review data
    review_file = f"raw/review_categories/{category}.jsonl"
    review_cache = CACHE_DIR / category / "reviews.jsonl"
    print(f"  Loading {category} reviews...")
    review_path = _download_hf_file(review_file, review_cache)

    # Download product metadata
    meta_file = f"raw/meta_categories/meta_{category}.jsonl"
    meta_cache = CACHE_DIR / category / "meta.jsonl"
    print(f"  Loading {category} metadata...")
    meta_path = _download_hf_file(meta_file, meta_cache)

    # Load product metadata (asin -> title)
    product_titles: dict[str, str] = {}
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            asin = item.get("parent_asin", "")
            title = item.get("title", "")
            if asin and title:
                product_titles[asin] = title

    print(f"  Loaded {len(product_titles):,} product titles")

    # Group reviews by user, sorted by timestamp
    user_reviews: dict[str, list[dict]] = defaultdict(list)
    n_total = 0
    with open(review_path, "r", encoding="utf-8") as f:
        for line in f:
            review = json.loads(line)
            user_id = review.get("user_id", "")
            if not user_id or not review.get("text", "").strip():
                continue
            user_reviews[user_id].append(review)
            n_total += 1

    print(f"  {n_total:,} reviews from {len(user_reviews):,} users")

    # Sort each user's reviews by timestamp
    for uid in user_reviews:
        user_reviews[uid].sort(key=lambda r: r.get("timestamp", 0))

    # Filter users with enough reviews
    qualified_users = [
        uid for uid, reviews in user_reviews.items()
        if len(reviews) >= min_reviews_per_user
    ]
    print(f"  {len(qualified_users):,} users with >={min_reviews_per_user} reviews")

    # Shuffle users for reproducibility
    rng.shuffle(qualified_users)

    # Build evaluation samples: last review = target, rest = profile
    samples = []
    for uid in qualified_users:
        reviews = user_reviews[uid]
        target_review = reviews[-1]

        # Build profile from prior reviews
        prior = reviews[:-1]
        if len(prior) > max_profile_size:
            prior = prior[-max_profile_size:]  # keep most recent

        profile = []
        for r in prior:
            asin = r.get("parent_asin", r.get("asin", ""))
            profile.append({
                "text": r.get("text", ""),
                "title": product_titles.get(asin, r.get("title", "")),
                "rating": r.get("rating", 0),
                "asin": asin,
            })

        # Input: product title for the target
        target_asin = target_review.get(
            "parent_asin", target_review.get("asin", "")
        )
        product_title = product_titles.get(
            target_asin, target_review.get("title", "Unknown Product")
        )

        samples.append(AmazonSample(
            id=f"{uid}_{target_asin}",
            user_id=uid,
            input_text=f"Write a review for the following product: {product_title}",
            target=target_review.get("text", ""),
            target_rating=target_review.get("rating", 0),
            product_asin=target_asin,
            product_title=product_title,
            profile=profile,
        ))

        if max_samples and len(samples) >= max_samples:
            break

    avg_profile = (
        sum(len(s.profile) for s in samples) / max(len(samples), 1)
    )
    print(f"  Built {len(samples)} evaluation samples, "
          f"avg profile size: {avg_profile:.1f}")

    return AmazonDataset(category=category, samples=samples)


def amazon_to_lamp_format(data: AmazonDataset):
    """Convert AmazonDataset to LaMPDataset-compatible format.

    This allows reusing all existing baselines without modification.
    """
    from hare.evaluation.lamp import LaMPSample, LaMPDataset

    lamp_samples = []
    for s in data.samples:
        lamp_samples.append(LaMPSample(
            id=s.id,
            input_text=s.input_text,
            target=s.target,
            profile=s.profile,
        ))

    return LaMPDataset(
        task=f"Amazon-{data.category}",
        split="eval",
        samples=lamp_samples,
    )
