#!/usr/bin/env python3
"""Push evaluation results to HuggingFace dataset repo.

Usage:
    python scripts/push_results_to_hf.py results/lamp4_results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi


DATASET_REPO = "DTanzillo/hare-lamp4-eval"


def push_results(results_path: Path) -> None:
    api = HfApi()

    with open(results_path) as f:
        data = json.load(f)

    # Upload the results JSON
    filename = results_path.name
    api.upload_file(
        path_or_fileobj=str(results_path),
        path_in_repo=f"results/{filename}",
        repo_id=DATASET_REPO,
        repo_type="dataset",
        commit_message=f"Upload evaluation results: {filename}",
    )
    print(f"Uploaded {filename} to {DATASET_REPO}")

    # Print summary
    n_samples = data.get("n_samples", "?")
    print(f"\nResults summary ({n_samples} samples):")
    for name, result in sorted(
        data.get("results", {}).items(),
        key=lambda x: x[1].get("tier", ""),
    ):
        scores = result.get("scores", {})
        print(f"  {result.get('tier', '?')}: {name} — "
              f"ROUGE-1={scores.get('rouge1', 0):.4f}, "
              f"ROUGE-L={scores.get('rougeL', 0):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Push results to HuggingFace")
    parser.add_argument("results", type=Path, help="Path to results JSON file")
    args = parser.parse_args()

    if not args.results.exists():
        print(f"Error: {args.results} not found")
        return

    push_results(args.results)


if __name__ == "__main__":
    main()
