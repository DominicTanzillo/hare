#!/usr/bin/env python3
"""HARE Amazon Reviews Benchmark Evaluation.

Evaluates HARE on personalized review generation using the Amazon Reviews 2023
dataset (McAuley Lab). This demonstrates HARE as a domain-agnostic architecture:
the same attention mechanism that personalizes news headlines (LaMP-4) also
personalizes product reviews — without architectural changes.

Task: Given a product and the user's review history, generate a review
in the user's writing style.

Three-tier evaluation (same as LaMP):
  Tier 1 — Naive: Random Profile, Most Recent, Input Copy
  Tier 2 — Classical ML: TF-IDF Retrieval, BM25 Retrieval
  Tier 3 — Neural: Vanilla GPT-2, RAG + GPT-2, HARE + GPT-2

Metrics: ROUGE-1, ROUGE-L (F1)

Usage:
    # Quick test (10 samples, no neural)
    python experiments/amazon_experiment.py --max-samples 10 --skip-neural

    # Full evaluation with learnable attention
    python experiments/amazon_experiment.py --max-samples 100 \\
        --attention-checkpoint checkpoints/attention_weights.pt

    # Different category
    python experiments/amazon_experiment.py --category Video_Games --max-samples 50
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from hare.evaluation.amazon import load_amazon_reviews, amazon_to_lamp_format
from hare.evaluation.baselines import get_all_baselines, get_task_config
from hare.evaluation.lamp import evaluate_rouge


def run_amazon_experiment(
    category: str = "Digital_Music",
    max_samples: int | None = None,
    skip_neural: bool = False,
    output_path: Path | None = None,
    seed: int = 42,
    checkpoint: str | None = None,
    attention_checkpoint: str | None = None,
    min_reviews: int = 5,
) -> dict:
    """Run Amazon Reviews evaluation across all baselines.

    Parameters
    ----------
    category : str
        Amazon product category.
    max_samples : int or None
        Limit number of evaluation samples.
    skip_neural : bool
        If True, skip Tier 3 baselines.
    output_path : Path or None
        Save results JSON.
    seed : int
        Random seed.
    checkpoint : str or None
        Path to fine-tuned GPT-2 checkpoint.
    attention_checkpoint : str or None
        Path to trained HARE attention weights.
    min_reviews : int
        Minimum reviews per user to qualify.

    Returns
    -------
    dict with results per baseline.
    """
    print("=" * 70)
    print(f"HARE Amazon Reviews Benchmark — {category}")
    print("=" * 70)

    # Load Amazon data
    print(f"\nLoading Amazon {category}...")
    amazon_data = load_amazon_reviews(
        category=category,
        min_reviews_per_user=min_reviews,
        max_samples=max_samples,
        seed=seed,
    )

    # Convert to LaMP-compatible format for baseline reuse
    data = amazon_to_lamp_format(amazon_data)
    print(f"  {len(data)} samples ready for evaluation\n")

    # Get baselines with Amazon task config
    baselines = get_all_baselines(
        include_neural=not skip_neural,
        checkpoint=checkpoint,
        task="amazon",
        attention_checkpoint=attention_checkpoint,
    )

    # Run evaluation
    all_results = {}
    for baseline in baselines:
        print(f"Running: {baseline.name}...")
        predictions = []
        references = []
        t0 = time.time()

        for i, sample in enumerate(data.samples):
            pred = baseline.predict(sample.input_text, sample.profile)
            predictions.append(pred)
            references.append(sample.target)

            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                print(f"  {i + 1}/{len(data)} ({elapsed:.1f}s)")

        elapsed = time.time() - t0
        scores = evaluate_rouge(predictions, references)
        scores["time_seconds"] = round(elapsed, 2)

        all_results[baseline.name] = {
            "tier": _get_tier(baseline),
            "scores": scores,
            "sample_predictions": [
                {
                    "id": data.samples[i].id,
                    "prediction": predictions[i][:200],
                    "reference": references[i][:200],
                }
                for i in range(min(5, len(predictions)))
            ],
        }

        print(f"  {baseline.name}: ROUGE-1={scores['rouge1']:.4f}  "
              f"ROUGE-L={scores['rougeL']:.4f}  ({elapsed:.1f}s)")

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"RESULTS SUMMARY — Amazon {category}")
    print(f"{'=' * 70}")
    print(f"\n{'Tier':<8s} {'Method':<22s} {'ROUGE-1':<10s} {'ROUGE-L':<10s} {'Time':<8s}")
    print("-" * 58)

    for name, result in sorted(all_results.items(), key=lambda x: x[1]["tier"]):
        tier = result["tier"]
        r1 = result["scores"]["rouge1"]
        rl = result["scores"]["rougeL"]
        t = result["scores"]["time_seconds"]
        print(f"{tier:<8s} {name:<22s} {r1:<10.4f} {rl:<10.4f} {t:<8.1f}s")

    # Cross-domain comparison note
    print(f"\n{'=' * 70}")
    print("CROSS-DOMAIN VALIDATION")
    print(f"{'=' * 70}")
    print("  Same HARE architecture evaluated on two domains:")
    print("  - LaMP-4: Personalized News Headline Generation")
    print(f"  - Amazon {category}: Personalized Review Generation")
    print("  No architectural changes between domains — only task config differs.")
    print("  This demonstrates HARE as a domain-agnostic personalization layer.")

    # Save
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({
                "experiment": f"amazon_{category}_evaluation",
                "category": category,
                "n_samples": len(data),
                "min_reviews_per_user": min_reviews,
                "results": all_results,
            }, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return all_results


def _get_tier(baseline) -> str:
    from hare.evaluation.baselines import (
        RandomProfile, MostRecent, InputCopy,
        TfidfRetrieval, BM25Retrieval,
    )
    if isinstance(baseline, (RandomProfile, MostRecent, InputCopy)):
        return "Tier 1"
    elif isinstance(baseline, (TfidfRetrieval, BM25Retrieval)):
        return "Tier 2"
    else:
        return "Tier 3"


def main():
    parser = argparse.ArgumentParser(
        description="HARE Amazon Reviews Benchmark Evaluation"
    )
    parser.add_argument(
        "--category", type=str, default="Digital_Music",
        help="Amazon product category (default: Digital_Music).",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit evaluation samples.",
    )
    parser.add_argument(
        "--skip-neural", action="store_true",
        help="Skip Tier 3 neural baselines.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output path for results JSON.",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to fine-tuned GPT-2 checkpoint.",
    )
    parser.add_argument(
        "--attention-checkpoint", type=str, default=None,
        help="Path to trained HARE attention weights.",
    )
    parser.add_argument("--min-reviews", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output = args.output or Path(f"results/amazon_{args.category}_results.json")
    run_amazon_experiment(
        category=args.category,
        max_samples=args.max_samples,
        skip_neural=args.skip_neural,
        output_path=output,
        seed=args.seed,
        checkpoint=args.checkpoint,
        attention_checkpoint=args.attention_checkpoint,
        min_reviews=args.min_reviews,
    )


if __name__ == "__main__":
    main()
