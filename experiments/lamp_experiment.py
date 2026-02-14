#!/usr/bin/env python3
"""HARE LaMP-4 Benchmark Evaluation.

Evaluates HARE against 3 tiers of baselines on the LaMP-4 benchmark
(Personalized News Headline Generation):

Tier 1 -- Naive:
    Random Profile, Most Recent, Input Copy

Tier 2 -- Classical ML:
    TF-IDF Retrieval, BM25 Retrieval

Tier 3 -- Neural / Deep Learning:
    Vanilla GPT-2, RAG + GPT-2, HARE + GPT-2

Metrics: ROUGE-1, ROUGE-L (F1)

Usage:
    # Quick test (10 samples, no neural baselines)
    python experiments/lamp_experiment.py --max-samples 10 --skip-neural

    # Full evaluation (all baselines, 100 samples)
    python experiments/lamp_experiment.py --max-samples 100

    # Full dev set
    python experiments/lamp_experiment.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from hare.evaluation.lamp import load_lamp4, evaluate_rouge
from hare.evaluation.baselines import get_all_baselines


def run_experiment(
    max_samples: int | None = None,
    skip_neural: bool = False,
    output_path: Path | None = None,
    seed: int = 42,
) -> dict:
    """Run LaMP-4 evaluation across all baselines.

    Parameters
    ----------
    max_samples : int or None
        Limit number of evaluation samples.
    skip_neural : bool
        If True, skip Tier 3 baselines (faster).
    output_path : Path or None
        Save results JSON to this path.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with results per baseline.
    """
    print("=" * 70)
    print("HARE LaMP-4 Benchmark: Personalized News Headline Generation")
    print("=" * 70)

    # Load data
    print("\nLoading LaMP-4 dev set...")
    data = load_lamp4(split="dev", max_samples=max_samples)
    print(f"  {len(data)} samples loaded\n")

    # Get baselines
    baselines = get_all_baselines(include_neural=not skip_neural)

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
                {"id": data.samples[i].id, "prediction": predictions[i],
                 "reference": references[i]}
                for i in range(min(5, len(predictions)))
            ],
        }

        print(f"  {baseline.name}: ROUGE-1={scores['rouge1']:.4f}  "
              f"ROUGE-L={scores['rougeL']:.4f}  ({elapsed:.1f}s)")

    # Print summary table
    print(f"\n{'=' * 70}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n{'Tier':<8s} {'Method':<22s} {'ROUGE-1':<10s} {'ROUGE-L':<10s} {'Time':<8s}")
    print("-" * 58)

    for name, result in sorted(all_results.items(), key=lambda x: x[1]["tier"]):
        tier = result["tier"]
        r1 = result["scores"]["rouge1"]
        rl = result["scores"]["rougeL"]
        t = result["scores"]["time_seconds"]
        print(f"{tier:<8s} {name:<22s} {r1:<10.4f} {rl:<10.4f} {t:<8.1f}s")

    # Highlight best per tier
    print(f"\n{'=' * 70}")
    print("TIER COMPARISON")
    print(f"{'=' * 70}")
    for tier_name in ["Tier 1", "Tier 2", "Tier 3"]:
        tier_results = {k: v for k, v in all_results.items() if v["tier"] == tier_name}
        if tier_results:
            best = max(tier_results.items(), key=lambda x: x[1]["scores"]["rouge1"])
            print(f"  Best {tier_name}: {best[0]} "
                  f"(ROUGE-1={best[1]['scores']['rouge1']:.4f})")

    # Save results
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({
                "experiment": "lamp4_evaluation",
                "n_samples": len(data),
                "results": all_results,
            }, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return all_results


def _get_tier(baseline) -> str:
    """Determine which tier a baseline belongs to."""
    from hare.evaluation.baselines import (
        RandomProfile, MostRecent, InputCopy,
        TfidfRetrieval, BM25Retrieval,
        VanillaGPT2, RAGGPT2, HareGPT2,
    )
    if isinstance(baseline, (RandomProfile, MostRecent, InputCopy)):
        return "Tier 1"
    elif isinstance(baseline, (TfidfRetrieval, BM25Retrieval)):
        return "Tier 2"
    else:
        return "Tier 3"


def main():
    parser = argparse.ArgumentParser(
        description="HARE LaMP-4 Benchmark Evaluation"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit evaluation samples (default: full dev set)."
    )
    parser.add_argument(
        "--skip-neural", action="store_true",
        help="Skip Tier 3 neural baselines (faster)."
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/lamp4_results.json"),
        help="Output path for results JSON."
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_experiment(
        max_samples=args.max_samples,
        skip_neural=args.skip_neural,
        output_path=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
