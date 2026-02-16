#!/usr/bin/env python3
"""HARE PENS Benchmark Evaluation.

Evaluates HARE on personalized headline generation using the PENS dataset
(Ao et al., ACL 2021). Unlike LaMP-4 where profiles are text pairs, PENS
profiles are click-behavior logs from 445K+ real Microsoft News users.

This demonstrates HARE's generality across profile representations:
- LaMP-4: profile = past (article, headline) pairs
- PENS: profile = past articles the user clicked on

Three-tier evaluation:
  Tier 1 -- Naive: Random Profile, Most Recent, Input Copy
  Tier 2 -- Classical ML: TF-IDF Retrieval, BM25 Retrieval
  Tier 3 -- Neural: Vanilla GPT-2, RAG + GPT-2, HARE + GPT-2

Metrics: ROUGE-1, ROUGE-L (F1)

Usage:
    # Quick test (10 samples, no neural)
    python experiments/pens_experiment.py --max-samples 10 --skip-neural

    # Full evaluation
    python experiments/pens_experiment.py --max-samples 100

    # With learnable attention
    python experiments/pens_experiment.py --max-samples 100 \\
        --attention-checkpoint checkpoints/attention_weights.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from hare.evaluation.baselines import get_all_baselines
from hare.evaluation.lamp import evaluate_rouge
from hare.evaluation.pens import load_pens


def run_pens_experiment(
    split: str = "val",
    max_samples: int | None = None,
    skip_neural: bool = False,
    output_path: Path | None = None,
    checkpoint: str | None = None,
    attention_checkpoint: str | None = None,
) -> dict:
    """Run PENS evaluation across all baselines."""
    print("=" * 70)
    print("HARE PENS Benchmark -- Personalized News Headlines")
    print("=" * 70)

    print(f"\nLoading PENS ({split})...")
    data = load_pens(split=split, max_samples=max_samples)
    print(f"  {len(data)} samples ready for evaluation\n")

    baselines = get_all_baselines(
        include_neural=not skip_neural,
        checkpoint=checkpoint,
        task="pens",
        attention_checkpoint=attention_checkpoint,
    )

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

        print(
            f"  {baseline.name}: ROUGE-1={scores['rouge1']:.4f}  "
            f"ROUGE-L={scores['rougeL']:.4f}  ({elapsed:.1f}s)"
        )

    # Summary
    print(f"\n{'=' * 70}")
    print("RESULTS SUMMARY -- PENS (Personalized News Headlines)")
    print(f"{'=' * 70}")
    print(f"\n{'Method':<22s} {'ROUGE-1':<10s} {'ROUGE-L':<10s} {'Time':<8s}")
    print("-" * 50)

    for name, result in all_results.items():
        r1 = result["scores"]["rouge1"]
        rl = result["scores"]["rougeL"]
        t = result["scores"]["time_seconds"]
        print(f"{name:<22s} {r1:<10.4f} {rl:<10.4f} {t:<8.1f}s")

    # Cross-benchmark comparison
    print(f"\n{'=' * 70}")
    print("CROSS-BENCHMARK VALIDATION")
    print(f"{'=' * 70}")
    print("  HARE evaluated across different profile representations:")
    print("  - LaMP-4: text pairs (article, headline)")
    print("  - PENS: click-behavior logs (articles user read)")
    print("  - Amazon: review histories")
    print("  Same architecture, no changes between domains.")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({
                "experiment": "pens_evaluation",
                "n_samples": len(data),
                "split": split,
                "results": all_results,
            }, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="HARE PENS Benchmark Evaluation"
    )
    parser.add_argument(
        "--split", type=str, default="val",
        choices=["train", "val", "test"],
        help="Dataset split (default: val).",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--skip-neural", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--attention-checkpoint", type=str, default=None)
    args = parser.parse_args()

    output = args.output or Path("results/pens_results.json")
    run_pens_experiment(
        split=args.split,
        max_samples=args.max_samples,
        skip_neural=args.skip_neural,
        output_path=output,
        checkpoint=args.checkpoint,
        attention_checkpoint=args.attention_checkpoint,
    )


if __name__ == "__main__":
    main()
