#!/usr/bin/env python3
"""HARE LongLaMP Benchmark Evaluation.

Evaluates HARE on long-form personalized text generation using the LongLaMP
benchmark (Kumar et al., 2024). Demonstrates HARE as a domain-agnostic
architecture: the same attention mechanism works on both short-form (LaMP)
and long-form (LongLaMP) generation tasks.

Tasks:
  - Abstract Generation: scientific abstract from title + keywords + author history
  - Review Writing: product review from item info + reviewer history

Three-tier evaluation (same framework as LaMP):
  Tier 1 -- Naive: Random Profile, Most Recent, Input Copy
  Tier 2 -- Classical ML: TF-IDF Retrieval, BM25 Retrieval
  Tier 3 -- Neural: Vanilla GPT-2, RAG + GPT-2, HARE + GPT-2

Metrics: ROUGE-1, ROUGE-L (F1)

Usage:
    # Quick test (10 samples, no neural)
    python experiments/longlamp_experiment.py --task abstract --max-samples 10 --skip-neural

    # Full evaluation
    python experiments/longlamp_experiment.py --task abstract --max-samples 100

    # Review writing task
    python experiments/longlamp_experiment.py --task review --max-samples 50
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from hare.evaluation.baselines import get_all_baselines
from hare.evaluation.lamp import evaluate_rouge
from hare.evaluation.longlamp import load_longlamp


TASK_TO_CONFIG = {
    "abstract": "longlamp_abstract",
    "review": "longlamp_review",
    "topic": "longlamp_topic",
}


def run_longlamp_experiment(
    task: str = "abstract",
    max_samples: int | None = None,
    skip_neural: bool = False,
    output_path: Path | None = None,
    checkpoint: str | None = None,
    attention_checkpoint: str | None = None,
) -> dict:
    """Run LongLaMP evaluation across all baselines."""
    print("=" * 70)
    print(f"HARE LongLaMP Benchmark -- {task.capitalize()}")
    print("=" * 70)

    print(f"\nLoading LongLaMP {task}...")
    data = load_longlamp(task=task, split="val", max_samples=max_samples)
    print(f"  {len(data)} samples ready for evaluation\n")

    # Get baselines with appropriate task config
    config_key = TASK_TO_CONFIG.get(task, f"longlamp_{task}")
    baselines = get_all_baselines(
        include_neural=not skip_neural,
        checkpoint=checkpoint,
        task=config_key,
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
                    "prediction": predictions[i][:300],
                    "reference": references[i][:300],
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
    print(f"RESULTS SUMMARY -- LongLaMP {task.capitalize()}")
    print(f"{'=' * 70}")
    print(f"\n{'Method':<22s} {'ROUGE-1':<10s} {'ROUGE-L':<10s} {'Time':<8s}")
    print("-" * 50)

    for name, result in all_results.items():
        r1 = result["scores"]["rouge1"]
        rl = result["scores"]["rougeL"]
        t = result["scores"]["time_seconds"]
        print(f"{name:<22s} {r1:<10.4f} {rl:<10.4f} {t:<8.1f}s")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({
                "experiment": f"longlamp_{task}_evaluation",
                "task": task,
                "n_samples": len(data),
                "results": all_results,
            }, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="HARE LongLaMP Benchmark Evaluation"
    )
    parser.add_argument(
        "--task", type=str, default="abstract",
        choices=["abstract", "review", "topic"],
        help="LongLaMP task (default: abstract).",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--skip-neural", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--attention-checkpoint", type=str, default=None)
    args = parser.parse_args()

    output = args.output or Path(f"results/longlamp_{args.task}_results.json")
    run_longlamp_experiment(
        task=args.task,
        max_samples=args.max_samples,
        skip_neural=args.skip_neural,
        output_path=output,
        checkpoint=args.checkpoint,
        attention_checkpoint=args.attention_checkpoint,
    )


if __name__ == "__main__":
    main()
