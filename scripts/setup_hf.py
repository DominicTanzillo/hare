#!/usr/bin/env python3
"""Set up HuggingFace repos for the HARE project.

Creates:
1. DTanzillo/hare-lamp4-eval — Dataset repo for LaMP-4 evaluation results
2. DTanzillo/hare-eval-portal — Gradio Space for the human evaluation portal

Usage:
    python scripts/setup_hf.py
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi, create_repo


def setup_dataset_repo(api: HfApi) -> str:
    """Create the evaluation dataset repo."""
    repo_id = "DTanzillo/hare-lamp4-eval"

    print(f"\n--- Setting up dataset repo: {repo_id} ---")
    try:
        repo_url = create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            exist_ok=True,
            private=False,
        )
        print(f"  Created/found: {repo_url}")
    except Exception as e:
        print(f"  Error: {e}")
        return repo_id

    # Upload README
    readme_content = """---
license: apache-2.0
task_categories:
  - text-generation
  - summarization
language:
  - en
tags:
  - personalization
  - recommendation
  - lamp
  - headline-generation
pretty_name: HARE LaMP-4 Evaluation
---

# HARE LaMP-4 Evaluation Dataset

Evaluation results from **HARE (Hybrid Attention-Reinforced Exploration)** on the
[LaMP-4 benchmark](https://lamp-benchmark.github.io/) (Personalized News Headline Generation).

## Baselines

| Tier | Method | Description |
|------|--------|-------------|
| 1 (Naive) | Random Profile | Randomly select a headline from user's profile |
| 1 (Naive) | Most Recent | Use the most recent profile headline |
| 1 (Naive) | Input Copy | Extract first sentence of article |
| 2 (Classical ML) | TF-IDF Retrieval | Retrieve most similar profile item by TF-IDF |
| 2 (Classical ML) | BM25 Retrieval | BM25-based retrieval from profile |
| 3 (Neural) | Vanilla GPT-2 | Fine-tuned DistilGPT2 (no personalization) |
| 3 (Neural) | RAG + GPT-2 | TF-IDF retrieval + GPT-2 generation |
| 3 (Neural) | HARE + GPT-2 | User-conditioned attention retrieval + GPT-2 |

## Metrics

- **ROUGE-1** (F1): Unigram overlap
- **ROUGE-L** (F1): Longest common subsequence

## Citation

```bibtex
@article{tanzillo2026hare,
  title={HARE: Hybrid Attention-Reinforced Exploration for Generative Recommendation},
  author={Tanzillo, Dominic},
  year={2026}
}
```

## Links

- [GitHub Repository](https://github.com/DominicTanzillo/hare)
- [Evaluation Portal](https://huggingface.co/spaces/DTanzillo/hare-eval-portal)
"""
    api.upload_file(
        path_or_fileobj=readme_content.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Initial dataset card",
    )
    print("  Uploaded README.md")

    return repo_id


def setup_space(api: HfApi) -> str:
    """Create the Gradio Space for the evaluation portal."""
    repo_id = "DTanzillo/hare-eval-portal"

    print(f"\n--- Setting up Space: {repo_id} ---")
    try:
        repo_url = create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="gradio",
            exist_ok=True,
            private=False,
        )
        print(f"  Created/found: {repo_url}")
    except Exception as e:
        print(f"  Error: {e}")
        return repo_id

    # Upload app.py for the Space
    # The Space needs a self-contained app.py
    space_app = '''"""HARE Evaluation Portal — HuggingFace Space.

Blind A/B evaluation of headline generation methods on LaMP-4.
"""

import json
import random
import re
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path

import gradio as gr


@dataclass
class EvalSample:
    sample_id: str
    article: str
    reference_headline: str
    methods: dict  # method_name -> generated headline


@dataclass
class ParticipantResponse:
    session_id: str
    participant_id: str
    sample_id: str
    timestamp: float
    ratings: dict
    ranking: list
    time_spent_seconds: float
    notes: str = ""


class ResponseStore:
    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, response: ParticipantResponse) -> None:
        with open(self.output_path, "a") as f:
            f.write(json.dumps(asdict(response)) + "\\n")

    def count(self) -> int:
        if not self.output_path.exists():
            return 0
        with open(self.output_path) as f:
            return sum(1 for _ in f)


# Load demo samples (embedded for the Space)
DEMO_SAMPLES = [
    EvalSample(
        sample_id="demo_1",
        article="The rise of artificial intelligence has transformed how businesses "
                "approach customer service. Chatbots and virtual assistants now handle "
                "millions of queries daily, reducing wait times and operational costs. "
                "However, critics argue that the lack of human empathy in AI interactions "
                "could damage brand loyalty in the long run.",
        reference_headline="AI Transforms Customer Service, But At What Cost?",
        methods={
            "Method A": "AI Transforms Customer Service, But At What Cost?",
            "Method B": "The rise of artificial intelligence has transformed how businesses",
            "Method C": "Chatbots Replace Humans in Customer Service Revolution",
            "Method D": "AI Customer Service: The Future of Business Communication",
        },
    ),
    EvalSample(
        sample_id="demo_2",
        article="A new study published in Nature reveals that ocean temperatures have "
                "risen faster than previously estimated over the past decade. Researchers "
                "analyzed data from thousands of deep-sea sensors and found that warming "
                "trends in the Atlantic and Pacific oceans could accelerate sea level rise "
                "by up to 30% more than current models predict.",
        reference_headline="Ocean Warming Outpaces Predictions, Study Finds",
        methods={
            "Method A": "A new study published in Nature reveals that ocean temperatures",
            "Method B": "Ocean Warming Outpaces Predictions, New Research Shows",
            "Method C": "Deep Sea Sensors Reveal Alarming Temperature Trends",
            "Method D": "Sea Level Rise Could Accelerate 30% Faster Than Expected",
        },
    ),
]


def get_shuffled_methods(sample):
    items = list(sample.methods.items())
    random.shuffle(items)
    labeled = []
    for i, (real_name, headline) in enumerate(items):
        label = f"Headline {chr(65 + i)}"
        labeled.append((label, real_name, headline))
    return labeled


store = ResponseStore(Path("responses/human_eval.jsonl"))


with gr.Blocks(title="HARE Evaluation Portal", theme=gr.themes.Soft()) as demo:
    session_id = gr.State(lambda: str(uuid.uuid4()))
    current_idx = gr.State(0)
    sample_start_time = gr.State(time.time)
    shuffled_methods = gr.State([])
    participant_id = gr.State("")

    with gr.Column(visible=True) as welcome_screen:
        gr.Markdown(
            "# HARE Evaluation Portal\\n\\n"
            "## Personalized News Headline Generation\\n\\n"
            "Rate computer-generated headlines on quality, relevance, and specificity.\\n\\n"
            f"**{len(DEMO_SAMPLES)} articles** to evaluate (~5 minutes)."
        )
        pid_input = gr.Textbox(label="Participant ID (e.g. your initials)", placeholder="JD")
        start_btn = gr.Button("Begin Evaluation", variant="primary", size="lg")

    with gr.Column(visible=False) as eval_screen:
        progress_md = gr.Markdown("")
        with gr.Accordion("Article", open=True):
            article_text = gr.Markdown("")
        gr.Markdown("---\\n### Generated Headlines")

        headline_components = []
        rating_components = []
        for i in range(6):
            with gr.Group(visible=False) as group:
                headline_md = gr.Markdown(f"**Headline {chr(65+i)}**: ...")
                with gr.Row():
                    q = gr.Slider(1, 5, step=1, value=3, label="Quality")
                    r = gr.Slider(1, 5, step=1, value=3, label="Relevance")
                    p = gr.Slider(1, 5, step=1, value=3, label="Specificity")
            headline_components.append((group, headline_md))
            rating_components.append((q, r, p))

        ranking_input = gr.Textbox(label="Rank best to worst (e.g. A, C, B, D)", placeholder="A, B, C, D")
        notes_input = gr.Textbox(label="Optional notes", placeholder="Comments...")
        submit_btn = gr.Button("Submit & Next", variant="primary", size="lg")

    with gr.Column(visible=False) as done_screen:
        gr.Markdown("# Thank you!\\nYour responses have been saved.")
        response_count = gr.Markdown("")

    def start_eval(pid, sess_id):
        if not pid.strip():
            pid = f"anon_{sess_id[:8]}"
        sample = DEMO_SAMPLES[0]
        methods = get_shuffled_methods(sample)
        updates = [
            gr.Column(visible=False), gr.Column(visible=True), gr.Column(visible=False),
            f"**Article 1 / {len(DEMO_SAMPLES)}**", sample.article,
            0, time.time(), methods, pid.strip(),
        ]
        for i in range(6):
            if i < len(methods):
                label, _, headline = methods[i]
                updates.extend([gr.Group(visible=True), f"**{label}**: {headline}"])
            else:
                updates.extend([gr.Group(visible=False), ""])
        updates.append(", ".join([chr(65+i) for i in range(len(methods))]))
        return updates

    def submit_and_next(idx, sess_id, pid, methods, start_t, *args):
        n = len(methods)
        sliders = args[:n*3]
        ranking_text = args[n*3]
        notes = args[n*3+1]
        ratings = {}
        for i, (label, real_name, _) in enumerate(methods):
            ratings[label] = {"method": real_name, "quality": int(sliders[i*3]), "relevance": int(sliders[i*3+1]), "personalization": int(sliders[i*3+2])}
        ranking = [r.strip().upper() for r in ranking_text.split(",") if r.strip()]
        response = ParticipantResponse(sess_id, pid, DEMO_SAMPLES[idx].sample_id, time.time(), ratings, ranking, round(time.time()-start_t,1), notes or "")
        store.save(response)
        next_idx = idx + 1
        if next_idx >= len(DEMO_SAMPLES):
            return [gr.Column(visible=False), gr.Column(visible=False), gr.Column(visible=True), "", "", next_idx, time.time(), [], pid, *[gr.Group(visible=False) for _ in range(6)], *["" for _ in range(6)], "", f"**Total responses: {store.count()}**"]
        sample = DEMO_SAMPLES[next_idx]
        new_methods = get_shuffled_methods(sample)
        updates = [gr.Column(visible=False), gr.Column(visible=True), gr.Column(visible=False), f"**Article {next_idx+1} / {len(DEMO_SAMPLES)}**", sample.article, next_idx, time.time(), new_methods, pid]
        for i in range(6):
            if i < len(new_methods):
                label, _, headline = new_methods[i]
                updates.extend([gr.Group(visible=True), f"**{label}**: {headline}"])
            else:
                updates.extend([gr.Group(visible=False), ""])
        updates.extend([", ".join([chr(65+i) for i in range(len(new_methods))]), ""])
        return updates

    start_outputs = [welcome_screen, eval_screen, done_screen, progress_md, article_text, current_idx, sample_start_time, shuffled_methods, participant_id]
    for g, m in headline_components:
        start_outputs.extend([g, m])
    start_outputs.append(ranking_input)
    start_btn.click(start_eval, [pid_input, session_id], start_outputs)

    submit_inputs = [current_idx, session_id, participant_id, shuffled_methods, sample_start_time]
    for q, r, p in rating_components:
        submit_inputs.extend([q, r, p])
    submit_inputs.extend([ranking_input, notes_input])
    submit_outputs = [welcome_screen, eval_screen, done_screen, progress_md, article_text, current_idx, sample_start_time, shuffled_methods, participant_id]
    for g, m in headline_components:
        submit_outputs.extend([g, m])
    submit_outputs.extend([ranking_input, response_count])
    submit_btn.click(submit_and_next, submit_inputs, submit_outputs)


if __name__ == "__main__":
    demo.launch()
'''

    api.upload_file(
        path_or_fileobj=space_app.encode(),
        path_in_repo="app.py",
        repo_id=repo_id,
        repo_type="space",
        commit_message="Initial evaluation portal",
    )
    print("  Uploaded app.py")

    # Upload requirements.txt
    requirements = "gradio>=4.0\n"
    api.upload_file(
        path_or_fileobj=requirements.encode(),
        path_in_repo="requirements.txt",
        repo_id=repo_id,
        repo_type="space",
        commit_message="Add requirements",
    )
    print("  Uploaded requirements.txt")

    return repo_id


def main():
    api = HfApi()
    user = api.whoami()
    print(f"Logged in as: {user['name']}")

    dataset_id = setup_dataset_repo(api)
    space_id = setup_space(api)

    print("\n" + "=" * 60)
    print("HuggingFace setup complete!")
    print(f"  Dataset: https://huggingface.co/datasets/{dataset_id}")
    print(f"  Space:   https://huggingface.co/spaces/{space_id}")
    print("=" * 60)


if __name__ == "__main__":
    main()
