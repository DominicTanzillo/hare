"""Data collection pipeline for HARE training and evaluation.

Collects structured AI skill/prompt data from public sources:
1. HuggingFace datasets (awesome-chatgpt-prompts, system prompt libraries)
2. GitHub repos (Claude skills, agent skills)
3. RecSys benchmarks (Amazon Reviews subset)

All data is validated through the same prompt injection pipeline
as local skill files (see hare.data.skills._validate_skill).

Usage:
    python -m hare.data.collect --source chatgpt-prompts
    python -m hare.data.collect --source agent-skills
    python -m hare.data.collect --source all
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from hare.data.skills import Skill, _validate_skill, save_skills

DATA_DIR = Path(__file__).parent.parent.parent / "data"


# -- HuggingFace: awesome-chatgpt-prompts ------------------------------------

def collect_chatgpt_prompts(output_dir: Path | None = None) -> list[Skill]:
    """Download fka/awesome-chatgpt-prompts from HuggingFace.

    1,131 act/prompt pairs, CC0 license.
    Maps: act -> title, prompt -> instructions.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [ERROR] Install 'datasets' package: pip install datasets")
        return []

    print("  Downloading fka/awesome-chatgpt-prompts from HuggingFace...")
    ds = load_dataset("fka/awesome-chatgpt-prompts", split="train")

    skills: list[Skill] = []
    for row in ds:
        act = row.get("act", "").strip()
        prompt = row.get("prompt", "").strip()

        if not act or not prompt:
            continue

        raw = {
            "title": act,
            "category": "chatgpt-prompt",
            "description": f"Act as {act}. {prompt[:200]}",
            "trigger": f"When the user needs a {act}",
            "instructions": prompt,
        }

        validated = _validate_skill(raw, source="chatgpt-prompts")
        if validated is not None:
            skills.append(validated)

    out = output_dir or DATA_DIR / "skills"
    out.mkdir(parents=True, exist_ok=True)
    save_skills(skills, out / "chatgpt_prompts.json")
    print(f"  Collected {len(skills)} skills from awesome-chatgpt-prompts")
    return skills


# -- GitHub: Clone and parse structured skill repos --------------------------

SKILL_REPOS = [
    {
        "name": "awesome-agent-skills",
        "url": "https://github.com/VoltAgent/awesome-agent-skills.git",
        "parser": "skill_md",
        "description": "300+ agent skills from VoltAgent community (MIT)",
    },
    {
        "name": "claude-skills-jeffallan",
        "url": "https://github.com/Jeffallan/claude-skills.git",
        "parser": "skill_md",
        "description": "66 specialized Claude skills across 12 categories",
    },
]


def _parse_skill_md(path: Path) -> dict | None:
    """Parse a SKILL.md or similar markdown file with YAML frontmatter."""
    text = path.read_text(encoding="utf-8", errors="replace")

    # Try YAML frontmatter
    title = path.stem.replace("-", " ").replace("_", " ").title()
    description = ""
    instructions = text

    # Extract title from first # heading
    heading_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    if heading_match:
        title = heading_match.group(1).strip()

    # Extract YAML frontmatter if present
    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", text, re.DOTALL)
    if fm_match:
        frontmatter = fm_match.group(1)
        body = fm_match.group(2)

        for line in frontmatter.split("\n"):
            if line.startswith("title:"):
                title = line.split(":", 1)[1].strip().strip('"').strip("'")
            elif line.startswith("description:"):
                description = line.split(":", 1)[1].strip().strip('"').strip("'")

        instructions = body.strip()

    if not description:
        # Use first paragraph as description
        paragraphs = [p.strip() for p in instructions.split("\n\n") if p.strip()]
        description = paragraphs[0][:500] if paragraphs else title

    return {
        "title": title,
        "category": "agent-skill",
        "description": description,
        "trigger": f"When the user needs help with {title.lower()}",
        "instructions": instructions[:5000],  # Truncate to max field length
    }


def collect_github_skills(
    repo_url: str,
    repo_name: str,
    output_dir: Path | None = None,
) -> list[Skill]:
    """Clone a GitHub repo and extract skills from markdown files."""
    import tempfile

    out = output_dir or DATA_DIR / "skills"
    out.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        clone_dir = Path(tmpdir) / repo_name
        print(f"  Cloning {repo_url}...")
        result = subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(clone_dir)],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            print(f"  [ERROR] git clone failed: {result.stderr[:200]}")
            return []

        # Find all markdown files that look like skills
        md_files = list(clone_dir.rglob("*.md"))
        # Filter out READMEs, CONTRIBUTINGs, etc.
        skill_files = [
            f for f in md_files
            if f.name.upper() not in ("README.MD", "CONTRIBUTING.MD", "LICENSE.MD",
                                       "CHANGELOG.MD", "CODE_OF_CONDUCT.MD")
            and len(f.read_text(encoding="utf-8", errors="replace")) > 50
        ]

        print(f"  Found {len(skill_files)} potential skill files")

        skills: list[Skill] = []
        for md_file in skill_files:
            raw = _parse_skill_md(md_file)
            if raw is None:
                continue
            validated = _validate_skill(raw, source=f"{repo_name}/{md_file.name}")
            if validated is not None:
                skills.append(validated)

        save_skills(skills, out / f"{repo_name}.json")
        print(f"  Collected {len(skills)} valid skills from {repo_name}")
        return skills


def collect_all_github_skills(output_dir: Path | None = None) -> list[Skill]:
    """Collect skills from all configured GitHub repos."""
    all_skills: list[Skill] = []
    for repo in SKILL_REPOS:
        skills = collect_github_skills(
            repo_url=repo["url"],
            repo_name=repo["name"],
            output_dir=output_dir,
        )
        all_skills.extend(skills)
    return all_skills


# -- Summary and statistics ---------------------------------------------------

def dataset_summary(data_dir: Path | None = None) -> dict:
    """Print summary statistics of collected data."""
    d = data_dir or DATA_DIR / "skills"
    if not d.exists():
        print("No data directory found.")
        return {}

    total = 0
    by_source: dict[str, int] = {}

    for path in sorted(d.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        count = len(data) if isinstance(data, list) else 1
        by_source[path.stem] = count
        total += count

    print(f"\nDataset Summary ({d}):")
    print(f"  Total skills: {total}")
    for source, count in sorted(by_source.items()):
        print(f"  - {source}: {count}")

    return {"total": total, "by_source": by_source}


# -- CLI entry point ----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="HARE Data Collection")
    parser.add_argument(
        "--source",
        choices=["chatgpt-prompts", "agent-skills", "all", "summary"],
        default="summary",
        help="Which data source to collect from.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.source == "summary":
        dataset_summary(args.output_dir)
    elif args.source == "chatgpt-prompts":
        collect_chatgpt_prompts(args.output_dir)
        dataset_summary(args.output_dir)
    elif args.source == "agent-skills":
        collect_all_github_skills(args.output_dir)
        dataset_summary(args.output_dir)
    elif args.source == "all":
        collect_chatgpt_prompts(args.output_dir)
        collect_all_github_skills(args.output_dir)
        dataset_summary(args.output_dir)


if __name__ == "__main__":
    main()
