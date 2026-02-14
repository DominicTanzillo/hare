"""Claude Skills dataset: loading, parsing, and synthetic generation.

Claude Skills are structured markdown files that define reusable behaviors
for Claude. They serve as HARE's primary training domain because they are:
- Structured enough to evaluate quality (valid markdown, correct format)
- Open-ended enough to benefit from generation (infinite possible skills)
- Practically useful -- synthesized output is immediately deployable
- Novel -- no existing work on "skill recommendation as generation"
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "skills"


class Skill(TypedDict):
    """A Claude Skill with structured fields."""
    title: str
    category: str
    description: str
    trigger: str
    instructions: str


# -- Built-in skill corpus (curated examples) ---------------------------------

BUILTIN_SKILLS: list[Skill] = [
    {
        "title": "Git Commit Helper",
        "category": "developer-tools",
        "description": "Analyzes staged changes and generates conventional commit messages following the Conventional Commits spec with type, scope, and description. Supports gitmoji prefix option.",
        "trigger": "When the user asks to commit code or review staged changes",
        "instructions": "Analyze the git diff, identify the type of change (feat, fix, refactor, docs, test, chore), write a concise commit message with a descriptive body. If gitmoji is requested, prefix with the appropriate emoji.",
    },
    {
        "title": "Code Reviewer",
        "category": "developer-tools",
        "description": "Reviews code for bugs, security vulnerabilities, and style issues. Provides inline suggestions with severity levels. Checks for OWASP top 10 vulnerabilities.",
        "trigger": "When the user asks for a code review or shares code for feedback",
        "instructions": "Analyze the code systematically: first check for correctness bugs, then security issues (injection, XSS, auth flaws), then style/maintainability. Assign severity (critical/high/medium/low) to each finding. Suggest fixes inline.",
    },
    {
        "title": "API Documentation Writer",
        "category": "developer-tools",
        "description": "Generates OpenAPI/Swagger documentation from code. Extracts endpoint signatures, request/response schemas, and authentication requirements.",
        "trigger": "When the user asks to document an API or generate API docs",
        "instructions": "Parse the route definitions, extract HTTP methods, paths, parameters, request bodies, and response types. Generate OpenAPI 3.0 YAML or markdown documentation with examples for each endpoint.",
    },
    {
        "title": "Test Generator",
        "category": "developer-tools",
        "description": "Creates unit tests covering happy path, edge cases, and error scenarios. Supports pytest, jest, and go test frameworks with fixtures and mocks.",
        "trigger": "When the user asks to write tests or improve test coverage",
        "instructions": "Identify all public functions/methods. For each, generate tests covering: normal input, boundary values, empty/null input, error conditions. Use appropriate mocking for external dependencies. Follow the AAA pattern (Arrange, Act, Assert).",
    },
    {
        "title": "Dependency Auditor",
        "category": "security",
        "description": "Scans project dependencies for known vulnerabilities and outdated packages. Cross-references CVE databases and suggests safe upgrade paths.",
        "trigger": "When the user asks about dependency security or outdated packages",
        "instructions": "Parse the dependency manifest (package.json, requirements.txt, go.mod, etc.). For each dependency, check for known CVEs, identify the latest safe version, assess upgrade risk (breaking changes), and provide a prioritized upgrade plan.",
    },
    {
        "title": "Performance Profiler",
        "category": "developer-tools",
        "description": "Analyzes code for performance bottlenecks. Identifies O(n^2) patterns, unnecessary allocations, and N+1 query problems with estimated impact.",
        "trigger": "When the user asks about performance, optimization, or speed",
        "instructions": "Scan the code for: nested loops over collections, repeated database queries in loops, unnecessary object creation, missing caching opportunities, synchronous operations that could be async. Quantify estimated impact (e.g., 'this loop is O(n^2), with n=1000 that is 1M operations').",
    },
    {
        "title": "Database Migration Writer",
        "category": "developer-tools",
        "description": "Generates database migration scripts from schema changes. Supports PostgreSQL, MySQL, and SQLite with rollback support.",
        "trigger": "When the user needs to create or modify database schemas",
        "instructions": "Compare the current schema with the desired state. Generate forward migration SQL (CREATE TABLE, ALTER TABLE, CREATE INDEX) and reverse migration SQL. Handle data transformations if existing data needs migration. Include transaction wrapping for safety.",
    },
    {
        "title": "Refactoring Assistant",
        "category": "developer-tools",
        "description": "Identifies code smells and suggests refactoring patterns. Handles extract method, extract class, and replace conditional with polymorphism.",
        "trigger": "When the user asks to refactor, clean up, or improve code structure",
        "instructions": "Identify code smells: long methods (>20 lines), deep nesting (>3 levels), duplicate code, god classes, feature envy. For each, suggest a specific refactoring pattern and show the before/after transformation. Verify behavior is preserved.",
    },
    {
        "title": "Error Message Improver",
        "category": "developer-tools",
        "description": "Rewrites error messages to be user-friendly, actionable, and informative. Converts stack traces and technical errors into guidance.",
        "trigger": "When the user shares an error message or asks for help with errors",
        "instructions": "Parse the error. Identify: what went wrong (root cause), why it happened (common triggers), how to fix it (specific steps), and how to prevent it (best practices). Rewrite the error message to lead with the fix, not the blame.",
    },
    {
        "title": "Markdown Report Generator",
        "category": "productivity",
        "description": "Converts raw data, notes, or analysis into formatted markdown reports with tables, charts descriptions, and executive summaries.",
        "trigger": "When the user asks to create a report or format findings",
        "instructions": "Structure the content with clear hierarchy: executive summary (3-5 bullet points), detailed findings (with tables where appropriate), methodology, and recommendations. Use markdown formatting consistently. Include a table of contents for long reports.",
    },
    {
        "title": "Regex Builder",
        "category": "developer-tools",
        "description": "Builds and explains regular expressions from natural language descriptions. Tests against sample inputs and explains each component.",
        "trigger": "When the user needs to create, understand, or debug a regex",
        "instructions": "Parse the natural language requirement. Build the regex step by step, explaining each component. Test against provided sample inputs (or generate test cases). Show matches and non-matches. Provide the regex in multiple flavors (Python, JavaScript, PCRE) if they differ.",
    },
    {
        "title": "Environment Setup Guide",
        "category": "developer-tools",
        "description": "Creates step-by-step environment setup guides for projects. Handles Python, Node.js, Go, Rust, and Docker environments.",
        "trigger": "When the user needs help setting up a development environment",
        "instructions": "Detect the project type from config files. Generate a step-by-step guide covering: prerequisites, installation, configuration, environment variables, database setup, running locally, and running tests. Include troubleshooting tips for common issues on macOS, Linux, and Windows.",
    },
    {
        "title": "Changelog Writer",
        "category": "developer-tools",
        "description": "Generates changelogs from git history following Keep a Changelog format. Groups changes by type and links to relevant PRs/issues.",
        "trigger": "When the user asks to generate a changelog or release notes",
        "instructions": "Parse git log between specified tags/commits. Categorize each commit: Added, Changed, Deprecated, Removed, Fixed, Security. Group by category, write human-readable descriptions (not raw commit messages). Link to PRs/issues where available.",
    },
    {
        "title": "Data Validator",
        "category": "data-engineering",
        "description": "Generates data validation schemas and checks from sample data. Supports JSON Schema, Pydantic models, and SQL constraints.",
        "trigger": "When the user needs to validate data format or quality",
        "instructions": "Analyze sample data to infer types, ranges, patterns, required fields, and relationships. Generate validation code in the requested format. Include edge case handling: nulls, empty strings, out-of-range values, format mismatches. Provide both schema definition and runtime validation code.",
    },
    {
        "title": "Learning Path Designer",
        "category": "education",
        "description": "Creates personalized learning paths for technical topics. Assesses current knowledge, identifies gaps, and sequences resources.",
        "trigger": "When the user wants to learn a new technology or skill",
        "instructions": "First assess what the user already knows (ask 2-3 diagnostic questions or infer from context). Identify prerequisite gaps. Create a sequenced learning plan with: topics ordered by dependency, estimated time per topic, recommended resources (docs, tutorials, exercises), and milestones to track progress.",
    },
    {
        "title": "Interview Prep Coach",
        "category": "career",
        "description": "Generates technical interview questions with solutions, hints, and evaluation criteria. Covers algorithms, system design, and behavioral questions.",
        "trigger": "When the user is preparing for a technical interview",
        "instructions": "Tailor questions to the target role and company. For coding questions: provide the problem, hints at three levels, optimal solution, time/space complexity analysis. For system design: provide the prompt, key discussion points, common pitfalls. For behavioral: provide STAR format templates.",
    },
    {
        "title": "Config File Generator",
        "category": "developer-tools",
        "description": "Generates configuration files for common tools: ESLint, Prettier, TypeScript, Docker, CI/CD pipelines, with opinionated defaults.",
        "trigger": "When the user needs to create or update configuration files",
        "instructions": "Detect the project stack. Generate config files with sensible defaults that match the project's existing patterns. Include comments explaining non-obvious settings. For CI/CD, generate workflows for the detected platform (GitHub Actions, GitLab CI, etc.).",
    },
    {
        "title": "SQL Query Optimizer",
        "category": "data-engineering",
        "description": "Analyzes SQL queries for performance issues and suggests optimizations. Explains query execution plans and recommends indexes.",
        "trigger": "When the user shares a slow SQL query or asks for optimization help",
        "instructions": "Analyze the query structure: identify full table scans, missing indexes, unnecessary joins, subqueries that could be CTEs, and N+1 patterns. Suggest specific optimizations with expected impact. Recommend index creation with the exact CREATE INDEX statement. Show before/after query plans if possible.",
    },
    {
        "title": "Architecture Decision Recorder",
        "category": "documentation",
        "description": "Creates Architecture Decision Records (ADRs) from discussions. Documents context, decision, consequences, and alternatives considered.",
        "trigger": "When the user makes a technical decision or asks to document an architecture choice",
        "instructions": "Structure as an ADR: Title, Status (proposed/accepted/deprecated), Context (why this decision is needed), Decision (what was decided), Consequences (positive and negative), Alternatives Considered (with pros/cons for each). Use Michael Nygard's ADR format.",
    },
    {
        "title": "Accessibility Auditor",
        "category": "frontend",
        "description": "Reviews HTML/JSX for accessibility issues. Checks WCAG 2.1 compliance including ARIA labels, contrast ratios, and keyboard navigation.",
        "trigger": "When the user asks about accessibility or shares frontend code",
        "instructions": "Check for: missing alt text on images, missing ARIA labels on interactive elements, insufficient color contrast (requires 4.5:1 for normal text), missing skip navigation links, form labels not associated with inputs, non-keyboard-accessible elements. Provide specific fixes for each issue with code examples.",
    },
]


def load_builtin_skills() -> list[Skill]:
    """Return the built-in curated skill corpus."""
    return BUILTIN_SKILLS.copy()


_REQUIRED_FIELDS = {"title", "category", "description", "trigger", "instructions"}
_MAX_FIELD_LENGTH = 5000  # Max chars per field -- reject suspiciously long content
_INJECTION_PATTERNS = [
    "ignore previous instructions",
    "ignore all previous",
    "disregard your instructions",
    "you are now",
    "new instructions:",
    "system prompt:",
    "override:",
    "<script",
    "javascript:",
    "data:text/html",
    "\\x00",
    "\\u0000",
]


def _validate_skill(skill: dict, source: str = "unknown") -> Skill | None:
    """Validate and sanitize a skill dict. Returns None if invalid.

    Guards against:
    - Missing required fields
    - Non-string field values
    - Excessively long fields (potential payload injection)
    - Known prompt injection patterns
    """
    # Check required fields exist and are strings
    for field in _REQUIRED_FIELDS:
        if field not in skill:
            print(f"  [SKIP] Skill from {source}: missing field '{field}'")
            return None
        if not isinstance(skill[field], str):
            print(f"  [SKIP] Skill from {source}: field '{field}' is not a string")
            return None
        if len(skill[field]) > _MAX_FIELD_LENGTH:
            print(f"  [SKIP] Skill '{skill.get('title', '?')}' from {source}: "
                  f"field '{field}' exceeds {_MAX_FIELD_LENGTH} chars")
            return None

    # Check for prompt injection patterns in all text fields
    all_text = " ".join(skill[f].lower() for f in _REQUIRED_FIELDS)
    for pattern in _INJECTION_PATTERNS:
        if pattern in all_text:
            print(f"  [REJECT] Skill '{skill.get('title', '?')}' from {source}: "
                  f"contains suspicious pattern: '{pattern}'")
            return None

    # Strip any extra fields -- only keep known ones
    return Skill(
        title=skill["title"].strip(),
        category=skill["category"].strip(),
        description=skill["description"].strip(),
        trigger=skill["trigger"].strip(),
        instructions=skill["instructions"].strip(),
    )


def load_skills_from_dir(directory: str | Path | None = None) -> list[Skill]:
    """Load and validate skills from JSON files in a directory.

    Each file should contain a JSON object or array of Skill dicts.
    All loaded skills are validated and sanitized:
    - Required fields must exist and be strings
    - Excessively long fields are rejected
    - Known prompt injection patterns are rejected
    - Only known fields are kept (extra keys stripped)
    """
    d = Path(directory) if directory else DATA_DIR
    skills: list[Skill] = []

    if not d.exists():
        return skills

    for path in sorted(d.glob("*.json")):
        source = path.name
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"  [SKIP] {source}: failed to parse JSON: {e}")
            continue

        items = data if isinstance(data, list) else [data]
        for item in items:
            if not isinstance(item, dict):
                continue
            validated = _validate_skill(item, source=source)
            if validated is not None:
                skills.append(validated)

    return skills


def skill_to_text(skill: Skill) -> str:
    """Convert a skill to a single text string for embedding."""
    return (
        f"{skill['title']} ({skill['category']}): {skill['description']} "
        f"Trigger: {skill['trigger']} "
        f"Instructions: {skill['instructions']}"
    )


def skills_to_texts(skills: list[Skill]) -> list[str]:
    """Convert a list of skills to text strings."""
    return [skill_to_text(s) for s in skills]


def get_skill_categories(skills: list[Skill]) -> dict[str, list[int]]:
    """Group skill indices by category."""
    categories: dict[str, list[int]] = {}
    for i, s in enumerate(skills):
        cat = s.get("category", "uncategorized")
        categories.setdefault(cat, []).append(i)
    return categories


def save_skills(skills: list[Skill], path: str | Path) -> None:
    """Save skills to a JSON file."""
    with open(path, "w") as f:
        json.dump(skills, f, indent=2)
