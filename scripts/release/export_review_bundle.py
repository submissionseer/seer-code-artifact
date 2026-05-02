#!/usr/bin/env python3
"""Build an anonymized code-first review bundle.

The bundle intentionally avoids raw data, checkpoints, local run logs, private
context notes, and generated experiment artifacts. It includes the canonical
execution surface plus tests and configs so reviewers can inspect and run the
reproducible pipeline without inheriting stale operator scripts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


REPO_ROOT = Path(__file__).resolve().parents[2]

TOP_LEVEL_FILES = [
    ".env.example",
    "LICENSE",
    "Makefile",
    "README.md",
    "REPRO.md",
    "RELEASE.md",
    "pyproject.toml",
    "uv.lock",
]

OPTIONAL_LOCK_FILES = [
    "poetry.lock",
]

CANONICAL_SCRIPT_FILES = [
    "scripts/bootstrap.sh",
    "scripts/env_check.py",
    "scripts/merge_model.py",
    "scripts/download_colbert_index.py",
    "scripts/download_datasets.py",
    "scripts/normalize_datasets.py",
    "scripts/release/export_review_bundle.py",
    "scripts/release/release_smoke.py",
    "scripts/rollouts/README.md",
    "scripts/rollouts/generate.py",
    "scripts/sft/README.md",
    "scripts/sft/prepare_data.py",
    "scripts/sft/train.py",
    "scripts/sft/eval.py",
    "scripts/dpo/analyze_pairs.py",
    "scripts/dpo/generate.py",
    "scripts/dpo/prepare.py",
    "scripts/dpo/score.py",
    "scripts/dpo/train.py",
    "scripts/dpo/validate.py",
    "scripts/eval/compare_rollout_reader_em.py",
    "scripts/eval/monitoring_signal_report.py",
    "scripts/eval/score_rollouts.py",
]

RUNPOD_SCRIPT_GLOBS = [
    "scripts/runpod_musique/*.py",
    "scripts/runpod_musique/*.sh",
    "scripts/runpod_musique/*.txt",
    "scripts/runpod_musique/README.md",
]

CANONICAL_CONFIG_FILES = [
    "configs/default.yaml",
    "configs/datasets.yaml",
    "configs/judges.yaml",
    "configs/llama3_sft_gold_k3_N5000_runpod.yml",
    "configs/llama3_sft_jina_mmr_k3_N5000_runpod.yml",
    "configs/llama3_sft_gold_iteration_20k_25k_runpod.yml",
    "configs/llama3_sft_mmr_iteration_20k_25k_runpod.yml",
    "configs/llama3_sft_gold_musique_k3_N10000_runpod.yml",
    "configs/llama3_sft_mmr_musique_k3_N10000_fullhop_runpod.yml",
    "configs/dpo_gold_5k_r64.yml",
    "configs/dpo_mmr_5k_r64.yml",
    "configs/dpo_raw_jina_maxmean_5k_r64.yml",
    "configs/dpo_gold_10k_r64.yml",
    "configs/dpo_mmr_10k_r64.yml",
    "configs/dpo_gold_iteration_20k_25k_r64.yml",
    "configs/dpo_mmr_iteration_20k_25k_r64.yml",
    "configs/dpo_musique_gold_10k_r64.seer_sft.yml",
    "configs/dpo_musique_mmr_10k_r64.seer_sft.yml",
    "configs/dpo_musique_raw_jina_maxmean_10k_r64.seer_sft.yml",
]

CANONICAL_TEST_FILES = [
    "tests/test_sft_prepare_data.py",
    "tests/test_dpo_utils.py",
    "tests/test_dpo_validate.py",
    "tests/test_normalize_datasets_script.py",
]

TEST_FIXTURE_GLOBS = [
    "tests/fixtures/**/*",
]

EVIDENCE_GLOBS = [
    "context/evidence/**/*.json",
    "context/evidence/**/*.md",
    "context/artifact_map.md",
    "context/results_matrix.md",
    "context/seed_reproduction_results.md",
    "context/statistical_claim_audit.md",
]

PAPER_SOURCE_GLOBS = [
    "paper/*.tex",
    "paper/*.bib",
    "paper/*.md",
    "paper/tables/*.tex",
    "paper/figs/*.png",
    "paper/figs/*.pdf",
    "paper/figs/*.svg",
]

EXCLUDED_PARTS = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "data",
    "dist",
    "logs",
}

# Research-side exploratory modules that are not part of the paper's canonical
# reviewer surface. Keep them in the working repo, but do not ship them in the
# anonymized review bundle.
EXCLUDED_REVIEW_FILES = {
    "seer/adversarial.py",
    "seer/baselines_ares.py",
    "seer/baselines_crossenc.py",
    "seer/baselines_qpp.py",
    "seer/baselines_ragas.py",
    "seer/direct_judge.py",
    "seer/dspy_seer.py",
    "seer/judge_agreement.py",
    "seer/optimize_rewriter.py",
}

TEXT_SUFFIXES = {
    "",
    ".bib",
    ".cfg",
    ".csv",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".sh",
    ".tex",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

HIGH_RISK_PATTERNS = [
    ("api_key_value", re.compile(r"(?i)(api[_-]?key|secret)\s*[:=]\s*['\"]?[A-Za-z0-9_./=+\-]{20,}")),
    ("openai_or_openrouter_key", re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}")),
    ("jina_key", re.compile(r"\bjina_[A-Za-z0-9_\-]{20,}")),
    ("local_user_path", re.compile(r"/Users/[A-Za-z0-9_.\-]+")),
    ("local_netrc", re.compile(r"\.netrc")),
    ("s3_uri", re.compile(r"\bs3://[A-Za-z0-9_.\-/]+")),
]

GENERATED_REVIEW_FILES = [
    "README.review.md",
    "MANIFEST.review.json",
]


@dataclass(frozen=True)
class ScanFinding:
    path: Path
    kind: str
    line: int
    excerpt: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="dist/seer_review_bundle.zip", help="Output zip path")
    parser.add_argument("--include-evidence", action="store_true", help="Include anonymized evidence docs")
    parser.add_argument("--include-paper-source", action="store_true", help="Include paper LaTeX source")
    parser.add_argument("--allow-warnings", action="store_true", help="Create bundle despite scan findings")
    parser.add_argument("--dry-run", action="store_true", help="Print planned files without writing a zip")
    parser.add_argument("--config", help=argparse.SUPPRESS)
    parser.add_argument("--include-artifacts", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def iter_existing(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for rel in paths:
        path = REPO_ROOT / rel
        if path.exists() and path.is_file():
            out.append(path)
    return out


def iter_globs(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for pattern in patterns:
        out.extend(path for path in REPO_ROOT.glob(pattern) if path.is_file())
    return out


def should_exclude(path: Path) -> bool:
    rel = path.relative_to(REPO_ROOT)
    if any(part in EXCLUDED_PARTS for part in rel.parts):
        return True
    if rel.as_posix() in EXCLUDED_REVIEW_FILES:
        return True
    if path.suffix in {".pyc", ".pyo", ".log", ".aux", ".blg"}:
        return True
    return False


def collect_files(include_evidence: bool, include_paper_source: bool) -> list[Path]:
    files: list[Path] = []
    files.extend(iter_existing(TOP_LEVEL_FILES))
    files.extend(iter_existing(OPTIONAL_LOCK_FILES))
    files.extend(iter_existing(CANONICAL_SCRIPT_FILES))
    files.extend(iter_globs(RUNPOD_SCRIPT_GLOBS))
    files.extend(iter_existing(CANONICAL_CONFIG_FILES))
    files.extend(iter_existing(CANONICAL_TEST_FILES))
    files.extend(iter_globs(TEST_FIXTURE_GLOBS))
    files.extend(path for path in (REPO_ROOT / "seer").rglob("*.py") if path.is_file())

    if include_evidence or include_paper_source:
        files.extend(iter_globs(EVIDENCE_GLOBS if include_evidence else []))
    if include_paper_source:
        files.extend(iter_globs(PAPER_SOURCE_GLOBS))

    unique = sorted({path.resolve() for path in files})
    return [path for path in unique if not should_exclude(path)]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_text_file(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES


def scan_file(path: Path) -> list[ScanFinding]:
    if not is_text_file(path):
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8", errors="ignore")

    findings: list[ScanFinding] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if path.name == "export_review_bundle.py" and "re.compile(" in line:
            continue
        if line.strip().startswith("#") and "API_KEY=" in line:
            continue
        for kind, pattern in HIGH_RISK_PATTERNS:
            if pattern.search(line):
                excerpt = pattern.sub("<REDACTED>", line.strip())
                findings.append(ScanFinding(path.relative_to(REPO_ROOT), kind, line_no, excerpt[:180]))
    return findings


def scan_files(files: list[Path]) -> list[ScanFinding]:
    findings: list[ScanFinding] = []
    for path in files:
        findings.extend(scan_file(path))
    return findings


def write_manifest(files: list[Path], dest: Path, findings: list[ScanFinding]) -> None:
    readme_path = dest / "README.review.md"
    manifest = {
        "created_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "bundle_type": "anonymized-code-review",
        "dataset_claim": False,
        "file_count": len(files) + 1,
        "bundle_file_count": len(files) + len(GENERATED_REVIEW_FILES),
        "inventory_excludes": ["MANIFEST.review.json"],
        "scan_findings": [
            {
                "path": str(f.path),
                "kind": f.kind,
                "line": f.line,
                "excerpt": f.excerpt,
            }
            for f in findings
        ],
        "files": [
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        ]
        + [
            {
                "path": "README.review.md",
                "size_bytes": readme_path.stat().st_size,
                "sha256": sha256_file(readme_path),
            }
        ],
    }
    (dest / "MANIFEST.review.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def copy_files(files: list[Path], dest: Path) -> None:
    for src in files:
        rel = src.relative_to(REPO_ROOT)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)


def write_bundle_readme(dest: Path) -> None:
    content = """# SEER Review Bundle

This is an anonymized, code-first review bundle for the SEER Evaluations & Datasets submission.

## Quick Start

```bash
uv sync
make env-check
make release-smoke
uv run pytest tests/test_sft_prepare_data.py tests/test_dpo_utils.py tests/test_dpo_validate.py tests/test_normalize_datasets_script.py
```

Use `README.md` for the repo surface, `REPRO.md` for the paper reproduction map,
and `RELEASE.md` for release-policy / asset-redistribution constraints.

## Canonical Pipeline Surface

- `scripts/rollouts/generate.py`
- `scripts/sft/prepare_data.py`
- `scripts/sft/train.py`
- `scripts/sft/eval.py`
- `scripts/dpo/generate.py`
- `scripts/dpo/score.py`
- `scripts/dpo/prepare.py`
- `scripts/dpo/train.py`
- `scripts/eval/compare_rollout_reader_em.py`
- `scripts/eval/monitoring_signal_report.py`

## External Dependencies

API-backed runs require credentials in environment variables documented in `.env.example`.
Raw datasets, generated rollouts, checkpoints, and private logs are intentionally not included.

## Dataset Position

This bundle does not claim or host a new dataset. It contains code and configuration for reproducing the evaluation method and experiments.
"""
    (dest / "README.review.md").write_text(content, encoding="utf-8")


def create_zip(bundle_dir: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as zf:
        for path in sorted(bundle_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(bundle_dir))


def print_plan(files: list[Path], findings: list[ScanFinding]) -> None:
    print(f"Planned bundle files: {len(files) + len(GENERATED_REVIEW_FILES)}")
    print(f"Planned manifest inventory: {len(files) + 1} (+ README.review.md; excludes MANIFEST.review.json)")
    for path in files:
        print(path.relative_to(REPO_ROOT))
    for generated in GENERATED_REVIEW_FILES:
        print(generated)
    if findings:
        print("\nScan findings:", file=sys.stderr)
        for finding in findings:
            print(
                f"{finding.path}:{finding.line}: {finding.kind}: {finding.excerpt}",
                file=sys.stderr,
            )


def main() -> int:
    args = parse_args()
    if args.config:
        print("--config is ignored by the canonical review-bundle exporter.", file=sys.stderr)
    if args.include_artifacts:
        print(
            "--include-artifacts is deprecated; use --include-evidence after anonymity review.",
            file=sys.stderr,
        )

    files = collect_files(
        include_evidence=args.include_evidence or args.include_artifacts,
        include_paper_source=args.include_paper_source,
    )
    findings = scan_files(files)

    if args.dry_run:
        print_plan(files, findings)
        return 1 if findings and not args.allow_warnings else 0

    if findings and not args.allow_warnings:
        print_plan(files, findings)
        print(
            "\nRefusing to create review bundle until scan findings are resolved. "
            "Use --allow-warnings only for local debugging.",
            file=sys.stderr,
        )
        return 1

    output = (REPO_ROOT / args.output).resolve()
    with tempfile.TemporaryDirectory(prefix="seer_review_bundle_") as tmp:
        bundle_dir = Path(tmp) / "seer_review_bundle"
        bundle_dir.mkdir()
        copy_files(files, bundle_dir)
        write_bundle_readme(bundle_dir)
        write_manifest(files, bundle_dir, findings)
        create_zip(bundle_dir, output)

    size_mb = output.stat().st_size / (1024 * 1024)
    try:
        output_display = str(output.relative_to(REPO_ROOT))
    except ValueError:
        output_display = str(output)
    bundle_count = len(files) + len(GENERATED_REVIEW_FILES)
    print(f"Wrote {output_display} ({size_mb:.2f} MB, {bundle_count} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
