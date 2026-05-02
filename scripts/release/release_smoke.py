#!/usr/bin/env python3
"""Minimal end-to-end release smoke check for the canonical pipeline."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "datasets"

SFT_SMOKE_ROLLOUTS = [
    {
        "base_qid": "smoke-q1",
        "prompt_variant_id": 0,
        "question": "Which company developed the game voiced by Alice David as Lara Croft?",
        "retrieved": [
            {"hop": 1, "title": "Alice David", "text": "Alice David voiced Lara Croft in Tomb Raider."},
            {"hop": 2, "title": "Crystal Dynamics", "text": "Crystal Dynamics developed Tomb Raider (2013)."},
        ],
        "seer_label": "",
        "hop_details": [
            {"hop": 1, "query": "Alice David Lara Croft game", "coverage_after_hop": 0.5},
            {"hop": 2, "query": "Crystal Dynamics Tomb Raider developer", "coverage_after_hop": 1.0},
        ],
    },
    {
        "base_qid": "smoke-q1",
        "prompt_variant_id": 1,
        "question": "Which company developed the game voiced by Alice David as Lara Croft?",
        "retrieved": [
            {"hop": 1, "title": "Alice David", "text": "Alice David is a French actress."},
            {"hop": 2, "title": "Square Enix", "text": "Square Enix published Tomb Raider (2013)."},
        ],
        "seer_label": "",
        "hop_details": [
            {"hop": 1, "query": "Alice David actress", "coverage_after_hop": 0.0},
            {"hop": 2, "query": "Square Enix Tomb Raider", "coverage_after_hop": 0.5},
        ],
    },
]


DPO_SMOKE_ROLLOUTS = [
    {
        "qid": "smoke-dpo-q1",
        "question": "Which city is the headquarters of the Oberoi hotel company located in?",
        "gold_titles": ["The Oberoi Group", "Gurgaon"],
        "hop1": {
            "prompt_context": "N/A",
            "candidates": [
                {
                    "candidate_idx": 0,
                    "query": "\"Mumbai\"",
                    "retrieved": [
                        {"title": "Mumbai", "text": "Mumbai is a major city in India.", "doc_id": 1},
                    ],
                    "gold_ap": 0.0,
                },
                {
                    "candidate_idx": 1,
                    "query": "head office of Oberoi hotel company city",
                    "retrieved": [
                        {"title": "The Oberoi Group", "text": "The Oberoi Group is headquartered in Gurgaon.", "doc_id": 2},
                    ],
                    "gold_ap": 0.5,
                },
                {
                    "candidate_idx": 2,
                    "query": "Oberoi hotel company headquarters Gurgaon",
                    "retrieved": [
                        {"title": "The Oberoi Group", "text": "The Oberoi Group is headquartered in Gurgaon.", "doc_id": 2},
                        {"title": "Gurgaon", "text": "Gurgaon is a city in Haryana.", "doc_id": 3},
                    ],
                    "gold_ap": 1.0,
                },
            ],
        },
        "hop2": {
            "prompt_context": "Title: The Oberoi Group\nText: The Oberoi Group is headquartered in Gurgaon.",
            "context_source": {"selection_hop": 1, "selected_candidate_idx": 2},
            "exclude_pids": [2, 3],
            "candidates": [
                {
                    "candidate_idx": 0,
                    "query": "The city where the head office of the Oberoi hotel company is located.\n\n[[ ## answer ## ]]\nGurgaon",
                    "retrieved": [
                        {"title": "Gurgaon", "text": "Gurgaon is a city in Haryana.", "doc_id": 3},
                    ],
                    "gold_ap_hop2_only": 0.5,
                    "gold_ap_cumulative": 1.0,
                },
                {
                    "candidate_idx": 1,
                    "query": "\"The Oberoi family hotel company head office city\"",
                    "retrieved": [
                        {"title": "Mumbai", "text": "Mumbai is a major city in India.", "doc_id": 1},
                    ],
                    "gold_ap_hop2_only": 0.0,
                    "gold_ap_cumulative": 0.5,
                },
            ],
        },
    }
]


EVAL_SMOKE_ROLLOUTS = [
    {
        "qid": "smoke-eval-q1",
        "question": "Which company developed the game voiced by Alice David as Lara Croft?",
        "retrieved": [
            {"hop": 1, "title": "Alice David", "text": "Alice David voiced Lara Croft in Tomb Raider."},
            {"hop": 2, "title": "Crystal Dynamics", "text": "Crystal Dynamics developed Tomb Raider (2013)."},
        ],
        "hop_details": [
            {"hop": 1, "query": "Alice David Lara Croft game"},
            {"hop": 2, "query": "Crystal Dynamics Tomb Raider developer"},
        ],
    }
]


def _load_fixture_records(name: str) -> list[dict]:
    payload = json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))
    return payload["records"]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _run(cmd: list[str], *, env: dict[str, str] | None = None, capture: bool = False) -> subprocess.CompletedProcess[str] | None:
    print(f"\n$ {' '.join(cmd)}")
    if capture:
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            check=True,
            env=env,
            text=True,
            capture_output=True,
        )
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
        return result

    subprocess.run(cmd, cwd=REPO_ROOT, check=True, env=env)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=".tmp/release_smoke",
        help="Workspace root for temporary smoke outputs",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep the smoke workspace instead of deleting it first",
    )
    parser.add_argument(
        "--with-jina",
        action="store_true",
        help="Opt in to one Jina-backed scoring leg; omitted by default so the smoke remains fully offline.",
    )
    args = parser.parse_args()

    smoke_root = (REPO_ROOT / args.root).resolve()
    if smoke_root.exists() and not args.keep:
        shutil.rmtree(smoke_root)
    smoke_root.mkdir(parents=True, exist_ok=True)

    config_path = smoke_root / "smoke_config.yaml"
    config = {
        "raw_dir": str(smoke_root / "raw"),
        "norm_dir": str(smoke_root / "norm"),
        "logs_dir": str(smoke_root / "logs"),
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")

    sft_input = smoke_root / "fixtures" / "sft_rollouts.jsonl"
    dpo_input = smoke_root / "fixtures" / "dpo_rollouts.jsonl"
    eval_input = smoke_root / "fixtures" / "eval_rollouts.jsonl"
    requirements_path = smoke_root / "fixtures" / "requirements.json"
    hotpot_raw = smoke_root / "raw" / "hotpot" / "raw.jsonl"
    musique_raw = smoke_root / "raw" / "musique" / "raw.jsonl"
    _write_jsonl(sft_input, SFT_SMOKE_ROLLOUTS)
    _write_jsonl(dpo_input, DPO_SMOKE_ROLLOUTS)
    _write_jsonl(eval_input, EVAL_SMOKE_ROLLOUTS)
    requirements_path.parent.mkdir(parents=True, exist_ok=True)
    requirements_path.write_text(
        json.dumps(
            {
                "smoke-dpo-q1": ["Identify the Oberoi Group headquarters city."],
                "smoke-eval-q1": ["Find who developed the Lara Croft game voiced by Alice David."],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_jsonl(hotpot_raw, _load_fixture_records("hotpot_raw_samples.json"))
    _write_jsonl(musique_raw, _load_fixture_records("musique_raw_samples.json"))

    py = [sys.executable]
    env = os.environ.copy()

    _run(py + ["scripts/env_check.py", "--config", str(config_path)])
    _run(py + ["scripts/download_datasets.py", "--help"])
    _run(py + ["scripts/download_colbert_index.py", "--help"])
    _run(py + ["scripts/normalize_datasets.py", "--config", str(config_path), "--force"])
    _run(py + ["scripts/rollouts/generate.py", "--help"])
    _run(py + ["scripts/dpo/generate.py", "--help"])
    _run(
        py
        + [
            "scripts/dpo/generate.py",
            "--config",
            str(config_path),
            "--input",
            str(dpo_input),
            "--output",
            str(smoke_root / "outputs" / "dpo_generate_gold.jsonl"),
            "--context-select",
            "gold",
            "--remote-url",
            "http://127.0.0.1:65535",
            "--validate-only",
        ]
    )
    _run(
        py
        + [
            "scripts/dpo/generate.py",
            "--config",
            str(config_path),
            "--input",
            str(dpo_input),
            "--output",
            str(smoke_root / "outputs" / "dpo_generate_mmr.jsonl"),
            "--context-select",
            "mmr",
            "--requirements",
            str(requirements_path),
            "--remote-url",
            "http://127.0.0.1:65535",
            "--validate-only",
        ]
    )
    _run(
        py
        + [
            "scripts/dpo/score.py",
            "--input",
            str(dpo_input),
            "--output",
            str(smoke_root / "outputs" / "dpo_score_seer.jsonl"),
            "--mode",
            "seer",
            "--validate-only",
        ]
    )
    _run(
        py
        + [
            "scripts/dpo/score.py",
            "--input",
            str(dpo_input),
            "--output",
            str(smoke_root / "outputs" / "dpo_score_mmr.jsonl"),
            "--mode",
            "mmr",
            "--requirements",
            str(requirements_path),
            "--validate-only",
        ]
    )
    _run(py + ["scripts/sft/eval.py", "--help"])
    _run(
        py
        + [
            "scripts/eval/score_rollouts.py",
            "--input",
            str(eval_input),
            "--output",
            str(smoke_root / "outputs" / "eval_scored.jsonl"),
            "--validate-only",
        ]
    )
    _run(
        py
        + [
            "scripts/sft/prepare_data.py",
            "--selector",
            "gold",
            "--input",
            str(sft_input),
            "--output",
            str(smoke_root / "outputs" / "sft_gold.jsonl"),
        ]
    )
    _run(
        py
        + [
            "scripts/dpo/prepare.py",
            "--input",
            str(dpo_input),
            "--output",
            str(smoke_root / "outputs" / "dpo_pairs_gold.jsonl"),
            "--scoring",
            "gold",
            "--min-gap",
            "0.0",
            "--pair-mode",
            "adjacent",
        ]
    )
    _run(py + ["scripts/sft/train.py", "--config", "configs/llama3_sft_gold_k3_N5000_runpod.yml", "--dry-run"])
    _run(py + ["scripts/dpo/train.py", "--config", "configs/dpo_mmr_5k_r64.yml", "--dry-run"])
    bundle_dry_run = _run(
        py + ["scripts/release/export_review_bundle.py", "--dry-run"],
        capture=True,
    )
    assert bundle_dry_run is not None
    if "configs/judges.yaml" not in bundle_dry_run.stdout:
        raise RuntimeError(
            "Release bundle dry-run omitted configs/judges.yaml; canonical DPO/SEER paths would break."
        )

    if args.with_jina:
        if not env.get("JINA_AI_API_KEY"):
            raise RuntimeError("--with-jina was requested but JINA_AI_API_KEY is not set.")
        scored_path = smoke_root / "outputs" / "dpo_rollouts_raw_jina.jsonl"
        _run(
            py
            + [
                "scripts/dpo/score.py",
                "--input",
                str(dpo_input),
                "--output",
                str(scored_path),
                "--mode",
                "raw_jina_maxmean",
                "--limit",
                "1",
            ],
            env=env,
        )
        _run(
            py
            + [
                "scripts/dpo/prepare.py",
                "--input",
                str(scored_path),
                "--output",
                str(smoke_root / "outputs" / "dpo_pairs_raw_jina.jsonl"),
                "--scoring",
                "raw_jina_maxmean",
                "--min-gap",
                "0.0",
                "--pair-mode",
                "adjacent",
            ]
        )
    else:
        print("\nSkipping Jina-backed DPO scoring; pass --with-jina to include the optional online check.")

    print("\nRelease smoke check completed.")
    print(f"Workspace: {smoke_root}")
    print(
        "Note: this smoke path is offline and fixture-based by default. Full local Hotpot rollout/eval "
        "still requires a ColBERT passage collection (collection.tsv) in addition to the HF "
        "wiki2017 index."
    )


if __name__ == "__main__":
    main()
