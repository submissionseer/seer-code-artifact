#!/usr/bin/env python3
"""Normalize staged raw HotpotQA and MuSiQue data into canonical JSONL."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from seer.datasets import HotpotAdapter, MuSiQueAdapter
from seer.util_io import read_jsonl, write_jsonl
from seer.util_log import get_logger


def _normalize_jsonl_dataset(
    *,
    raw_path: Path,
    norm_path: Path,
    logger,
    adapter,
    dataset_label: str,
    force: bool = False,
) -> None:
    if norm_path.exists() and not force:
        logger.info("%s normalized data already exists at %s", dataset_label, norm_path)
        return
    if not raw_path.exists():
        logger.info("%s raw data not found at %s; skipping", dataset_label, raw_path)
        return

    rows = [adapter.normalize_to_record(item) for item in read_jsonl(raw_path)]
    write_jsonl(norm_path, rows)


def _normalize_musique(raw_dir: Path, norm_path: Path, logger, force: bool = False) -> None:
    _normalize_jsonl_dataset(
        raw_path=raw_dir / "musique" / "raw.jsonl",
        norm_path=norm_path,
        logger=logger,
        adapter=MuSiQueAdapter(),
        dataset_label="MuSiQue",
        force=force,
    )


def _normalize_hotpot(raw_dir: Path, norm_path: Path, logger, force: bool = False) -> None:
    _normalize_jsonl_dataset(
        raw_path=raw_dir / "hotpot" / "raw.jsonl",
        norm_path=norm_path,
        logger=logger,
        adapter=HotpotAdapter(),
        dataset_label="HotpotQA",
        force=force,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    norm_dir = Path(config["norm_dir"])
    norm_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(config["raw_dir"])
    logger = get_logger("normalize", Path(config["logs_dir"]) / "normalize.log")

    _normalize_musique(raw_dir, norm_dir / "musique.jsonl", logger, force=args.force)
    _normalize_hotpot(raw_dir, norm_dir / "hotpot.jsonl", logger, force=args.force)

    available = {
        "musique": (norm_dir / "musique.jsonl").exists(),
        "hotpot": (norm_dir / "hotpot.jsonl").exists(),
    }
    if not any(available.values()):
        raise FileNotFoundError(
            "No normalizable datasets found. Expected staged raw files under "
            f"{raw_dir}/musique/raw.jsonl and/or {raw_dir}/hotpot/raw.jsonl."
        )

    logger.info(
        "Normalized datasets available: %s",
        ", ".join(name for name, exists in available.items() if exists),
    )
    logger.info("Normalization complete (MuSiQue + HotpotQA).")


if __name__ == "__main__":
    main()
