#!/usr/bin/env python3
"""Download the paper's raw benchmark datasets from official/public sources."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from datasets import load_dataset

from seer.util_io import ensure_dir, write_jsonl
from seer.util_log import get_logger


def _download_musique(raw_dir: Path, logger) -> None:
    output = raw_dir / "musique" / "raw.jsonl"
    if output.exists():
        logger.info("MuSiQue raw data already exists at %s", output)
        return
    logger.info("Downloading MuSiQue from Hugging Face (dgslibisey/MuSiQue)")
    dataset = load_dataset("dgslibisey/MuSiQue")
    rows = []
    for split in dataset.keys():
        for item in dataset[split]:
            row = dict(item)
            row["split"] = split
            rows.append(row)
    ensure_dir(output.parent)
    write_jsonl(output, rows)
    logger.info("Wrote %d MuSiQue rows to %s", len(rows), output)


def _download_hotpot(raw_dir: Path, logger) -> None:
    output = raw_dir / "hotpot" / "raw.jsonl"
    if output.exists():
        logger.info("HotpotQA distractor raw data already exists at %s", output)
        return
    logger.info("Downloading HotpotQA distractor split from Hugging Face (hotpot_qa, distractor)")
    dataset = load_dataset("hotpot_qa", "distractor")
    rows = []
    for split in dataset.keys():
        for item in dataset[split]:
            row = dict(item)
            row["split"] = split
            rows.append(row)
    ensure_dir(output.parent)
    write_jsonl(output, rows)
    logger.info("Wrote %d HotpotQA distractor rows to %s", len(rows), output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["musique", "hotpot"],
        choices=["musique", "hotpot"],
        help="Datasets to download from public sources",
    )
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    raw_dir = Path(config["raw_dir"])
    logger = get_logger("download_datasets", Path(config["logs_dir"]) / "download_datasets.log")
    ensure_dir(raw_dir)

    for dataset in args.datasets:
        if dataset == "musique":
            _download_musique(raw_dir, logger)
        elif dataset == "hotpot":
            _download_hotpot(raw_dir, logger)

    logger.info("Dataset download step complete.")


if __name__ == "__main__":
    main()
