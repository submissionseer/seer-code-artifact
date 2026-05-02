#!/usr/bin/env python3
"""Download Hotpot fullwiki questions and the prebuilt ColBERT wiki2017 index."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import yaml

from seer.util_io import ensure_dir
from seer.util_log import get_logger


COLBERT_INDEX_REPO = "sher222/ColBERTv2-wiki2017-index"

FULLWIKI_TRAIN_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json"
FULLWIKI_DEV_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json"
FULLWIKI_TEST_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_test_fullwiki_v1.json"


def download_colbert_index(index_dir: Path, logger) -> None:
    """Download the pre-built ColBERTv2 Wikipedia index from Hugging Face."""
    if (index_dir / "wiki2017").exists():
        logger.info("ColBERTv2 index already exists at %s/wiki2017", index_dir)
        return

    ensure_dir(index_dir)

    logger.info("Downloading ColBERTv2 Wikipedia index from HuggingFace...")
    logger.info("This is ~33GB and may take a while depending on your connection.")
    logger.info("Repository: %s", COLBERT_INDEX_REPO)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error("huggingface_hub not installed. Installing...")
        subprocess.run(["uv", "pip", "install", "huggingface_hub"], check=True)
        from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=COLBERT_INDEX_REPO,
        local_dir=index_dir / "wiki2017",
        repo_type="model",
        local_dir_use_symlinks=False,
    )
    logger.info("Download complete!")


def download_fullwiki_questions(raw_dir: Path, logger) -> None:
    """Download the HotpotQA train and fullwiki dev question files."""
    import json
    from urllib.request import urlretrieve

    fullwiki_dir = raw_dir / "hotpot_fullwiki"
    ensure_dir(fullwiki_dir)

    train_path = fullwiki_dir / "hotpot_train_v1.1.json"
    if train_path.exists():
        logger.info("HotpotQA train set already exists")
    else:
        logger.info("Downloading HotpotQA train set (~535MB)...")
        urlretrieve(FULLWIKI_TRAIN_URL, train_path)
        with open(train_path) as handle:
            data = json.load(handle)
        logger.info("Downloaded %d train questions", len(data))

    dev_path = fullwiki_dir / "hotpot_dev_fullwiki_v1.json"
    if dev_path.exists():
        logger.info("HotpotQA fullwiki dev set already exists")
    else:
        logger.info("Downloading HotpotQA fullwiki dev set...")
        urlretrieve(FULLWIKI_DEV_URL, dev_path)
        with open(dev_path) as handle:
            data = json.load(handle)
        logger.info("Downloaded %d dev questions", len(data))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--skip-index", action="store_true", help="Skip downloading the large index")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    raw_dir = Path(config["raw_dir"])
    logs_dir = Path(config["logs_dir"])
    logger = get_logger("download_colbert", logs_dir / "download_colbert.log")

    download_fullwiki_questions(raw_dir, logger)

    if args.skip_index:
        logger.info("Skipping index download (--skip-index)")
    else:
        index_dir = raw_dir / "colbert_index"
        download_colbert_index(index_dir, logger)

    logger.info("Done!")


if __name__ == "__main__":
    main()
