#!/usr/bin/env python3
"""Canonical entrypoint for environment checks."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from seer.util_log import get_logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    logs_dir = Path(config["logs_dir"])
    logger = get_logger("setup", logs_dir / "setup_env_check.log")
    logger.info("Environment check complete.")


if __name__ == "__main__":
    main()
