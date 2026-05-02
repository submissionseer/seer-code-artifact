#!/usr/bin/env python3
"""Canonical SFT training entrypoint (Axolotl)."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SFT training via Axolotl.")
    parser.add_argument("--config", required=True, help="Axolotl YAML config path")
    parser.add_argument(
        "--workdir",
        default=None,
        help="Optional working directory for launch (defaults to current directory).",
    )
    parser.add_argument(
        "--accelerate-bin",
        default="accelerate",
        help="Accelerate executable (default: accelerate).",
    )
    parser.add_argument(
        "--module",
        default="axolotl.cli.train",
        help="Training module (default: axolotl.cli.train).",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Extra argument appended to training command (repeatable).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print command only; do not execute.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cmd = [
        args.accelerate_bin,
        "launch",
        "-m",
        args.module,
        str(config_path),
        *args.extra_arg,
    ]
    print("SFT train command:")
    print(shlex.join(cmd))

    if args.dry_run:
        return

    subprocess.run(cmd, check=True, cwd=args.workdir)


if __name__ == "__main__":
    main()
