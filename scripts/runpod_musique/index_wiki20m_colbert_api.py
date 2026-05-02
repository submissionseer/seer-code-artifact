#!/usr/bin/env python3
"""
Build a ColBERT index via the Python API (avoids CLI/version incompatibilities).

Intended for large-scale RunPod builds (e.g., DPR psgs_w100 -> wiki20M index).
"""

from __future__ import annotations

import argparse
import inspect
import os
import sys
import time
from pathlib import Path


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--collection", required=True, help="Path to ColBERT collection.tsv")
    p.add_argument("--index-root", required=True, help="Directory to write index artifacts")
    p.add_argument("--index-name", required=True, help="Name of the ColBERT index")
    p.add_argument(
        "--checkpoint",
        default="colbert-ir/colbertv2.0",
        help="HF/Local ColBERT checkpoint (default: colbert-ir/colbertv2.0)",
    )
    p.add_argument("--doc-maxlen", type=int, default=180)
    p.add_argument("--nbits", type=int, default=2)
    p.add_argument("--nranks", type=int, default=1)
    p.add_argument(
        "--avoid-fork-if-possible",
        action="store_true",
        default=True,
        help="Set ColBERTConfig.avoid_fork_if_possible=True (recommended on some RunPod/FUSE envs)",
    )
    p.add_argument(
        "--experiment",
        default="musique_colbert_build",
        help="Run/experiment name under index root",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing index directory if present",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    collection = Path(args.collection)
    index_root = Path(args.index_root)
    index_root.mkdir(parents=True, exist_ok=True)

    if not collection.exists():
        log(f"ERROR: collection file not found: {collection}")
        return 2

    try:
        from colbert import Indexer
        from colbert.infra import ColBERTConfig, Run, RunConfig
        # Some ColBERT builds ignore avoid_fork_if_possible from configs and still call
        # Launcher.launch() (fork path), which can hang on RunPod/FUSE after plan.json.
        # Force the no-fork launcher path when requested.
        try:
            from colbert.infra import launcher as _launcher_mod  # type: ignore

            _Launcher = getattr(_launcher_mod, "Launcher", None)
            if _Launcher is not None and hasattr(_Launcher, "launch") and hasattr(_Launcher, "launch_without_fork"):
                _orig_launch = _Launcher.launch

                def _forced_no_fork_launch(self, *args, **kwargs):  # noqa: ANN001
                    return self.launch_without_fork(*args, **kwargs)

                if args.avoid_fork_if_possible:
                    _Launcher.launch = _forced_no_fork_launch
                    log("launcher monkeypatch applied: Launcher.launch -> launch_without_fork")
                else:
                    log("launcher monkeypatch skipped (avoid_fork_if_possible=False)")
            else:
                log("launcher monkeypatch unavailable: Launcher.launch/launch_without_fork not found")
        except Exception as _mp_e:
            log(f"WARNING: launcher monkeypatch failed: {_mp_e!r}")
    except Exception as e:
        log(f"ERROR: failed to import ColBERT APIs: {e!r}")
        return 3

    # Best-effort sanity info before long run.
    try:
        import torch

        log(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            log(f"cuda_device={torch.cuda.get_device_name(0)}")
    except Exception as e:  # pragma: no cover
        log(f"WARNING: torch sanity check failed: {e!r}")

    log(f"collection={collection}")
    log(f"collection_size_bytes={collection.stat().st_size}")
    log(f"index_root={index_root}")
    log(f"index_name={args.index_name}")
    log(f"checkpoint={args.checkpoint}")
    log(
        f"config: nbits={args.nbits} doc_maxlen={args.doc_maxlen} nranks={args.nranks} "
        f"experiment={args.experiment} overwrite={args.overwrite} "
        f"avoid_fork_if_possible={args.avoid_fork_if_possible}"
    )

    start = time.time()
    try:
        # Apply no-fork on both RunConfig and ColBERTConfig when supported. Some ColBERT
        # versions read this from one config layer but not the other.
        run_kwargs = dict(nranks=args.nranks, experiment=args.experiment, root=str(index_root))
        if args.avoid_fork_if_possible:
            try:
                if "avoid_fork_if_possible" in inspect.signature(RunConfig).parameters:
                    run_kwargs["avoid_fork_if_possible"] = True
            except Exception:
                pass
        log(f"run_config_kwargs={run_kwargs}")
        with Run().context(RunConfig(**run_kwargs)):
            config_kwargs = dict(nbits=args.nbits, doc_maxlen=args.doc_maxlen)
            if args.avoid_fork_if_possible:
                # Some ColBERT builds can hang after plan creation on certain environments
                # when multiprocessing/fork is enabled; expose an explicit no-fork path.
                try:
                    if "avoid_fork_if_possible" in inspect.signature(ColBERTConfig).parameters:
                        config_kwargs["avoid_fork_if_possible"] = True
                    else:
                        config_kwargs["avoid_fork_if_possible"] = True
                except Exception:
                    config_kwargs["avoid_fork_if_possible"] = True
            log(f"colbert_config_kwargs={config_kwargs}")
            config = ColBERTConfig(**config_kwargs)
            try:
                log(f"resolved flags: run.avoid_fork_if_possible={getattr(RunConfig(**run_kwargs), 'avoid_fork_if_possible', 'NA')} colbert.avoid_fork_if_possible={getattr(config, 'avoid_fork_if_possible', 'NA')}")
            except Exception:
                pass
            indexer = Indexer(checkpoint=args.checkpoint, config=config)
            # ColBERT's Indexer rebuilds config from checkpoint+Run().config and may reset
            # avoid_fork_if_possible to the checkpoint/default value. Force it here, because
            # __launch() checks indexer.config.avoid_fork_if_possible.
            if args.avoid_fork_if_possible:
                try:
                    indexer.configure(avoid_fork_if_possible=True)
                except Exception:
                    try:
                        setattr(indexer.config, "avoid_fork_if_possible", True)
                    except Exception:
                        pass
            try:
                log(
                    "post-indexer flags: "
                    f"indexer.config.avoid_fork_if_possible={getattr(indexer.config, 'avoid_fork_if_possible', 'NA')}"
                )
            except Exception:
                pass
            log("starting indexer.index(...)")
            # API signature differs across versions; keep kwargs minimal and common.
            indexer.index(
                name=args.index_name,
                collection=str(collection),
                overwrite=args.overwrite,
            )
    except TypeError as e:
        log(f"ERROR: ColBERT API signature mismatch during indexing: {e!r}")
        log("Try rerunning without --overwrite or adjust script for this ColBERT version.")
        return 4
    except Exception as e:
        log(f"ERROR: indexing failed: {e!r}")
        return 5

    elapsed = time.time() - start
    log(f"DONE elapsed_hours={elapsed/3600:.2f} elapsed_seconds={elapsed:.1f}")

    # Show resulting index path(s) to make follow-up serving easier.
    try:
        children = sorted(index_root.iterdir())
        if children:
            log("index_root_contents:")
            for c in children:
                kind = "dir" if c.is_dir() else "file"
                try:
                    size = c.stat().st_size
                except OSError:
                    size = -1
                log(f"  - {c.name} [{kind}] size={size}")
    except Exception as e:  # pragma: no cover
        log(f"WARNING: failed to list index root contents: {e!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
