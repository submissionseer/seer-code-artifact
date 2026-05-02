#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml
try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    def load_dotenv() -> None:
        return None
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **_: Any):
        return iterable

from seer.openrouter import OpenRouterConfig
from seer.scoring import (
    generate_requirements_map,
    load_requirements_map,
    score_mmr_and_binary_candidates_batch,
)
from seer.util_io import read_jsonl, write_json, write_jsonl

load_dotenv()


def _requirements_for_row(
    *,
    row: dict[str, Any],
    requirements_map: dict[str, list[str]],
) -> tuple[list[str], str]:
    qid = str(row.get("qid", ""))
    question = str(row.get("question", ""))
    reqs = requirements_map.get(qid)
    if reqs:
        return list(reqs), "qid"
    reqs = requirements_map.get(question)
    if reqs:
        return list(reqs), "question"
    return [], "missing"


def _build_cumulative_candidates(
    row: dict[str, Any],
) -> tuple[list[int], list[list[dict[str, Any]]]]:
    by_hop: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for p in row.get("retrieved", []) or []:
        if not isinstance(p, dict):
            continue
        hop = int(p.get("hop", 1) or 1)
        by_hop[hop].append(p)

    if not by_hop:
        return [], []

    hops = sorted(by_hop.keys())
    cumulative_candidates: list[list[dict[str, Any]]] = []
    seen: set[tuple[str, str]] = set()
    running: list[dict[str, Any]] = []
    for hop in hops:
        for p in by_hop[hop]:
            key = (str(p.get("title", "")), str(p.get("text", "")))
            if key in seen:
                continue
            seen.add(key)
            running.append(
                {
                    "title": str(p.get("title", "")),
                    "text": str(p.get("text", "")),
                }
            )
        cumulative_candidates.append(list(running))
    return hops, cumulative_candidates


def _load_judges_openrouter_cfg(requirements_concurrency: int) -> OpenRouterConfig:
    judges_cfg = yaml.safe_load(Path("configs/judges.yaml").read_text())
    openrouter_cfg = judges_cfg["openrouter"]
    api_key = os.getenv(openrouter_cfg["api_key_env"], "")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required to auto-generate requirements")
    return OpenRouterConfig(
        base_url=openrouter_cfg["base_url"],
        api_key=api_key,
        concurrency=requirements_concurrency,
        temperature=0.0,
    )


def _load_judges_cfg() -> dict[str, Any]:
    return yaml.safe_load(Path("configs/judges.yaml").read_text())


async def _score_one_row(
    *,
    row: dict[str, Any],
    requirements: list[str],
    jina_api_key: str,
    jina_cache_dir: Path,
    jina_model: str,
    jina_concurrency: int,
    decomp_threshold: float,
) -> tuple[list[float], list[float]]:
    _, candidates = _build_cumulative_candidates(row)
    if not candidates or not requirements:
        return [0.0] * len(candidates), [0.0] * len(candidates)

    mmr_scores, binary_scores = await score_mmr_and_binary_candidates_batch(
        requirements=requirements,
        candidates_passages=candidates,
        jina_api_key=jina_api_key,
        cache_dir=jina_cache_dir,
        jina_model=jina_model,
        concurrency=jina_concurrency,
        threshold=decomp_threshold,
    )
    return list(mmr_scores), list(binary_scores)


def main() -> None:
    ap = argparse.ArgumentParser(description="Post-score eval rollout JSONL with MMR/decomp-binary scores.")
    ap.add_argument("--input", required=True, help="Eval rollout JSONL path.")
    ap.add_argument("--output", required=True, help="Scored JSONL output path.")
    ap.add_argument("--requirements", default=None, help="Optional requirements JSON map keyed by qid/question.")
    ap.add_argument("--auto-requirements-out", default=None, help="Where to write generated requirements map.")
    ap.add_argument("--requirements-model", default="openai/gpt-4o-mini")
    ap.add_argument("--requirements-concurrency", type=int, default=20)
    ap.add_argument("--jina-model", default="jina-reranker-v3")
    ap.add_argument("--jina-concurrency", type=int, default=16)
    ap.add_argument("--jina-cache-dir", default=None)
    ap.add_argument("--decomp-threshold", type=float, default=0.5)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate inputs, config, and external-service setup without scoring rows.",
    )
    args = ap.parse_args()

    rows = read_jsonl(args.input)
    if args.offset:
        rows = rows[args.offset :]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError("No rows selected from input")

    requirements_map: dict[str, list[str]] = {}
    if args.requirements:
        req_path = Path(args.requirements)
        if not req_path.exists():
            raise FileNotFoundError(f"Requirements file not found: {req_path}")
        requirements_map = load_requirements_map(req_path)

    missing_qid_to_question: dict[str, str] = {}
    for row in rows:
        reqs, _ = _requirements_for_row(row=row, requirements_map=requirements_map)
        if reqs:
            continue
        qid = str(row.get("qid", ""))
        question = str(row.get("question", "")).strip()
        if qid and question:
            missing_qid_to_question[qid] = question

    auto_req_path = Path(
        args.auto_requirements_out
        or (Path(args.output).with_suffix("").as_posix() + ".requirements.json")
    )
    if missing_qid_to_question:
        print(f"Auto-generating requirements for {len(missing_qid_to_question)} questions...", flush=True)
        if args.validate_only:
            print(f"Would write generated requirements to: {auto_req_path}", flush=True)
        else:
            cfg = _load_judges_openrouter_cfg(args.requirements_concurrency)
            generated = asyncio.run(
                generate_requirements_map(
                    missing_qid_to_question,
                    model=args.requirements_model,
                    openrouter_config=cfg,
                    cache_dir=auto_req_path.parent / f"{auto_req_path.stem}.cache",
                    output_path=auto_req_path,
                )
            )
            requirements_map.update(generated)
            print(f"Requirements written: {auto_req_path}", flush=True)

    jina_api_key = os.getenv("JINA_AI_API_KEY", "")
    if not jina_api_key and not args.validate_only:
        raise RuntimeError("JINA_AI_API_KEY is required")

    out_path = Path(args.output)
    jina_cache_dir = Path(args.jina_cache_dir or out_path.with_suffix(".jina_cache"))
    if not args.validate_only:
        jina_cache_dir.mkdir(parents=True, exist_ok=True)

    if args.validate_only:
        if missing_qid_to_question:
            judges_cfg = _load_judges_cfg()
            key_env = judges_cfg["openrouter"]["api_key_env"]
            print(f"OpenRouter key env {key_env}: {'set' if os.getenv(key_env) else 'missing'}", flush=True)
        print(f"JINA_AI_API_KEY: {'set' if os.getenv('JINA_AI_API_KEY') else 'missing'}", flush=True)
        print("Validation-only check completed.", flush=True)
        return

    scored_rows: list[dict[str, Any]] = []
    missing_requirements = 0
    for row in tqdm(rows, desc="Scoring eval rollouts"):
        reqs, req_source = _requirements_for_row(row=row, requirements_map=requirements_map)
        if not reqs:
            missing_requirements += 1
        hops, _ = _build_cumulative_candidates(row)

        mmr_scores, binary_scores = asyncio.run(
            _score_one_row(
                row=row,
                requirements=reqs,
                jina_api_key=jina_api_key,
                jina_cache_dir=jina_cache_dir,
                jina_model=args.jina_model,
                jina_concurrency=args.jina_concurrency,
                decomp_threshold=args.decomp_threshold,
            )
        )

        out = dict(row)
        out["mmr_num_requirements"] = len(reqs)
        out["mmr_requirements_source"] = req_source
        out["mmr_ap_final"] = float(mmr_scores[-1]) if mmr_scores else 0.0
        out["decomp_binary_ap_final"] = float(binary_scores[-1]) if binary_scores else 0.0

        hop_scores = {hop: (mmr_scores[i], binary_scores[i]) for i, hop in enumerate(hops)}
        hop_details = []
        for detail in out.get("hop_details", []) or []:
            if not isinstance(detail, dict):
                hop_details.append(detail)
                continue
            hop_num = int(detail.get("hop", 0) or 0)
            mmr_ap, decomp_ap = hop_scores.get(hop_num, (None, None))
            updated = dict(detail)
            if mmr_ap is not None:
                updated["mmr_ap_after_hop"] = float(mmr_ap)
            if decomp_ap is not None:
                updated["decomp_binary_ap_after_hop"] = float(decomp_ap)
            hop_details.append(updated)
        out["hop_details"] = hop_details
        scored_rows.append(out)

    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    write_json(meta_path, {"meta": {
        "input": str(Path(args.input)),
        "output": str(Path(args.output)),
        "rows_scored": len(scored_rows),
        "missing_requirements": missing_requirements,
        "requirements_model": args.requirements_model,
        "jina_model": args.jina_model,
        "jina_cache_dir": str(jina_cache_dir),
        "decomp_threshold": args.decomp_threshold,
    }})
    # scored output jsonl
    jsonl_output = out_path if out_path.suffix == ".jsonl" else out_path.with_suffix(".jsonl")
    write_jsonl(jsonl_output, scored_rows)

    print(f"Scored rows: {len(scored_rows)}", flush=True)
    print(f"Missing requirements: {missing_requirements}", flush=True)
    print(f"Wrote metadata: {meta_path}", flush=True)
    print(f"Wrote scored output: {jsonl_output}", flush=True)


if __name__ == "__main__":
    main()
