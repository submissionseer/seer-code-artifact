#!/usr/bin/env python3
"""Canonical DPO rollout scorer (Seer / MMR / decomp-binary / raw-Jina)."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import statistics
from pathlib import Path
from typing import Any

import httpx
import yaml
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

from seer.dpo_utils import iter_hop_records, selected_context_docs_upto_hop
from seer.openrouter import OpenRouterConfig
from seer.scoring import (
    load_requirements_map,
    score_decomp_binary_candidates_batch,
    score_mmr_candidates_batch,
    score_seer_candidates_batch,
)
from seer.util_io import ensure_dir, read_jsonl, write_jsonl


PassageKey = tuple[str, str]


def _requirements_for_rollout(
    *,
    rollout: dict[str, Any],
    requirements: dict[str, list[str]],
) -> list[str]:
    qid = str(rollout.get("qid", ""))
    question = str(rollout.get("question", ""))
    reqs = requirements.get(qid)
    if reqs:
        return reqs
    return list(requirements.get(question, []))


def _cache_key(model: str, query: str, documents: list[str]) -> str:
    parts = [model, query, *documents]
    return hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()


def _passage_key(passage: dict[str, Any]) -> PassageKey:
    return (
        str(passage.get("title", "No Title")),
        str(passage.get("text", "")),
    )


def _passage_doc_text(key: PassageKey) -> str:
    title, text = key
    return f"{title} — {text}"


async def _jina_rerank_raw(
    *,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    api_key: str,
    model: str,
    query: str,
    documents: list[str],
    cache_dir: Path,
) -> list[dict[str, Any]]:
    key = _cache_key(model, query, documents)
    cache_path = cache_dir / f"{key}.json"

    if cache_path.exists():
        try:
            with cache_path.open() as f:
                return json.load(f)
        except Exception:
            cache_path.unlink(missing_ok=True)

    async with semaphore:
        retries = 8
        for attempt in range(retries):
            try:
                response = await client.post(
                    "https://api.jina.ai/v1/rerank",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "query": query,
                        "documents": documents,
                        "return_documents": False,
                    },
                    timeout=60,
                )
                if response.status_code in {429, 500, 502, 503, 504} and attempt < retries - 1:
                    wait = (2**attempt) + random.random()
                    if response.status_code == 429:
                        wait = max(wait, 10)
                    await asyncio.sleep(wait)
                    continue
                response.raise_for_status()
                results = response.json().get("results", [])
                with cache_path.open("w") as f:
                    json.dump(results, f, separators=(",", ":"))
                return results
            except httpx.TimeoutException:
                if attempt < retries - 1:
                    await asyncio.sleep((2**attempt) + random.random())
                    continue
                raise

    raise RuntimeError("Jina rerank failed after retries")


def build_rerank_tasks(
    rollouts: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build deduplicated (qid, hop) rerank tasks with candidate passage mappings."""
    tasks: list[dict[str, Any]] = []
    passage_maps: dict[tuple[str, int], dict[PassageKey, int]] = {}
    candidate_passage_map: dict[tuple[str, int, int], list[PassageKey]] = {}

    for rollout in rollouts:
        qid = str(rollout.get("qid", ""))
        question = str(rollout.get("question", ""))
        if not qid or not question:
            continue

        for hop_num, hop_record in iter_hop_records(rollout):
            candidates = list(hop_record.get("candidates", []) or [])
            if not candidates:
                continue

            shared_context = (
                []
                if hop_num == 1
                else selected_context_docs_upto_hop(rollout, hop_num)
            )
            unique_passages: dict[PassageKey, str] = {}

            for ci, candidate in enumerate(candidates):
                cand_passages = list(candidate.get("retrieved", []) or [])
                all_passages = [*shared_context, *cand_passages]
                keys: list[PassageKey] = []
                for passage in all_passages:
                    key = _passage_key(passage)
                    keys.append(key)
                    if key not in unique_passages:
                        unique_passages[key] = _passage_doc_text(key)
                candidate_passage_map[(qid, hop_num, ci)] = keys

            if not unique_passages:
                continue

            unique_keys = list(unique_passages.keys())
            unique_docs = [unique_passages[k] for k in unique_keys]
            passage_maps[(qid, hop_num)] = {k: i for i, k in enumerate(unique_keys)}
            tasks.append(
                {
                    "qid": qid,
                    "hop": hop_num,
                    "query": question,
                    "documents": unique_docs,
                }
            )

    return tasks, {
        "passage_maps": passage_maps,
        "candidate_passage_map": candidate_passage_map,
    }


async def run_jina_scoring(
    *,
    tasks: list[dict[str, Any]],
    api_key: str,
    model: str,
    cache_dir: Path,
    concurrency: int,
    batch_size: int,
) -> dict[tuple[str, int], dict[int, float]]:
    semaphore = asyncio.Semaphore(max(1, int(concurrency)))
    score_maps: dict[tuple[str, int], dict[int, float]] = {}

    async with httpx.AsyncClient() as client:
        pbar = tqdm(total=len(tasks), desc="Jina reranking (raw-query)")

        async def process_task(task: dict[str, Any]) -> None:
            results = await _jina_rerank_raw(
                client=client,
                semaphore=semaphore,
                api_key=api_key,
                model=model,
                query=str(task["query"]),
                documents=list(task["documents"]),
                cache_dir=cache_dir,
            )
            score_maps[(str(task["qid"]), int(task["hop"]))] = {
                int(item["index"]): float(item["relevance_score"])
                for item in results
                if "index" in item
            }
            pbar.update(1)

        task_batch_size = max(1, int(batch_size))
        for i in range(0, len(tasks), task_batch_size):
            await asyncio.gather(*[process_task(t) for t in tasks[i : i + task_batch_size]])

        pbar.close()

    return score_maps


def compute_candidate_scores(
    *,
    rollouts: list[dict[str, Any]],
    score_maps: dict[tuple[str, int], dict[int, float]],
    passage_maps: dict[tuple[str, int], dict[PassageKey, int]],
    candidate_passage_map: dict[tuple[str, int, int], list[PassageKey]],
    topk_mean: int,
) -> dict[tuple[str, int, int], dict[str, float]]:
    out: dict[tuple[str, int, int], dict[str, float]] = {}
    k_top = max(1, int(topk_mean))

    for rollout in rollouts:
        qid = str(rollout.get("qid", ""))
        if not qid:
            continue

        for hop_num, hop_record in iter_hop_records(rollout):
            candidates = list(hop_record.get("candidates", []) or [])
            if not candidates:
                continue

            score_map = score_maps.get((qid, hop_num), {})
            pmap = passage_maps.get((qid, hop_num), {})

            for ci, _candidate in enumerate(candidates):
                keys = candidate_passage_map.get((qid, hop_num, ci), [])
                vals: list[float] = []
                for pkey in keys:
                    idx = pmap.get(pkey)
                    if idx is not None and idx in score_map:
                        vals.append(score_map[idx])

                if not vals:
                    out[(qid, hop_num, ci)] = {
                        "raw_jina_maxmean_top3": 0.0,
                        "raw_jina_max": 0.0,
                    }
                    continue

                vals_sorted = sorted(vals, reverse=True)
                k = min(k_top, len(vals_sorted))
                out[(qid, hop_num, ci)] = {
                    "raw_jina_maxmean_top3": float(statistics.mean(vals_sorted[:k])),
                    "raw_jina_max": float(vals_sorted[0]),
                }

    return out


def merge_scores_into_rollouts(
    *,
    rollouts: list[dict[str, Any]],
    cand_scores: dict[tuple[str, int, int], dict[str, float]],
) -> int:
    total = 0
    for rollout in rollouts:
        qid = str(rollout.get("qid", ""))
        if not qid:
            continue

        for hop_num, hop_record in iter_hop_records(rollout):
            candidates = list(hop_record.get("candidates", []) or [])
            for ci, candidate in enumerate(candidates):
                score = cand_scores.get((qid, hop_num, ci))
                if not score:
                    continue
                candidate["raw_jina_maxmean_top3"] = round(score["raw_jina_maxmean_top3"], 6)
                candidate["raw_jina_max"] = round(score["raw_jina_max"], 6)
                total += 1

            # Preserve legacy hop{n} keys for compatibility with historical tooling.
            rollout[f"hop{hop_num}"] = hop_record
    return total


async def _score_rollouts(
    *,
    rollouts: list[dict[str, Any]],
    mode: str,
    requirements: dict[str, list[str]] | None,
    seer_model: str,
    seer_prompt_style: str,
    seer_openrouter_cfg: OpenRouterConfig | None,
    seer_cache_dir: Path | None,
    jina_api_key: str | None,
    jina_model: str,
    jina_cache_dir: Path | None,
    concurrency: int,
    decomp_threshold: float,
) -> tuple[int, int]:
    scored_candidates = 0
    missing_requirements = 0

    for rollout in tqdm(rollouts, desc=f"dpo_score[{mode}]"):
        question = str(rollout.get("question", ""))
        reqs: list[str] | None = None

        if mode in {"mmr", "decomp_binary"}:
            if requirements is None:
                raise RuntimeError("requirements map missing in mmr/decomp_binary mode")
            reqs = _requirements_for_rollout(rollout=rollout, requirements=requirements)
            if not reqs:
                missing_requirements += 1
                continue

        for hop_num, hop_record in iter_hop_records(rollout):
            candidates = list(hop_record.get("candidates", []) or [])
            if not candidates:
                continue

            shared_context = (
                []
                if hop_num == 1
                else selected_context_docs_upto_hop(rollout, hop_num)
            )
            candidate_passages = [
                (list(shared_context) if hop_num > 1 else [])
                + list(candidate.get("retrieved", []) or [])
                for candidate in candidates
            ]

            if mode == "seer":
                if seer_openrouter_cfg is None or seer_cache_dir is None:
                    raise RuntimeError("seer scorer requires OpenRouter configuration")
                scores = await score_seer_candidates_batch(
                    question=question,
                    candidates_passages=candidate_passages,
                    model=seer_model,
                    prompt_style=seer_prompt_style,
                    openrouter_config=seer_openrouter_cfg,
                    cache_dir=seer_cache_dir,
                    request_meta={"qid": rollout.get("qid"), "hop": hop_num},
                )
                for candidate, score in zip(candidates, scores):
                    candidate["seer_ap"] = float(score)
                    scored_candidates += 1
                continue

            if mode == "mmr":
                if not jina_api_key or jina_cache_dir is None:
                    raise RuntimeError("mmr scorer requires Jina configuration")
                scores = await score_mmr_candidates_batch(
                    requirements=reqs or [],
                    candidates_passages=candidate_passages,
                    jina_api_key=jina_api_key,
                    cache_dir=jina_cache_dir,
                    jina_model=jina_model,
                    concurrency=concurrency,
                )
                for candidate, score in zip(candidates, scores):
                    candidate["mmr_ap"] = float(score)
                    scored_candidates += 1
                continue

            if mode == "decomp_binary":
                if not jina_api_key or jina_cache_dir is None:
                    raise RuntimeError("decomp_binary scorer requires Jina configuration")
                scores = await score_decomp_binary_candidates_batch(
                    requirements=reqs or [],
                    candidates_passages=candidate_passages,
                    jina_api_key=jina_api_key,
                    cache_dir=jina_cache_dir,
                    jina_model=jina_model,
                    concurrency=concurrency,
                    threshold=decomp_threshold,
                )
                for candidate, score in zip(candidates, scores):
                    candidate["decomp_binary_ap"] = float(score)
                    scored_candidates += 1
                continue

            raise ValueError(f"Unsupported scoring mode: {mode}")

    return scored_candidates, missing_requirements


def main() -> None:
    parser = argparse.ArgumentParser(description="Score DPO rollout candidates")
    parser.add_argument("--input", required=True, help="Input DPO rollouts JSONL")
    parser.add_argument("--output", required=True, help="Output JSONL with scored candidates")
    parser.add_argument(
        "--mode",
        default="mmr",
        choices=["seer", "mmr", "decomp_binary", "raw_jina", "raw_jina_maxmean", "raw_jina_max"],
        help="Candidate scoring mode",
    )
    parser.add_argument(
        "--requirements",
        default=None,
        help="Requirements JSON (required for mmr/decomp_binary)",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Raw-Jina task batch size for asyncio.gather()",
    )

    parser.add_argument("--seer-model", default="openai/gpt-4o-mini")
    parser.add_argument(
        "--seer-prompt-style",
        default="xml_fewshot",
        choices=[
            "xml",
            "xml_strict",
            "xml_fewshot",
            "xml_v2",
            "xml_v2b",
            "xml_v3_hybrid",
            "xml_optimized",
        ],
    )
    parser.add_argument(
        "--seer-cache-dir",
        default="data/cache/seer_dpo_labels",
        help="OpenRouter cache dir for Seer scoring",
    )

    parser.add_argument(
        "--jina-model",
        "--model",
        dest="jina_model",
        default="jina-reranker-v3",
    )
    parser.add_argument(
        "--jina-cache-dir",
        "--cache-dir",
        dest="jina_cache_dir",
        default=None,
        help="Jina reranker cache dir (mode-specific default if omitted)",
    )
    parser.add_argument("--decomp-threshold", type=float, default=0.5)
    parser.add_argument(
        "--topk-mean",
        type=int,
        default=3,
        help="Top-k passages to average for raw_jina_maxmean_top3",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate inputs, config, and external-service setup without scoring candidates.",
    )
    args = parser.parse_args()

    mode = args.mode
    if mode in {"raw_jina_maxmean", "raw_jina_max"}:
        mode = "raw_jina"

    rollouts = read_jsonl(args.input)
    if args.limit is not None:
        rollouts = rollouts[: args.limit]
    print(f"Loaded rollouts: {len(rollouts)}")

    requirements: dict[str, list[str]] | None = None
    if mode in {"mmr", "decomp_binary"}:
        if not args.requirements:
            raise ValueError("--requirements is required for mmr/decomp_binary modes")
        requirements = load_requirements_map(args.requirements)
        print(f"Loaded requirements entries: {len(requirements)}")

    seer_cfg: OpenRouterConfig | None = None
    seer_cache_dir: Path | None = None
    if mode == "seer":
        judges_cfg = yaml.safe_load(Path("configs/judges.yaml").read_text())
        seer_cfg = OpenRouterConfig(
            base_url=judges_cfg["openrouter"]["base_url"],
            api_key=os.environ.get(judges_cfg["openrouter"]["api_key_env"], ""),
            concurrency=args.concurrency,
            temperature=0.0,
        )
        if not seer_cfg.api_key and not args.validate_only:
            raise RuntimeError("OPENROUTER_API_KEY not set")
        seer_cache_dir = Path(args.seer_cache_dir)
        if not args.validate_only:
            ensure_dir(seer_cache_dir)

    jina_api_key: str | None = None
    jina_cache_dir: Path | None = None
    if mode in {"mmr", "decomp_binary", "raw_jina"}:
        jina_api_key = os.environ.get("JINA_AI_API_KEY", "")
        if not jina_api_key and not args.validate_only:
            raise RuntimeError("JINA_AI_API_KEY not set")
        default_jina_cache = (
            "data/dpo_jina_cache_raw_query" if mode == "raw_jina" else "data/dpo_jina_cache"
        )
        jina_cache_dir = Path(args.jina_cache_dir or default_jina_cache)
        if not args.validate_only:
            ensure_dir(jina_cache_dir)

    if args.validate_only:
        print(f"Mode: {args.mode} (canonical={mode})")
        if mode == "seer":
            judges_cfg = yaml.safe_load(Path("configs/judges.yaml").read_text())
            key_env = judges_cfg["openrouter"]["api_key_env"]
            print(f"OpenRouter key env {key_env}: {'set' if os.environ.get(key_env) else 'missing'}")
        if mode in {"mmr", "decomp_binary", "raw_jina"}:
            print(f"JINA_AI_API_KEY: {'set' if os.environ.get('JINA_AI_API_KEY') else 'missing'}")
        print("Validation-only check completed.")
        return

    raw_topk_scores: list[float] = []
    raw_max_scores: list[float] = []
    if mode == "raw_jina":
        print("Building raw-query rerank tasks ...")
        tasks, meta = build_rerank_tasks(rollouts)
        print(f"Raw-query tasks: {len(tasks)}")

        if tasks:
            score_maps = asyncio.run(
                run_jina_scoring(
                    tasks=tasks,
                    api_key=jina_api_key or "",
                    model=args.jina_model,
                    cache_dir=jina_cache_dir or Path("data/dpo_jina_cache_raw_query"),
                    concurrency=args.concurrency,
                    batch_size=args.batch_size,
                )
            )
        else:
            score_maps = {}

        cand_scores = compute_candidate_scores(
            rollouts=rollouts,
            score_maps=score_maps,
            passage_maps=meta["passage_maps"],
            candidate_passage_map=meta["candidate_passage_map"],
            topk_mean=args.topk_mean,
        )
        scored_candidates = merge_scores_into_rollouts(
            rollouts=rollouts,
            cand_scores=cand_scores,
        )
        missing_requirements = 0

        raw_topk_scores = [v["raw_jina_maxmean_top3"] for v in cand_scores.values()]
        raw_max_scores = [v["raw_jina_max"] for v in cand_scores.values()]
    else:
        scored_candidates, missing_requirements = asyncio.run(
            _score_rollouts(
                rollouts=rollouts,
                mode=mode,
                requirements=requirements,
                seer_model=args.seer_model,
                seer_prompt_style=args.seer_prompt_style,
                seer_openrouter_cfg=seer_cfg,
                seer_cache_dir=seer_cache_dir,
                jina_api_key=jina_api_key,
                jina_model=args.jina_model,
                jina_cache_dir=jina_cache_dir,
                concurrency=args.concurrency,
                decomp_threshold=args.decomp_threshold,
            )
        )

    for rollout in rollouts:
        for hop_num, hop_record in iter_hop_records(rollout):
            rollout[f"hop{hop_num}"] = hop_record

    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    write_jsonl(output_path, rollouts)

    print(f"Mode: {args.mode} (canonical={mode})")
    print(f"Scored candidates: {scored_candidates}")
    if mode in {"mmr", "decomp_binary"}:
        print(f"Rollouts missing requirements: {missing_requirements}")
    if mode == "raw_jina" and raw_topk_scores:
        print(
            f"raw_jina_maxmean_top{args.topk_mean}: "
            f"mean={statistics.mean(raw_topk_scores):.4f} "
            f"median={statistics.median(raw_topk_scores):.4f}"
        )
        print(
            "raw_jina_max: "
            f"mean={statistics.mean(raw_max_scores):.4f} "
            f"median={statistics.median(raw_max_scores):.4f}"
        )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
