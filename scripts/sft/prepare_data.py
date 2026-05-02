#!/usr/bin/env python3
"""
Prepare SFT training data from rollout JSONL using a configurable selector.

Canonical selectors:
- gold: select best variant per (base_qid, hop) by coverage_after_hop.
- seer: select best variant per (base_qid, hop) by Seer AP.
- mmr: select best variant per (base_qid, hop) by Jina MMR score.
- decomp_binary: select best variant per (base_qid, hop) by decomposed binary AP.
- raw_jina_maxmean / raw_jina_max: select best variant per (base_qid, hop) by raw
  cross-encoder question-to-passages score. These selectors require score file input
  and do not run scoring inline.

For selector in {mmr, decomp_binary}:
- If --scores is provided, use it directly.
- If --scores is omitted, this script runs requirements generation + Jina scoring
  in-process and then prepares selector-based SFT data.

For selector == seer:
- If --scores is provided, use it directly.
- If --scores is omitted, this script builds Seer AP scores from existing
  `seer_label` fields and saves them; for rows missing `seer_label`, it can
  optionally call Seer scoring in-process.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import httpx
from tqdm import tqdm
from tqdm.asyncio import tqdm as tqdm_async

from seer.openrouter import OpenRouterConfig, normalize_model_slug, run_completions
from seer.parse_seer_xml import SeerParseResult
from seer.scoring import (
    REQ_SYSTEM_PROMPT,
    REQ_USER_TEMPLATE,
    decomp_binary_ap_from_rollout,
    parse_requirements as shared_parse_requirements,
    score_seer_candidates_batch,
    seer_ap_from_rollout_xml,
)
from seer.seer_metrics import compute_seer_metrics
from seer.util_hash import sha256_text
from seer.util_io import read_json, write_json, write_jsonl

DEFAULT_INSTRUCTION = (
    "Generate a search query to find information needed to answer the following "
    "question. Use the provided context if available."
)
SEARCH_QUERY_MARKER = "[[ ## search_query ## ]]"
COMPLETED_MARKER = "[[ ## completed ## ]]"
MMR_DEFAULT_OPENROUTER_MODEL = "openai/gpt-4o-mini"
MMR_DEFAULT_RERANKER_MODEL = "jina-reranker-v3"
SEER_DEFAULT_MODEL = "openai/gpt-4o-mini"

NARRATIVE_PHRASES = (
    "there is no",
    "no information",
    "the context",
    "the provided context",
    "the question asks",
    "search query should",
    "however",
)


def format_dspy_output(query: str) -> str:
    return f"{SEARCH_QUERY_MARKER}\n{query}\n\n{COMPLETED_MARKER}"


def _normalize_query_text(candidate: str) -> str:
    query = str(candidate or "").strip()
    query = query.strip("`")
    query = query.strip('"').strip("'").strip()
    query = re.sub(r"\s+", " ", query).strip()
    return query


def _looks_narrative_like(query: str) -> bool:
    text = str(query or "").strip().lower()
    if not text:
        return False
    words = text.split()
    if text.startswith("no such ") and "context" in text and len(words) >= 5:
        return True
    if not any(phrase in text for phrase in NARRATIVE_PHRASES):
        return False
    # Keep short keyword-like phrases; reject explanatory prose.
    return len(words) >= 10


def normalize_rollout_query(output_text: str) -> tuple[str | None, str | None]:
    text = str(output_text or "").strip()
    if not text:
        return None, "empty"

    if SEARCH_QUERY_MARKER in text:
        if COMPLETED_MARKER not in text:
            return None, "missing_completed_marker"
        payload = text.split(SEARCH_QUERY_MARKER, 1)[1]
        text = payload.split(COMPLETED_MARKER, 1)[0].strip()
        if not text:
            return None, "empty_between_markers"

    query = _normalize_query_text(text)
    if not query:
        return None, "empty_after_normalize"
    if "[[ ##" in query or "## ]]" in query:
        return None, "marker_residue"
    if _looks_narrative_like(query):
        return None, "narrative_like"
    return query, None


def clean_output(output_text: str) -> str:
    query, _reason = normalize_rollout_query(output_text)
    if not query:
        return ""
    return format_dspy_output(query)


def format_context(retrieved_docs: list[dict[str, Any]], hop: int) -> str:
    docs_before = [d for d in retrieved_docs if int(d.get("hop", 0) or 0) < hop]
    if not docs_before:
        return "N/A"
    parts = []
    for i, doc in enumerate(docs_before):
        title = doc.get("title", "Unknown")
        text = doc.get("text", "")
        parts.append(f"[{i+1}] «{title}: {text}»")
    return "\n".join(parts)


def load_rollouts(path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            base_qid = str(row.get("base_qid", row.get("qid", "")))
            if not base_qid:
                continue
            grouped[base_qid].append(row)
    return grouped


def load_selector_scores(
    scores_path: Path,
    selector: str,
) -> dict[tuple[str, Any, int], float]:
    scores: dict[tuple[str, Any, int], float] = {}
    with scores_path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            variant_id = row.get("variant_id", row.get("variant_idx"))
            key = (str(row["qid"]), variant_id, int(row["hop"]))
            if selector == "seer":
                scores[key] = float(row.get("seer_ap", 0.0) or 0.0)
                continue

            if selector == "mmr":
                req_details = row.get("req_details") or []
                if not req_details:
                    scores[key] = 0.0
                else:
                    scores[key] = statistics.mean(float(r["max_confidence"]) for r in req_details)
                continue

            if selector == "decomp_binary":
                score = row.get("decomp_binary_ap")
                if score is None:
                    score = row.get("decomposed_binary_ap")
                if score is None:
                    score = row.get("jina_binary_ap", 0.0)
                scores[key] = float(score or 0.0)
                continue

            if selector == "raw_jina_maxmean":
                score = row.get("raw_jina_maxmean_top3")
                if score is None:
                    score = row.get("raw_jina_maxmean")
                scores[key] = float(score or 0.0)
                continue

            if selector == "raw_jina_max":
                score = row.get("raw_jina_max")
                scores[key] = float(score or 0.0)
                continue

            raise ValueError(f"Unsupported selector for score loading: {selector}")
    return scores


def seer_ap_for_hop(rollout: dict[str, Any], hop_idx: int) -> float:
    return seer_ap_from_rollout_xml(
        seer_label=rollout.get("seer_label"),
        retrieved=list(rollout.get("retrieved", []) or []),
        hop_num=hop_idx + 1,
    )


def _default_min_score(selector: str) -> float:
    if selector in {"mmr", "decomp_binary", "raw_jina_maxmean", "raw_jina_max"}:
        return -999.0
    return 0.01


def _score_key(selector: str) -> str:
    if selector == "gold":
        return "hop_coverage"
    if selector == "seer":
        return "seer_ap"
    if selector == "mmr":
        return "mmr_score"
    if selector == "decomp_binary":
        return "decomp_binary_ap"
    if selector == "raw_jina_maxmean":
        return "raw_jina_maxmean_top3"
    if selector == "raw_jina_max":
        return "raw_jina_max"
    raise ValueError(f"Unknown selector: {selector}")


def decomp_binary_ap_for_hop(rollout: dict[str, Any], hop_idx: int) -> float:
    return decomp_binary_ap_from_rollout(rollout, hop_idx)


def parse_requirements(text: str) -> list[str]:
    return shared_parse_requirements(text)


def _truncate_grouped_by_limit(
    grouped: dict[str, list[dict[str, Any]]],
    limit: int | None,
) -> dict[str, list[dict[str, Any]]]:
    if limit is None:
        return grouped
    limited_keys = list(grouped.keys())[:limit]
    return {k: grouped[k] for k in limited_keys}


def _build_requirement_requests(
    grouped: dict[str, list[dict[str, Any]]],
    model: str,
) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    model_slug = normalize_model_slug(model)
    for base_qid, variants in grouped.items():
        if not variants:
            continue
        question = str(variants[0].get("question", "")).strip()
        if not question:
            continue
        user_prompt = REQ_USER_TEMPLATE.format(question=question)
        cache_key = sha256_text(
            model_slug,
            REQ_SYSTEM_PROMPT,
            user_prompt,
            question,
            "",
            "decomposed_req",
        )
        requests.append(
            {
                "model": model,
                "messages": [
                    {"role": "system", "content": REQ_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "cache_key": cache_key,
                "meta": {"base_qid": base_qid, "question": question},
            }
        )
    return requests


PassageKey = tuple[str, str]


def _build_jina_tasks(
    grouped: dict[str, list[dict[str, Any]]],
    requirements_by_question: dict[str, list[str]],
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], dict[PassageKey, int]], dict[tuple[str, int, int], list[PassageKey]]]:
    tasks: list[dict[str, Any]] = []
    passage_maps: dict[tuple[str, int], dict[PassageKey, int]] = {}
    variant_passage_map: dict[tuple[str, int, int], list[PassageKey]] = {}

    for base_qid, variants in grouped.items():
        if not variants:
            continue
        question = str(variants[0].get("question", "")).strip()
        reqs = requirements_by_question.get(question, [])
        if not reqs:
            continue

        max_hops = max(len(v.get("hop_details", [])) for v in variants)
        for hop_num in range(1, max_hops + 1):
            unique_passages: dict[PassageKey, str] = {}
            for variant_idx, variant in enumerate(variants):
                retrieved = variant.get("retrieved", [])
                hop_passages = [p for p in retrieved if int(p.get("hop", 0) or 0) <= hop_num]

                variant_keys: list[PassageKey] = []
                for passage in hop_passages:
                    title = str(passage.get("title", "No Title"))
                    text = str(passage.get("text", ""))
                    key: PassageKey = (title, text)
                    variant_keys.append(key)
                    if key not in unique_passages:
                        unique_passages[key] = f"{title} — {text}"
                variant_passage_map[(base_qid, variant_idx, hop_num)] = variant_keys

            if not unique_passages:
                continue

            unique_keys = list(unique_passages.keys())
            unique_docs = [unique_passages[k] for k in unique_keys]
            passage_maps[(base_qid, hop_num)] = {k: i for i, k in enumerate(unique_keys)}

            for req_idx, requirement in enumerate(reqs):
                tasks.append(
                    {
                        "base_qid": base_qid,
                        "hop": hop_num,
                        "req_idx": req_idx,
                        "requirement": requirement,
                        "documents": unique_docs,
                    }
                )

    return tasks, passage_maps, variant_passage_map


def _compute_decomposed_ap(
    match_results: list[dict[str, Any]],
    num_reqs: int,
    num_passages: int,
    threshold: float = 0.5,
) -> dict[str, Any]:
    if num_reqs == 0:
        return {"binary_ap": 1.0, "req_details": []}

    conf_matrix: dict[int, dict[int, float]] = defaultdict(dict)
    for match in match_results:
        conf_matrix[int(match["req_idx"])][int(match["passage_idx"])] = float(match["confidence"])

    present_map: dict[str, list[int]] = {}
    for req_idx in range(num_reqs):
        fid = f"f{req_idx + 1}"
        matching = [
            pi + 1
            for pi in range(num_passages)
            if conf_matrix.get(req_idx, {}).get(pi, 0.0) >= threshold
        ]
        if matching:
            present_map[fid] = matching

    fake_parsed = SeerParseResult(
        requirements=[f"req_{i}" for i in range(num_reqs)],
        present_map=present_map,
        missing=[],
        raw_text="",
        valid=True,
    )
    binary_ap = float(compute_seer_metrics(fake_parsed, num_passages).get("seer_ap", 0.0))

    req_details: list[dict[str, Any]] = []
    for req_idx in range(num_reqs):
        confidences = [conf_matrix.get(req_idx, {}).get(pi, 0.0) for pi in range(num_passages)]
        req_details.append(
            {
                "req_idx": req_idx,
                "max_confidence": max(confidences) if confidences else 0.0,
                "best_passage": (confidences.index(max(confidences)) + 1) if confidences else None,
                "confidences": confidences,
            }
        )

    return {"binary_ap": binary_ap, "req_details": req_details}


def _compute_all_jina_scores(
    grouped: dict[str, list[dict[str, Any]]],
    requirements_by_question: dict[str, list[str]],
    jina_scores: dict[tuple[str, int, int], dict[int, float]],
    passage_maps: dict[tuple[str, int], dict[PassageKey, int]],
    variant_passage_map: dict[tuple[str, int, int], list[PassageKey]],
    threshold: float,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for base_qid, variants in grouped.items():
        if not variants:
            continue
        question = str(variants[0].get("question", "")).strip()
        reqs = requirements_by_question.get(question, [])
        if not reqs:
            continue

        max_hops = max(len(v.get("hop_details", [])) for v in variants)
        for variant_idx, variant in enumerate(variants):
            for hop_num in range(1, max_hops + 1):
                variant_keys = variant_passage_map.get((base_qid, variant_idx, hop_num))
                if variant_keys is None:
                    continue
                passage_map = passage_maps.get((base_qid, hop_num))
                if passage_map is None:
                    continue

                n_passages = len(variant_keys)
                match_results: list[dict[str, Any]] = []
                for req_idx in range(len(reqs)):
                    scores = jina_scores.get((base_qid, hop_num, req_idx), {})
                    for passage_idx, passage_key in enumerate(variant_keys):
                        unique_idx = passage_map.get(passage_key)
                        confidence = scores.get(unique_idx, 0.0) if unique_idx is not None else 0.0
                        match_results.append(
                            {
                                "req_idx": req_idx,
                                "passage_idx": passage_idx,
                                "confidence": confidence,
                            }
                        )

                decomposed = _compute_decomposed_ap(
                    match_results,
                    len(reqs),
                    n_passages,
                    threshold=threshold,
                )

                hop_details = variant.get("hop_details", [])
                gold_coverage = 0.0
                if hop_num - 1 < len(hop_details):
                    gold_coverage = float(hop_details[hop_num - 1].get("coverage_after_hop", 0.0) or 0.0)

                results.append(
                    {
                        "qid": base_qid,
                        "question": question,
                        "variant_idx": variant_idx,
                        "variant_id": variant.get("prompt_variant_id", variant_idx),
                        "hop": hop_num,
                        "gold_coverage": gold_coverage,
                        "jina_binary_ap": decomposed["binary_ap"],
                        "num_requirements": len(reqs),
                        "num_passages": n_passages,
                        "req_details": decomposed["req_details"],
                    }
                )

    return results


def _jina_cache_key(model: str, query: str, documents: list[str]) -> str:
    joined = "||".join([model, query, *documents])
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


async def _jina_rerank(
    *,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    api_key: str,
    model: str,
    query: str,
    documents: list[str],
    cache_dir: Path,
) -> list[dict[str, Any]]:
    cache_key = _jina_cache_key(model, query, documents)
    cache_path = cache_dir / f"{cache_key}.json"
    if cache_path.exists():
        try:
            with cache_path.open() as handle:
                return json.load(handle)
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
                    wait = 2**attempt + random.random()
                    if response.status_code == 429:
                        wait = max(wait, 10)
                    await asyncio.sleep(wait)
                    continue
                response.raise_for_status()
                results = response.json().get("results", [])
                with cache_path.open("w") as handle:
                    json.dump(results, handle, separators=(",", ":"))
                return results
            except httpx.TimeoutException:
                if attempt < retries - 1:
                    await asyncio.sleep(2**attempt + random.random())
                    continue
                raise
    raise RuntimeError("Jina rerank failed after retries")


async def _build_mmr_scores(
    *,
    rollouts_path: Path,
    requirements_dir: Path,
    scores_dir: Path,
    openrouter_model: str,
    openrouter_concurrency: int,
    openrouter_temperature: float,
    jina_concurrency: int,
    threshold: float,
    limit: int | None,
    skip_requirements: bool,
    skip_scoring: bool,
) -> Path:
    requirements_file = requirements_dir / "requirements.json"
    scores_file = scores_dir / "jina_candidate_scores.jsonl"

    if not skip_requirements and not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is required to auto-build MMR scores")
    if not skip_scoring and not os.environ.get("JINA_AI_API_KEY"):
        raise RuntimeError("JINA_AI_API_KEY is required to auto-build MMR scores")

    requirements_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)

    if not skip_requirements:
        print("\n" + "=" * 80)
        print("MMR Step 1/3: Requirement generation")
        print("=" * 80)
        requirements_dir.mkdir(parents=True, exist_ok=True)
        req_cache = requirements_dir / "cache_reqs"
        req_cache.mkdir(parents=True, exist_ok=True)

        grouped = _truncate_grouped_by_limit(load_rollouts(rollouts_path), limit)
        record_count = sum(len(v) for v in grouped.values())
        print(f"  Loaded {record_count} rollout records across {len(grouped)} questions")

        req_requests = _build_requirement_requests(grouped, openrouter_model)
        print(f"  Requirement requests: {len(req_requests)}")

        config = OpenRouterConfig(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
            concurrency=openrouter_concurrency,
            temperature=openrouter_temperature,
        )
        req_results = await run_completions(config, req_cache, req_requests)

        requirements_by_question: dict[str, list[str]] = {}
        valid = 0
        total_reqs = 0
        for result, req in zip(req_results, req_requests):
            question = req["meta"]["question"]
            response = result.get("response", {})
            choices = response.get("choices", [])
            content = choices[0]["message"]["content"] if choices else ""
            reqs = parse_requirements(content)
            requirements_by_question[question] = reqs
            if reqs:
                valid += 1
                total_reqs += len(reqs)

        write_json(requirements_file, requirements_by_question)
        avg_reqs = total_reqs / max(valid, 1)
        print(f"  Requirements valid: {valid}/{len(req_requests)} (avg reqs={avg_reqs:.2f})")
        print(f"  Saved: {requirements_file}")
    elif not requirements_file.exists():
        raise FileNotFoundError(f"--mmr-skip-requirements set but missing {requirements_file}")
    requirements_by_question = read_json(requirements_file)

    if not skip_scoring:
        print("\n" + "=" * 80)
        print("MMR Step 2/3: Jina reranker scoring")
        print("=" * 80)

        grouped = _truncate_grouped_by_limit(load_rollouts(rollouts_path), limit)
        scores_dir.mkdir(parents=True, exist_ok=True)
        cache_dir = scores_dir / "cache_jina"
        cache_dir.mkdir(parents=True, exist_ok=True)

        tasks, passage_maps, variant_passage_map = _build_jina_tasks(grouped, requirements_by_question)
        total_docs = sum(len(t["documents"]) for t in tasks)
        print(f"  Tasks: {len(tasks)} | Docs: {total_docs} | Concurrency: {jina_concurrency}")

        semaphore = asyncio.Semaphore(jina_concurrency)
        limits = httpx.Limits(
            max_connections=jina_concurrency + 10,
            max_keepalive_connections=jina_concurrency,
        )
        rate_limiter = asyncio.Semaphore(8)  # ~500 RPM

        async def _rate_limit_refill():
            while True:
                await asyncio.sleep(0.12)
                try:
                    rate_limiter.release()
                except ValueError:
                    pass

        jina_scores: dict[tuple, dict[int, float]] = {}
        errors: list[str] = []

        async with httpx.AsyncClient(limits=limits) as client:
            refill_task = asyncio.create_task(_rate_limit_refill())

            async def _run_task(task: dict[str, Any]):
                try:
                    await rate_limiter.acquire()
                    results = await _jina_rerank(
                        client=client,
                        semaphore=semaphore,
                        api_key=os.environ["JINA_AI_API_KEY"],
                        model=MMR_DEFAULT_RERANKER_MODEL,
                        query=task["requirement"],
                        documents=task["documents"],
                        cache_dir=cache_dir,
                    )
                    score_map = {r["index"]: r["relevance_score"] for r in results}
                    return (task["base_qid"], task["hop"], task["req_idx"]), score_map
                except Exception as e:  # noqa: BLE001
                    errors.append(f"Q:{task['base_qid']} H:{task['hop']} R:{task['req_idx']}: {e}")
                    return None

            results = await tqdm_async.gather(*[_run_task(t) for t in tasks], desc="Jina Reranker v3 Reranking")
            refill_task.cancel()

        for item in results:
            if item is None:
                continue
            key, score_map = item
            jina_scores[key] = score_map

        if errors:
            print(f"  Reranker errors: {len(errors)} (showing first 5)")
            for err in errors[:5]:
                print(f"    {err}")

        candidate_scores = _compute_all_jina_scores(
            grouped,
            requirements_by_question,
            jina_scores,
            passage_maps,
            variant_passage_map,
            threshold,
        )
        write_jsonl(scores_file, candidate_scores)
        print(f"  Saved {len(candidate_scores)} score rows: {scores_file}")
    elif not scores_file.exists():
        raise FileNotFoundError(f"--mmr-skip-scoring set but missing {scores_file}")

    return scores_file


async def _build_seer_scores(
    *,
    rollouts_path: Path,
    scores_dir: Path,
    cache_dir: Path,
    model: str,
    prompt_style: str,
    openrouter_concurrency: int,
    openrouter_temperature: float,
    limit: int | None,
    skip_scoring: bool,
) -> Path:
    scores_file = scores_dir / "seer_candidate_scores.jsonl"
    scores_dir.mkdir(parents=True, exist_ok=True)

    grouped = _truncate_grouped_by_limit(load_rollouts(rollouts_path), limit)
    record_count = sum(len(v) for v in grouped.values())
    print(
        f"  Loaded {record_count} rollout records across {len(grouped)} questions for Seer score prep"
    )

    rows: list[dict[str, Any]] = []
    missing_tasks: list[dict[str, Any]] = []
    local_count = 0

    for base_qid, variants in grouped.items():
        if not variants:
            continue
        question = str(variants[0].get("question", "")).strip()
        if not question:
            continue
        max_hops = max(len(v.get("hop_details", [])) for v in variants)

        for hop_num in range(1, max_hops + 1):
            candidates_passages: list[list[dict[str, Any]]] = []
            candidate_meta: list[tuple[Any, int]] = []

            for variant_idx, variant in enumerate(variants):
                details = variant.get("hop_details", [])
                if len(details) < hop_num:
                    continue
                variant_id = variant.get("prompt_variant_id", variant_idx)
                seer_label = str(variant.get("seer_label", "") or "").strip()

                if seer_label:
                    rows.append(
                        {
                            "qid": base_qid,
                            "variant_id": variant_id,
                            "hop": hop_num,
                            "seer_ap": float(seer_ap_for_hop(variant, hop_num - 1)),
                            "source": "seer_label",
                        }
                    )
                    local_count += 1
                    continue

                candidate_meta.append((variant_id, variant_idx))
                passages = [
                    p
                    for p in list(variant.get("retrieved", []) or [])
                    if int(p.get("hop", 0) or 0) <= hop_num
                ]
                candidates_passages.append(passages)

            if candidate_meta:
                missing_tasks.append(
                    {
                        "qid": base_qid,
                        "question": question,
                        "hop": hop_num,
                        "candidates_passages": candidates_passages,
                        "candidate_meta": candidate_meta,
                    }
                )

    print(f"  Seer scores from existing labels: {local_count}")
    print(f"  Seer score tasks needing model calls: {len(missing_tasks)}")

    if missing_tasks:
        if skip_scoring:
            print("  --seer-skip-scoring enabled; missing-label tasks will be skipped")
        else:
            if not os.environ.get("OPENROUTER_API_KEY"):
                raise RuntimeError(
                    "OPENROUTER_API_KEY is required to score Seer rows missing seer_label"
                )
            cache_dir.mkdir(parents=True, exist_ok=True)
            config = OpenRouterConfig(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ.get("OPENROUTER_API_KEY", ""),
                concurrency=openrouter_concurrency,
                temperature=openrouter_temperature,
            )
            chunk_size = max(1, openrouter_concurrency)
            model_count = 0
            for i in tqdm(range(0, len(missing_tasks), chunk_size), desc="seer_score_calls"):
                chunk = missing_tasks[i : i + chunk_size]
                chunk_results = await asyncio.gather(
                    *[
                        score_seer_candidates_batch(
                            question=task["question"],
                            candidates_passages=task["candidates_passages"],
                            model=model,
                            prompt_style=prompt_style,
                            openrouter_config=config,
                            cache_dir=cache_dir,
                            request_meta={"qid": task["qid"], "hop": task["hop"]},
                        )
                        for task in chunk
                    ]
                )
                for task, scores in zip(chunk, chunk_results):
                    for (variant_id, _variant_idx), score in zip(task["candidate_meta"], scores):
                        rows.append(
                            {
                                "qid": task["qid"],
                                "variant_id": variant_id,
                                "hop": int(task["hop"]),
                                "seer_ap": float(score),
                                "source": "openrouter",
                            }
                        )
                        model_count += 1
            print(f"  Seer scores from model calls: {model_count}")

    rows.sort(key=lambda r: (str(r["qid"]), int(r["hop"]), str(r.get("variant_id"))))
    write_jsonl(scores_file, rows)
    print(f"  Saved {len(rows)} Seer score rows: {scores_file}")
    return scores_file


def build_examples(
    grouped: dict[str, list[dict[str, Any]]],
    *,
    selector: str,
    min_score: float,
    instruction: str,
    selector_scores: dict[tuple[str, Any, int], float] | None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    examples: list[dict[str, Any]] = []
    stats = {
        "questions_seen": 0,
        "hops_seen": 0,
        "hops_selected": 0,
        "variants_scored": 0,
        "variants_missing_score": 0,
        "variants_invalid_query": 0,
        "hops_skipped_no_valid_variant": 0,
        "invalid_query_reasons": Counter(),
    }
    score_key = _score_key(selector)

    for base_qid, variants in tqdm(grouped.items(), desc=f"prepare_sft[{selector}]"):
        if not variants:
            continue
        stats["questions_seen"] += 1
        num_hops = max(len(v.get("hop_details", [])) for v in variants)

        for hop_idx in range(num_hops):
            hop_num = hop_idx + 1
            stats["hops_seen"] += 1
            scored: list[tuple[float, dict[str, Any], str]] = []

            for variant in variants:
                details = variant.get("hop_details", [])
                if len(details) < hop_num:
                    continue
                stats["variants_scored"] += 1

                if selector == "gold":
                    score = float(details[hop_idx].get("coverage_after_hop", 0.0) or 0.0)
                elif selector == "seer":
                    key = (base_qid, variant.get("prompt_variant_id"), hop_num)
                    if selector_scores is not None:
                        if key not in selector_scores:
                            stats["variants_missing_score"] += 1
                            continue
                        score = float(selector_scores[key])
                    else:
                        score = seer_ap_for_hop(variant, hop_idx)
                elif selector == "mmr":
                    if selector_scores is None:
                        raise ValueError("selector_scores missing while selector=mmr")
                    key = (base_qid, variant.get("prompt_variant_id"), hop_num)
                    if key not in selector_scores:
                        stats["variants_missing_score"] += 1
                        continue
                    score = float(selector_scores[key])
                elif selector == "decomp_binary":
                    key = (base_qid, variant.get("prompt_variant_id"), hop_num)
                    if selector_scores is not None and key in selector_scores:
                        score = float(selector_scores[key])
                    else:
                        score = decomp_binary_ap_for_hop(variant, hop_idx)
                elif selector in {"raw_jina_maxmean", "raw_jina_max"}:
                    if selector_scores is None:
                        raise ValueError(f"selector_scores missing while selector={selector}")
                    key = (base_qid, variant.get("prompt_variant_id"), hop_num)
                    if key not in selector_scores:
                        stats["variants_missing_score"] += 1
                        continue
                    score = float(selector_scores[key])
                else:
                    raise ValueError(f"Unknown selector: {selector}")

                if score < min_score:
                    continue

                query_text, invalid_reason = normalize_rollout_query(details[hop_idx].get("query", ""))
                if not query_text:
                    stats["variants_invalid_query"] += 1
                    stats["invalid_query_reasons"][invalid_reason or "unknown"] += 1
                    continue

                scored.append((score, variant, query_text))

            if not scored:
                stats["hops_skipped_no_valid_variant"] += 1
                continue

            best_score, best_variant, best_query = max(scored, key=lambda x: x[0])
            target_output = format_dspy_output(best_query)
            if not target_output:
                continue

            context = format_context(best_variant.get("retrieved", []), hop_num)
            examples.append(
                {
                    "instruction": instruction,
                    "input": f"Context: {context}\n\nQuestion: {best_variant['question']}",
                    "output": target_output,
                    "meta": {
                        "base_qid": base_qid,
                        "hop": hop_num,
                        "selector": selector,
                        score_key: best_score,
                        "variant_id": best_variant.get("prompt_variant_id"),
                    },
                }
            )
            stats["hops_selected"] += 1

    return examples, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SFT data from rollout JSONL.")
    parser.add_argument("--input", required=True, help="Rollout JSONL")
    parser.add_argument("--output", required=True, help="SFT JSONL output path")
    parser.add_argument(
        "--selector",
        required=True,
        choices=[
            "gold",
            "seer",
            "mmr",
            "decomp_binary",
            "raw_jina_maxmean",
            "raw_jina_max",
        ],
        help="Variant selection metric",
    )
    parser.add_argument(
        "--scores",
        default=None,
        help=(
            "Score JSONL for selector in {seer,mmr,decomp_binary,raw_jina_maxmean,raw_jina_max}. "
            "If omitted, seer/mmr/decomp_binary can be auto-generated in-process; "
            "raw_jina_* requires explicit --scores."
        ),
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Minimum metric score to include (default depends on selector).",
    )
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)

    # Jina auto-score generation knobs (used only when selector in {mmr,decomp_binary} and --scores omitted)
    parser.add_argument("--mmr-requirements-dir", default="data/decomposed_seer_sft_musique")
    parser.add_argument("--mmr-scores-dir", default="data/decomposed_seer_sft_musique_jina")
    parser.add_argument("--mmr-openrouter-model", default=MMR_DEFAULT_OPENROUTER_MODEL)
    parser.add_argument("--mmr-openrouter-concurrency", type=int, default=20)
    parser.add_argument("--mmr-openrouter-temperature", type=float, default=0.0)
    parser.add_argument("--mmr-jina-concurrency", type=int, default=10)
    parser.add_argument("--mmr-threshold", type=float, default=0.5)
    parser.add_argument("--mmr-limit", type=int, default=None)
    parser.add_argument("--mmr-skip-requirements", action="store_true")
    parser.add_argument("--mmr-skip-scoring", action="store_true")

    # Seer auto-score generation knobs (used only when selector=seer and --scores omitted)
    parser.add_argument("--seer-scores-dir", default="data/seer_sft_scores")
    parser.add_argument("--seer-cache-dir", default="data/cache/seer_sft_prepare")
    parser.add_argument("--seer-model", default=SEER_DEFAULT_MODEL)
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
    parser.add_argument("--seer-openrouter-concurrency", type=int, default=20)
    parser.add_argument("--seer-openrouter-temperature", type=float, default=0.0)
    parser.add_argument("--seer-limit", type=int, default=None)
    parser.add_argument("--seer-skip-scoring", action="store_true")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    min_score = _default_min_score(args.selector) if args.min_score is None else float(args.min_score)

    selector_scores = None
    scores_path = None
    if args.selector in {"raw_jina_maxmean", "raw_jina_max"}:
        if not args.scores:
            raise ValueError(
                "raw_jina selectors require explicit --scores "
                "(score file created by scripts/dpo/score.py)."
            )
        scores_path = Path(args.scores)
    elif args.selector == "seer":
        if args.scores:
            scores_path = Path(args.scores)
        else:
            print("No --scores provided for selector=seer. Auto-building Seer score artifacts.")
            scores_path = asyncio.run(
                _build_seer_scores(
                    rollouts_path=input_path,
                    scores_dir=Path(args.seer_scores_dir),
                    cache_dir=Path(args.seer_cache_dir),
                    model=args.seer_model,
                    prompt_style=args.seer_prompt_style,
                    openrouter_concurrency=args.seer_openrouter_concurrency,
                    openrouter_temperature=args.seer_openrouter_temperature,
                    limit=args.seer_limit,
                    skip_scoring=args.seer_skip_scoring,
                )
            )
    elif args.selector in {"mmr", "decomp_binary"}:
        if args.scores:
            scores_path = Path(args.scores)
        else:
            print(
                f"No --scores provided for selector={args.selector}. "
                "Auto-building Jina scores from rollouts."
            )
            scores_path = asyncio.run(
                _build_mmr_scores(
                    rollouts_path=input_path,
                    requirements_dir=Path(args.mmr_requirements_dir),
                    scores_dir=Path(args.mmr_scores_dir),
                    openrouter_model=args.mmr_openrouter_model,
                    openrouter_concurrency=args.mmr_openrouter_concurrency,
                    openrouter_temperature=args.mmr_openrouter_temperature,
                    jina_concurrency=args.mmr_jina_concurrency,
                    threshold=args.mmr_threshold,
                    limit=args.mmr_limit,
                    skip_requirements=args.mmr_skip_requirements,
                    skip_scoring=args.mmr_skip_scoring,
                )
            )
    elif args.selector != "gold":
        raise ValueError(f"Unsupported selector: {args.selector}")

    if scores_path is not None:
        if not scores_path.exists():
            raise FileNotFoundError(f"Scores not found: {scores_path}")
        if args.selector != "gold":
            print(f"Loading selector scores from {scores_path}...")
            selector_scores = load_selector_scores(scores_path, args.selector)
            print(f"Loaded {len(selector_scores)} score entries for selector={args.selector}")

    print(f"Loading rollouts from {input_path}...")
    grouped = load_rollouts(input_path)
    if args.selector in {"mmr", "decomp_binary"} and not args.scores and args.mmr_limit is not None:
        grouped = _truncate_grouped_by_limit(grouped, args.mmr_limit)
        print(f"Applied --mmr-limit to final selection: {len(grouped)} base questions")
    if args.selector == "seer" and not args.scores and args.seer_limit is not None:
        grouped = _truncate_grouped_by_limit(grouped, args.seer_limit)
        print(f"Applied --seer-limit to final selection: {len(grouped)} base questions")
    print(f"Loaded {len(grouped)} base questions")

    examples, stats = build_examples(
        grouped,
        selector=args.selector,
        min_score=min_score,
        instruction=args.instruction,
        selector_scores=selector_scores,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for row in examples:
            handle.write(json.dumps(row) + "\n")

    print(f"Saved {len(examples)} examples to {output_path}")
    print(
        "Stats: "
        f"questions={stats['questions_seen']}, hops_seen={stats['hops_seen']}, "
        f"hops_selected={stats['hops_selected']}, variants_scored={stats['variants_scored']}, "
        f"missing_scores={stats['variants_missing_score']}, "
        f"invalid_query_candidates={stats['variants_invalid_query']}, "
        f"hops_skipped_no_valid_variant={stats['hops_skipped_no_valid_variant']}"
    )
    invalid_reasons = stats["invalid_query_reasons"]
    if invalid_reasons:
        reason_summary = ", ".join(f"{k}:{v}" for k, v in sorted(invalid_reasons.items()))
        print(f"Invalid query reasons: {reason_summary}")


if __name__ == "__main__":
    main()
