from __future__ import annotations

import asyncio
import hashlib
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx

from seer.parse_seer_xml import SeerParseResult
from seer.seer_metrics import compute_seer_metrics
from seer.util_io import ensure_dir


PassageKey = tuple[str, str]


def _passage_key(passage: dict[str, Any]) -> PassageKey:
    return (
        str(passage.get("title", "No Title")),
        str(passage.get("text", "")),
    )


def _cache_key(model: str, query: str, documents: list[str]) -> str:
    parts = [model, query, *documents]
    return hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()


async def _jina_rerank(
    *,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    api_key: str,
    model: str,
    requirement: str,
    documents: list[str],
    cache_dir: Path,
) -> list[dict[str, Any]]:
    ensure_dir(cache_dir)
    key = _cache_key(model, requirement, documents)
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
                        "query": requirement,
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


def _index_candidates(
    candidates_passages: list[list[dict[str, Any]]],
) -> tuple[list[str], list[list[int]]]:
    unique_index: dict[PassageKey, int] = {}
    documents: list[str] = []
    candidate_doc_indices: list[list[int]] = []

    for passages in candidates_passages:
        idxs: list[int] = []
        for passage in passages:
            key = _passage_key(passage)
            if key not in unique_index:
                unique_index[key] = len(documents)
                documents.append(f"{key[0]} — {key[1]}")
            idxs.append(unique_index[key])
        candidate_doc_indices.append(idxs)
    return documents, candidate_doc_indices


def _binary_ap_from_requirement_scores(
    requirement_to_scores: dict[int, dict[int, float]],
    candidate_indices: list[int],
    *,
    num_requirements: int,
    threshold: float,
) -> float:
    if num_requirements == 0:
        return 1.0

    present_map: dict[str, list[int]] = {}
    for req_idx in range(num_requirements):
        fid = f"f{req_idx + 1}"
        hits: list[int] = []
        scores = requirement_to_scores.get(req_idx, {})
        for local_idx, global_idx in enumerate(candidate_indices):
            score = float(scores.get(global_idx, 0.0))
            if score >= threshold:
                hits.append(local_idx + 1)
        if hits:
            present_map[fid] = hits

    parsed = SeerParseResult(
        requirements=[f"req_{i+1}" for i in range(num_requirements)],
        present_map=present_map,
        missing=[],
        raw_text="",
        valid=True,
    )
    return float(compute_seer_metrics(parsed, num_passages=len(candidate_indices)).get("seer_ap", 0.0))


def _aggregate_scores(
    *,
    requirement_to_scores: dict[int, dict[int, float]],
    candidate_doc_indices: list[list[int]],
    num_requirements: int,
    threshold: float,
) -> tuple[list[float], list[float]]:
    mmr_scores: list[float] = []
    binary_scores: list[float] = []

    for candidate_indices in candidate_doc_indices:
        if num_requirements == 0:
            mmr_scores.append(0.0)
            binary_scores.append(1.0)
            continue

        max_confs: list[float] = []
        for req_idx in range(num_requirements):
            scores = requirement_to_scores.get(req_idx, {})
            best = 0.0
            for global_idx in candidate_indices:
                best = max(best, float(scores.get(global_idx, 0.0)))
            max_confs.append(best)

        mmr_scores.append(float(statistics.mean(max_confs)))
        binary_scores.append(
            _binary_ap_from_requirement_scores(
                requirement_to_scores,
                candidate_indices,
                num_requirements=num_requirements,
                threshold=threshold,
            )
        )

    return mmr_scores, binary_scores


async def score_mmr_and_binary_candidates_batch(
    *,
    requirements: list[str],
    candidates_passages: list[list[dict[str, Any]]],
    jina_api_key: str,
    cache_dir: str | Path,
    jina_model: str = "jina-reranker-v3",
    concurrency: int = 20,
    threshold: float = 0.5,
) -> tuple[list[float], list[float]]:
    if not candidates_passages:
        return [], []

    if not requirements:
        zero_mmr = [0.0 for _ in candidates_passages]
        one_binary = [1.0 for _ in candidates_passages]
        return zero_mmr, one_binary

    documents, candidate_doc_indices = _index_candidates(candidates_passages)
    if not documents:
        zero_mmr = [0.0 for _ in candidates_passages]
        zero_binary = [0.0 for _ in candidates_passages]
        return zero_mmr, zero_binary

    semaphore = asyncio.Semaphore(max(1, int(concurrency)))
    requirement_to_scores: dict[int, dict[int, float]] = {}
    cache_path = Path(cache_dir)

    async with httpx.AsyncClient() as client:
        tasks = [
            _jina_rerank(
                client=client,
                semaphore=semaphore,
                api_key=jina_api_key,
                model=jina_model,
                requirement=requirement,
                documents=documents,
                cache_dir=cache_path,
            )
            for requirement in requirements
        ]
        results = await asyncio.gather(*tasks)

    for req_idx, req_results in enumerate(results):
        requirement_to_scores[req_idx] = {
            int(item["index"]): float(item["relevance_score"])
            for item in req_results
            if "index" in item
        }

    return _aggregate_scores(
        requirement_to_scores=requirement_to_scores,
        candidate_doc_indices=candidate_doc_indices,
        num_requirements=len(requirements),
        threshold=threshold,
    )


async def score_mmr_candidates_batch(
    *,
    requirements: list[str],
    candidates_passages: list[list[dict[str, Any]]],
    jina_api_key: str,
    cache_dir: str | Path,
    jina_model: str = "jina-reranker-v3",
    concurrency: int = 20,
) -> list[float]:
    mmr_scores, _ = await score_mmr_and_binary_candidates_batch(
        requirements=requirements,
        candidates_passages=candidates_passages,
        jina_api_key=jina_api_key,
        cache_dir=cache_dir,
        jina_model=jina_model,
        concurrency=concurrency,
    )
    return mmr_scores
