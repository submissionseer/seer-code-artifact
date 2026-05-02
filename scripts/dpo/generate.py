#!/usr/bin/env python3
"""Generate multi-hop DPO rollouts with shared-context selection after each hop.

Pipeline per question:
1) Reuse existing off-policy hop-1 candidates from rollout variants.
2) Select ONE shared context candidate from hop h (policy + metric).
3) Generate K query variants for hop h+1 from that shared context.
4) Retrieve hop h+1 candidates, then repeat until num_hops.

This generalizes the prior 2-hop DPO rollout flow to variable-hop datasets
(e.g., MuSiQue 2-4 hops) while preserving backward compatibility fields
(`hop1`, `hop2`, ...).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import requests
import yaml
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

from seer.openrouter import OpenRouterConfig, normalize_model_slug
from seer.retrieval_utils import (
    extract_gold_titles,
    get_num_hops,
    init_retriever_with_retry,
)
from seer.scoring import (
    generate_requirements_map,
    load_requirements_map,
    score_decomp_binary_candidates_batch,
    score_gold_candidate,
    score_key_for_metric,
    score_mmr_candidates_batch,
    score_seer_candidates_batch,
)
from seer.util_io import ensure_dir, read_jsonl
from seer.util_log import get_logger
from seer.util_text import safe_format_prompt


SEARCH_QUERY_MARKER = "[[ ## search_query ## ]]"
COMPLETED_MARKER = "[[ ## completed ## ]]"
QUERY_GENERATION_PROMPT = """You are a search query generator for multi-hop question answering.

Given a question and any context already retrieved, generate a search query to find information needed to answer the question.

Context:
{context}

Question: {question}

Generate a concise search query.
Output EXACTLY in this format:
[[ ## search_query ## ]]
<query>
[[ ## completed ## ]]"""
NARRATIVE_PHRASES = (
    "there is no",
    "no information",
    "the context",
    "the provided context",
    "the question asks",
    "search query should",
    "however",
)


def _normalize_query_text(candidate: str) -> str:
    query = str(candidate or "").strip()
    query = query.strip("`")
    query = query.strip('"').strip("'").strip()
    query = re.sub(r"\s+", " ", query).strip()
    if not query:
        raise ValueError("Parsed query is empty")
    return query


def _looks_narrative_like_query(query: str) -> bool:
    text = str(query or "").strip().lower()
    if not text:
        return False
    words = text.split()
    if text.startswith("no such ") and "context" in text and len(words) >= 5:
        return True
    if not any(phrase in text for phrase in NARRATIVE_PHRASES):
        return False
    return len(words) >= 10


def parse_dspy_output(response: str) -> str:
    """Extract query text from strict DSPy marker format."""
    text = str(response or "").strip()
    if not text:
        raise ValueError("Empty query-generation response")
    if SEARCH_QUERY_MARKER not in text:
        raise ValueError(
            "Missing required '[[ ## search_query ## ]]' marker in query-generation output: "
            f"{text[:200]!r}"
        )
    candidate = text.split(SEARCH_QUERY_MARKER, 1)[1]
    # Some provider/model outputs omit the completed marker; accept marker-only payload.
    if COMPLETED_MARKER in candidate:
        candidate = candidate.split(COMPLETED_MARKER, 1)[0]
    candidate = candidate.split("[[ ##", 1)[0].strip()
    if "\n" in candidate:
        lines = [ln.strip() for ln in candidate.splitlines() if ln.strip()]
        candidate = lines[0] if lines else ""
    if not candidate:
        raise ValueError("DSPy output missing search_query payload")
    query = _normalize_query_text(candidate)
    if _looks_narrative_like_query(query):
        raise ValueError(f"Narrative query-generation output rejected: {query[:200]!r}")
    return query


def parse_dspy_output_tolerant(response: str) -> str:
    """Best-effort parser used after strict parse retries are exhausted."""
    text = str(response or "").strip()
    if not text:
        return "unknown"
    if SEARCH_QUERY_MARKER in text:
        candidate = text.split(SEARCH_QUERY_MARKER, 1)[1]
        if COMPLETED_MARKER in candidate:
            candidate = candidate.split(COMPLETED_MARKER, 1)[0]
    else:
        candidate = text
    candidate = candidate.strip("`")
    candidate = candidate.strip('"').strip("'").strip()
    candidate = re.sub(r"\s+", " ", candidate).strip()
    candidate = candidate.split("[[ ##", 1)[0].strip()
    candidate = candidate.split("## ]]", 1)[0].strip()
    return candidate or "unknown"


def parse_or_normalize_rollout_query(query_text: str) -> str:
    """Parse a rollout-stored query.

    Historical rollout artifacts may store:
    - strict DSPy marker output, or
    - an already parsed plain query string.
    - legacy marker-only output without [[ ## completed ## ]]
    """
    text = str(query_text or "").strip()
    if not text:
        raise ValueError("Empty rollout query")
    if SEARCH_QUERY_MARKER in text and COMPLETED_MARKER in text:
        return parse_dspy_output(text)
    if SEARCH_QUERY_MARKER in text:
        # Legacy rollout artifact path: marker present, completed marker omitted.
        candidate = text.split(SEARCH_QUERY_MARKER, 1)[1].strip()
        # If there are additional markers/lines, keep the first non-empty logical line.
        candidate = candidate.split("[[ ##", 1)[0].strip()
        if "\n" in candidate:
            lines = [ln.strip() for ln in candidate.splitlines() if ln.strip()]
            candidate = lines[0] if lines else ""
        return _normalize_query_text(candidate)
    return _normalize_query_text(text)


def reward_weighted_sample(scores: list[float], rng: np.random.Generator) -> int:
    """LeReT-style reward-weighted sampling with perfect-score filtering."""
    eligible = [(i, s) for i, s in enumerate(scores) if s < 1.0]
    if not eligible:
        return 0

    indices, weights = zip(*eligible)
    weights = list(weights)

    max_idx = max(range(len(weights)), key=lambda j: weights[j])
    weights[max_idx] += 0.5

    min_w = min(weights)
    if min_w <= 0:
        weights = [w - min_w + 0.01 for w in weights]

    total = sum(weights)
    probs = [w / total for w in weights]

    chosen = int(rng.choice(len(indices), p=probs))
    return int(indices[chosen])


def select_index_by_policy(
    scores: list[float],
    *,
    policy: str,
    rng: np.random.Generator,
) -> int:
    if not scores:
        return 0

    if policy == "best":
        max_score = max(scores)
        best = [i for i, s in enumerate(scores) if s == max_score]
        return int(rng.choice(best))

    return reward_weighted_sample(scores, rng)


def format_context_from_passages(passages: list[dict[str, Any]]) -> str:
    if not passages:
        return "N/A"
    parts = []
    for i, doc in enumerate(passages):
        parts.append(f"[{i+1}] «{doc.get('title', '')}: {doc.get('text', '')}»")
    return "\n".join(parts)


def _extend_selected_context(state: dict[str, Any], new_passages: list[dict[str, Any]]) -> None:
    for passage in new_passages:
        pid = passage.get("doc_id")
        if pid is None:
            continue
        try:
            pid_int = int(pid)
        except (TypeError, ValueError):
            continue
        if pid_int in state["selected_context_pids"]:
            continue
        state["selected_context_pids"].add(pid_int)
        state["selected_context_passages"].append(passage)


class LLMCache:
    """Thread-safe LLM response cache with disk persistence."""

    def __init__(self, cache_path: Path | None):
        self.cache_path = cache_path
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._cache: dict[str, str] = {}
        if cache_path and cache_path.exists():
            try:
                with open(cache_path) as f:
                    self._cache = json.load(f)
            except Exception:
                self._cache = {}

    @staticmethod
    def make_key(question: str, context: str, template: str) -> str:
        return hashlib.md5(f"{question}||{context}||{template}".encode()).hexdigest()

    def get(self, key: str) -> str | None:
        with self._lock:
            if key in self._cache:
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

    def set(self, key: str, value: str):
        with self._lock:
            self._cache[key] = value

    def save(self):
        if not self.cache_path:
            return
        with self._lock:
            data = dict(self._cache)
            hits, misses = self._hits, self._misses
        ensure_dir(self.cache_path.parent)
        with open(self.cache_path, "w") as f:
            json.dump(data, f)
        return hits, misses, len(data)

    @property
    def stats(self) -> tuple[int, int, int]:
        with self._lock:
            return self._hits, self._misses, len(self._cache)


async def async_openrouter_batch(
    call_specs: list[tuple[str, str]],
    or_config: OpenRouterConfig,
    model: str,
    llm_cache: LLMCache,
    concurrency: int = 20,
) -> dict[str, str]:
    """Generate queries for a batch of (cache_key, rendered_prompt) pairs."""
    to_call = []
    results: dict[str, str] = {}
    failures: dict[str, str] = {}
    for cache_key, prompt in call_specs:
        cached = llm_cache.get(cache_key)
        if cached is not None:
            results[cache_key] = cached
        else:
            to_call.append((cache_key, prompt))

    if not to_call:
        return results

    normalized = normalize_model_slug(model)
    completed = 0
    total = len(to_call)

    async def _worker(client: httpx.AsyncClient, queue: asyncio.Queue) -> None:
        nonlocal completed
        while True:
            try:
                cache_key, prompt = queue.get_nowait()
            except asyncio.QueueEmpty:
                return

            payload = {
                "model": normalized,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": or_config.temperature,
            }
            parsed_query = None
            last_error = ""
            for attempt in range(3):
                try:
                    resp = await asyncio.wait_for(
                        client.post(
                            f"{or_config.base_url}/chat/completions",
                            headers={"Authorization": f"Bearer {or_config.api_key}"},
                            json=payload,
                        ),
                        timeout=30.0,
                    )
                    if resp.status_code in {429, 500, 502, 503, 504} and attempt < 2:
                        await asyncio.sleep(2**attempt + random.random())
                        continue
                    resp.raise_for_status()
                    content = resp.json()["choices"][0]["message"]["content"]
                    try:
                        parsed_query = parse_dspy_output(content)
                        break
                    except Exception as e:
                        last_error = f"{type(e).__name__}: {e}"
                        if attempt < 2:
                            await asyncio.sleep(2**attempt + random.random())
                            continue
                except Exception as e:
                    last_error = f"{type(e).__name__}: {e}"
                    if attempt < 2:
                        await asyncio.sleep(2**attempt + random.random())
                    continue

            if parsed_query:
                results[cache_key] = parsed_query
                llm_cache.set(cache_key, parsed_query)
            else:
                failures[cache_key] = last_error or "No content returned from OpenRouter"

            completed += 1
            if completed % 50 == 0 or completed == total:
                print(f"    OpenRouter: {completed}/{total}", flush=True)

    queue: asyncio.Queue = asyncio.Queue()
    for spec in to_call:
        queue.put_nowait(spec)

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(60.0),
        limits=httpx.Limits(
            max_connections=concurrency + 5,
            max_keepalive_connections=concurrency,
        ),
    ) as client:
        workers = [_worker(client, queue) for _ in range(concurrency)]
        await asyncio.gather(*workers)

    if failures:
        sample = list(failures.items())[:3]
        sample_msg = " | ".join(
            f"{k[:8]}: {str(v)[:160]}" for k, v in sample
        )
        # Fail fast only when failures are systemic. Small tails are recovered in-process.
        if len(failures) > max(5, int(0.10 * total)):
            raise RuntimeError(
                f"OpenRouter query generation failed for {len(failures)}/{total} calls. "
                f"Sample: {sample_msg}"
            )
        print(
            f"    OpenRouter warnings: {len(failures)}/{total} calls failed; "
            f"recovering from successful sibling variants. Sample: {sample_msg}",
            flush=True,
        )

    return results


def _local_inference_call(
    batch: list[tuple[str, str]],
    inference_url: str,
    temperature: float,
    timeout: float = 300.0,
    max_split_retries: int = 4,
    max_conn_retries: int = 10,
    conn_retry_delay: float = 30.0,
) -> list[str]:
    payload = {
        "requests": [
            {
                "messages": [{"role": "user", "content": prompt}],
                "max_new_tokens": 128,
                "temperature": temperature,
            }
            for _, prompt in batch
        ]
    }

    for conn_attempt in range(max_conn_retries):
        try:
            resp = requests.post(
                f"{inference_url}/v1/chat_batch",
                json=payload,
                timeout=timeout,
            )
            resp.raise_for_status()
            return resp.json()["responses"]

        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 500 and len(batch) > 1 and max_split_retries > 0:
                print(
                    f"    OOM/server error with batch size {len(batch)}, splitting...",
                    flush=True,
                )
                mid = len(batch) // 2
                left = _local_inference_call(
                    batch[:mid],
                    inference_url,
                    temperature,
                    timeout=timeout,
                    max_split_retries=max_split_retries - 1,
                    max_conn_retries=max_conn_retries,
                    conn_retry_delay=conn_retry_delay,
                )
                right = _local_inference_call(
                    batch[mid:],
                    inference_url,
                    temperature,
                    timeout=timeout,
                    max_split_retries=max_split_retries - 1,
                    max_conn_retries=max_conn_retries,
                    conn_retry_delay=conn_retry_delay,
                )
                return left + right
            raise

        except (requests.ConnectionError, requests.Timeout) as e:
            remaining = max_conn_retries - conn_attempt - 1
            if remaining > 0:
                print(
                    f"    Local inference unavailable (attempt {conn_attempt + 1}/{max_conn_retries}), waiting {conn_retry_delay:.0f}s... ({type(e).__name__})",
                    flush=True,
                )
                time.sleep(conn_retry_delay)
            else:
                raise RuntimeError(
                    f"Local inference unreachable after {max_conn_retries} attempts."
                ) from e


def local_inference_batch(
    call_specs: list[tuple[str, str]],
    inference_url: str,
    temperature: float,
    llm_cache: LLMCache,
    batch_size: int = 20,
    max_retries: int = 3,
) -> dict[str, str]:
    to_call = []
    results: dict[str, str] = {}
    for cache_key, prompt in call_specs:
        cached = llm_cache.get(cache_key)
        if cached is not None:
            results[cache_key] = cached
        else:
            to_call.append((cache_key, prompt))

    if not to_call:
        return results

    completed = 0
    total = len(to_call)

    for batch_start in range(0, total, batch_size):
        batch = to_call[batch_start : batch_start + batch_size]
        pending = list(batch)
        raw_by_key: dict[str, str] = {}

        for attempt in range(max_retries + 1):
            if not pending:
                break
            responses = _local_inference_call(pending, inference_url, temperature)
            if len(responses) != len(pending):
                raise RuntimeError(
                    f"Local inference response count mismatch: expected {len(pending)}, got {len(responses)}"
                )

            next_pending: list[tuple[str, str]] = []
            for (cache_key, prompt), response in zip(pending, responses):
                raw_by_key[cache_key] = response
                try:
                    query = parse_dspy_output(response)
                except Exception:
                    next_pending.append((cache_key, prompt))
                    continue
                results[cache_key] = query
                llm_cache.set(cache_key, query)

            if next_pending and attempt < max_retries:
                print(
                    f"    Local inference strict-parse retry {attempt + 1}/{max_retries} for {len(next_pending)} responses",
                    flush=True,
                )
            pending = next_pending

        for cache_key, _prompt in pending:
            raw = raw_by_key.get(cache_key, "")
            query = parse_dspy_output_tolerant(raw)
            print(
                f"    Local inference strict parse failed for key {cache_key[:8]}; using tolerant fallback",
                flush=True,
            )
            results[cache_key] = query
            llm_cache.set(cache_key, query)

        completed += len(batch)
        print(f"    Local inference: {completed}/{total}", flush=True)

    return results


class RemoteColBERTRetriever:
    """HTTP retriever client for serve_colbert_min.py endpoints."""

    def __init__(self, remote_url: str, timeout: float = 30.0):
        self.remote_url = remote_url.rstrip("/")
        self.timeout = timeout

    def retrieve(self, query: str, top_k: int = 10, exclude_pids: set[int] | None = None) -> list[dict[str, Any]]:
        exclude_pids = exclude_pids or set()
        fetch_k = min(max(top_k + len(exclude_pids), top_k), 200)

        resp = requests.get(
            f"{self.remote_url}/api/search",
            params={"query": query, "k": fetch_k},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        payload = resp.json()

        rows = payload.get("topk") or payload.get("results") or []
        passages: list[dict[str, Any]] = []
        for row in rows:
            pid = row.get("doc_id", row.get("pid"))
            if pid is None:
                continue
            try:
                pid = int(pid)
            except (TypeError, ValueError):
                continue
            if pid in exclude_pids:
                continue
            passages.append(
                {
                    "doc_id": pid,
                    "title": str(row.get("title", "")).strip(),
                    "text": str(row.get("text", "")).strip(),
                    "score": float(row.get("score", 0.0) or 0.0),
                }
            )
            if len(passages) >= top_k:
                break
        return passages


def load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if path.exists():
        try:
            with open(path) as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        completed[str(r["qid"])] = r
        except Exception:
            completed = {}
    return completed


def save_checkpoint(path: Path, records: dict[str, dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        for r in records.values():
            f.write(json.dumps(r) + "\n")
    tmp.rename(path)


def infer_num_hops_for_group(variants: list[dict[str, Any]]) -> int:
    hop_counts = [len(v.get("hop_details", [])) for v in variants]
    max_hops = max(hop_counts) if hop_counts else 0
    if max_hops > 0:
        return max_hops

    first = variants[0] if variants else {}
    qid = str(first.get("base_qid") or first.get("qid") or "")
    dataset = str(first.get("dataset") or "hotpot")
    return int(get_num_hops(qid, dataset, first))


def _choose_context_candidate(
    *,
    candidates: list[dict[str, Any]],
    mode: str,
    policy: str,
    hop_num: int,
    rng: np.random.Generator,
) -> tuple[int, str | None]:
    if mode == "uniform":
        return int(rng.integers(0, len(candidates))), None
    score_key = score_key_for_metric(mode, hop_num)
    scores = [float(c.get(score_key, 0.0) or 0.0) for c in candidates]
    idx = select_index_by_policy(scores, policy=policy, rng=rng)
    return idx, score_key


def _requirements_for_question(
    *,
    requirements_map: dict[str, list[str]] | None,
    qid: str,
    question: str,
) -> list[str] | None:
    if not requirements_map:
        return None
    direct = requirements_map.get(qid)
    if direct:
        return list(direct)
    by_question = requirements_map.get(question)
    if by_question:
        return list(by_question)
    return None


def _score_candidates_for_metric(
    *,
    qid: str,
    question: str,
    hop_num: int,
    candidates: list[dict[str, Any]],
    context_passages: list[dict[str, Any]],
    gold_titles: list[str],
    metric: str,
    requirements_by_qid: dict[str, list[str]] | None,
    seer_cfg: OpenRouterConfig | None,
    seer_model: str,
    seer_prompt_style: str,
    seer_cache_dir: str | Path,
    jina_api_key: str | None,
    jina_model: str,
    jina_cache_dir: str | Path,
    jina_concurrency: int,
    decomp_threshold: float,
) -> None:
    context_titles = [str(p.get("title", "")) for p in context_passages]
    for candidate in candidates:
        gold_scores = score_gold_candidate(
            retrieved_passages=list(candidate.get("retrieved", []) or []),
            gold_titles=gold_titles,
            prior_titles=context_titles,
        )
        candidate["gold_ap"] = float(gold_scores["gold_ap"])
        candidate["gold_ap_hop_only"] = float(gold_scores["gold_ap"])
        candidate["gold_ap_cumulative"] = float(gold_scores["gold_ap_cumulative"])

    if metric in {"uniform", "gold"} or not candidates:
        return

    candidates_passages: list[list[dict[str, Any]]] = []
    for candidate in candidates:
        candidates_passages.append(
            (list(context_passages) if hop_num > 1 else []) + list(candidate.get("retrieved", []) or [])
        )

    if metric == "seer":
        if seer_cfg is None:
            raise RuntimeError("Seer context selection requires OPENROUTER_API_KEY")
        scores = asyncio.run(
            score_seer_candidates_batch(
                question=question,
                candidates_passages=candidates_passages,
                model=seer_model,
                prompt_style=seer_prompt_style,
                openrouter_config=seer_cfg,
                cache_dir=seer_cache_dir,
                request_meta={"qid": qid, "hop": hop_num},
            )
        )
        for candidate, score in zip(candidates, scores):
            candidate["seer_ap"] = float(score)
        return

    if metric in {"mmr", "decomp_binary"}:
        if not jina_api_key:
            raise RuntimeError(f"{metric} context selection requires JINA_AI_API_KEY")
        if requirements_by_qid is None:
            raise RuntimeError(f"{metric} context selection requires requirements map")
        requirements = _requirements_for_question(
            requirements_map=requirements_by_qid,
            qid=qid,
            question=question,
        )
        if not requirements:
            raise RuntimeError(f"Missing requirements for qid={qid}")
        if metric == "mmr":
            scores = asyncio.run(
                score_mmr_candidates_batch(
                    requirements=requirements,
                    candidates_passages=candidates_passages,
                    jina_api_key=jina_api_key,
                    cache_dir=jina_cache_dir,
                    jina_model=jina_model,
                    concurrency=jina_concurrency,
                )
            )
            for candidate, score in zip(candidates, scores):
                candidate["mmr_ap"] = float(score)
            return

        scores = asyncio.run(
            score_decomp_binary_candidates_batch(
                requirements=requirements,
                candidates_passages=candidates_passages,
                jina_api_key=jina_api_key,
                cache_dir=str(jina_cache_dir),
                jina_model=jina_model,
                concurrency=jina_concurrency,
                threshold=decomp_threshold,
            )
        )
        for candidate, score in zip(candidates, scores):
            candidate["decomp_binary_ap"] = float(score)
        return

    raise ValueError(f"Unsupported context-select metric: {metric}")


def process_batch(
    batch_qids: list[str],
    groups: dict[str, list[dict[str, Any]]],
    prompt_variants: list[str],
    retriever: Any,
    or_config: OpenRouterConfig | None,
    model: str,
    llm_cache: LLMCache,
    context_select: str,
    context_policy: str,
    top_k: int,
    base_seed: int,
    concurrency: int,
    requirements_by_qid: dict[str, list[str]] | None,
    seer_cfg: OpenRouterConfig | None,
    seer_model: str,
    seer_prompt_style: str,
    seer_cache_dir: str | Path,
    jina_api_key: str | None,
    jina_model: str,
    jina_cache_dir: str | Path,
    jina_concurrency: int,
    decomp_threshold: float,
    inference_url: str | None = None,
    temperature: float = 0.0,
) -> tuple[list[dict[str, Any]], int]:
    errors = 0
    states: list[dict[str, Any]] = []

    for qid in batch_qids:
        variants = groups[qid]
        qid_hash = int(hashlib.md5(qid.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(base_seed + (qid_hash % (2**31)))

        question = str(variants[0].get("question", ""))
        gold_titles = variants[0].get("gold_titles") or extract_gold_titles(variants[0])
        num_hops = max(2, infer_num_hops_for_group(variants))

        hop1_candidates_internal: list[dict[str, Any]] = []
        for v_idx, variant in enumerate(variants):
            hop_details = variant.get("hop_details", [])
            if not hop_details:
                continue
            query = parse_or_normalize_rollout_query(hop_details[0].get("query", ""))
            retrieved = [
                d for d in variant.get("retrieved", []) if int(d.get("hop", 0) or 0) == 1
            ][:top_k]
            hop1_candidates_internal.append(
                {
                    "candidate_idx": v_idx,
                    "query": query,
                    "retrieved": retrieved,
                }
            )

        if not hop1_candidates_internal:
            errors += 1
            continue

        _score_candidates_for_metric(
            qid=qid,
            question=question,
            hop_num=1,
            candidates=hop1_candidates_internal,
            context_passages=[],
            gold_titles=gold_titles,
            metric=context_select,
            requirements_by_qid=requirements_by_qid,
            seer_cfg=seer_cfg,
            seer_model=seer_model,
            seer_prompt_style=seer_prompt_style,
            seer_cache_dir=seer_cache_dir,
            jina_api_key=jina_api_key,
            jina_model=jina_model,
            jina_cache_dir=jina_cache_dir,
            jina_concurrency=jina_concurrency,
            decomp_threshold=decomp_threshold,
        )

        hop1_record = {
            "hop": 1,
            "prompt_context": "N/A",
            "candidates": [
                {
                    "candidate_idx": c["candidate_idx"],
                    "query": c["query"],
                    "retrieved": c["retrieved"],
                    "gold_ap": c.get("gold_ap", 0.0),
                    "gold_ap_cumulative": c.get("gold_ap_cumulative", c.get("gold_ap", 0.0)),
                    "seer_ap": c.get("seer_ap"),
                    "decomp_binary_ap": c.get("decomp_binary_ap"),
                    "mmr_ap": c.get("mmr_ap"),
                }
                for c in hop1_candidates_internal
            ],
        }

        state = {
            "qid": qid,
            "question": question,
            "gold_titles": gold_titles,
            "num_hops": num_hops,
            "rng": rng,
            "hops": [hop1_record],
            "selected_context_passages": [],
            "selected_context_pids": set(),
            "next_context_source": None,
        }

        if num_hops > 1:
            selected_idx, score_key = _choose_context_candidate(
                candidates=hop1_candidates_internal,
                mode=context_select,
                policy=context_policy,
                hop_num=1,
                rng=rng,
            )
            selected = hop1_candidates_internal[selected_idx]
            _extend_selected_context(state, selected["retrieved"])
            state["next_context_source"] = {
                "method": f"{context_policy}_{context_select}",
                "selection_hop": 1,
                "selected_candidate_idx": selected_idx,
                "selected_query": selected.get("query", ""),
                "selected_score_key": score_key,
                "selected_score": selected.get(score_key) if score_key else None,
            }

        states.append(state)

    if not states:
        return [], errors

    max_hops = max(s["num_hops"] for s in states)

    for hop_num in range(2, max_hops + 1):
        active_states = [s for s in states if s["num_hops"] >= hop_num]
        if not active_states:
            continue

        call_specs: list[tuple[str, str]] = []
        call_index: list[tuple[int, int]] = []  # (active_idx, variant_idx)

        for active_idx, state in enumerate(active_states):
            context_str = format_context_from_passages(state["selected_context_passages"])
            for v_idx, template in enumerate(prompt_variants):
                rendered = safe_format_prompt(template, context=context_str, question=state["question"])
                cache_key = LLMCache.make_key(state["question"], context_str, template)
                call_specs.append((cache_key, rendered))
                call_index.append((active_idx, v_idx))

        if inference_url:
            print(
                f"  Hop {hop_num}: {len(call_specs)} local inference calls "
                f"({len(active_states)} questions × {len(prompt_variants)} variants)...",
                flush=True,
            )
            query_results = local_inference_batch(
                call_specs,
                inference_url,
                temperature,
                llm_cache,
                batch_size=min(8, concurrency),
            )
        else:
            print(
                f"  Hop {hop_num}: {len(call_specs)} OpenRouter calls "
                f"({len(active_states)} questions × {len(prompt_variants)} variants)...",
                flush=True,
            )
            query_results = asyncio.run(
                async_openrouter_batch(call_specs, or_config, model, llm_cache, concurrency)
            )

        queries_by_idx: dict[tuple[int, int], str | None] = {}
        for (cache_key, _), idx_pair in zip(call_specs, call_index):
            queries_by_idx[idx_pair] = query_results.get(cache_key)

        missing_queries = [idx for idx, q in queries_by_idx.items() if not q]
        if missing_queries:
            missing_ratio = len(missing_queries) / max(1, len(call_index))
            if missing_ratio > 0.10:
                raise RuntimeError(
                    f"Missing generated queries at hop={hop_num}: {len(missing_queries)}/{len(call_index)} "
                    f"(>10%, aborting batch)"
                )

            by_state_success: dict[int, list[str]] = defaultdict(list)
            for (state_idx, _variant_idx), query in queries_by_idx.items():
                if query:
                    by_state_success[state_idx].append(query)

            repaired = 0
            question_fallback = 0
            for state_idx, variant_idx in missing_queries:
                siblings = by_state_success.get(state_idx, [])
                if siblings:
                    queries_by_idx[(state_idx, variant_idx)] = siblings[0]
                    repaired += 1
                    continue

                fallback = _normalize_query_text(active_states[state_idx]["question"])
                queries_by_idx[(state_idx, variant_idx)] = fallback
                repaired += 1
                question_fallback += 1

            print(
                f"    Recovered {repaired}/{len(missing_queries)} missing queries at hop {hop_num} "
                f"(question-fallback={question_fallback})",
                flush=True,
            )

        print(f"  Hop {hop_num}: {len(call_specs)} ColBERT retrievals (serial)...", flush=True)
        passages_by_idx: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for i, idx_pair in enumerate(call_index):
            state_idx, variant_idx = idx_pair
            state = active_states[state_idx]
            query = queries_by_idx.get(idx_pair)
            if not query:
                passages_by_idx[idx_pair] = []
                continue
            passages = retriever.retrieve(
                query,
                top_k=top_k,
                exclude_pids=set(state["selected_context_pids"]),
            )
            passages_by_idx[idx_pair] = passages
            if (i + 1) % 100 == 0 or (i + 1) == len(call_index):
                print(f"    ColBERT: {i+1}/{len(call_index)}", flush=True)

        for active_idx, state in enumerate(active_states):
            context_str = format_context_from_passages(state["selected_context_passages"])

            candidates_internal: list[dict[str, Any]] = []
            for v_idx in range(len(prompt_variants)):
                idx_pair = (active_idx, v_idx)
                query = queries_by_idx.get(idx_pair) or ""
                retrieved = passages_by_idx.get(idx_pair, [])
                candidates_internal.append(
                    {
                        "candidate_idx": v_idx,
                        "query": query,
                        "retrieved": retrieved,
                    }
                )

            _score_candidates_for_metric(
                qid=state["qid"],
                question=state["question"],
                hop_num=hop_num,
                candidates=candidates_internal,
                context_passages=list(state["selected_context_passages"]),
                gold_titles=list(state["gold_titles"]),
                metric=context_select,
                requirements_by_qid=requirements_by_qid,
                seer_cfg=seer_cfg,
                seer_model=seer_model,
                seer_prompt_style=seer_prompt_style,
                seer_cache_dir=seer_cache_dir,
                jina_api_key=jina_api_key,
                jina_model=jina_model,
                jina_cache_dir=jina_cache_dir,
                jina_concurrency=jina_concurrency,
                decomp_threshold=decomp_threshold,
            )

            hop_record: dict[str, Any] = {
                "hop": hop_num,
                "prompt_context": context_str,
                "context_source": state.get("next_context_source") or {},
                "exclude_pids": sorted(state["selected_context_pids"]),
                "candidates": candidates_internal,
            }
            state["hops"].append(hop_record)

            if hop_num < state["num_hops"] and candidates_internal:
                selected_idx, score_key = _choose_context_candidate(
                    candidates=candidates_internal,
                    mode=context_select,
                    policy=context_policy,
                    hop_num=hop_num,
                    rng=state["rng"],
                )
                selected = candidates_internal[selected_idx]
                _extend_selected_context(state, selected["retrieved"])
                state["next_context_source"] = {
                    "method": f"{context_policy}_{context_select}",
                    "selection_hop": hop_num,
                    "selected_candidate_idx": selected_idx,
                    "selected_query": selected.get("query", ""),
                    "selected_score_key": score_key,
                    "selected_score": selected.get(score_key) if score_key else None,
                }

    records: list[dict[str, Any]] = []
    for state in states:
        record = {
            "qid": state["qid"],
            "question": state["question"],
            "gold_titles": state["gold_titles"],
            "num_hops": state["num_hops"],
            "hops": state["hops"],
        }
        for hop_record in state["hops"]:
            record[f"hop{hop_record['hop']}"] = hop_record
        records.append(record)

    return records, errors


def load_prompt_variants(path: str | None) -> list[str]:
    if path is None:
        return [QUERY_GENERATION_PROMPT]

    with open(path) as f:
        payload = json.load(f)

    if isinstance(payload, list):
        variants = [str(p) for p in payload if str(p).strip()]
    elif isinstance(payload, dict):
        if isinstance(payload.get("prompts"), list):
            variants = [str(p) for p in payload["prompts"] if str(p).strip()]
        else:
            variants = [str(v) for v in payload.values() if isinstance(v, str) and v.strip()]
    else:
        raise ValueError(f"Unsupported prompt variant payload type: {type(payload)!r}")

    if not variants:
        raise ValueError(f"No prompt variants found in {path}")
    return variants


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate multi-hop DPO rollouts with shared context selection"
    )
    parser.add_argument("--input", required=True, help="Variant rollouts JSONL")
    parser.add_argument("--output", required=True, help="Output DPO rollouts JSONL")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--model",
        default="meta-llama/llama-3-8b-instruct",
        help="OpenRouter model for hop>1 generation",
    )
    parser.add_argument(
        "--prompt-variants",
        default=None,
        help="Optional JSON file with prompt templates; if omitted, uses one built-in default template.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k-per-hop", type=int, default=3)
    parser.add_argument(
        "--context-select",
        default="seer",
        choices=["seer", "decomp_binary", "mmr", "gold", "uniform"],
        help="Metric used to select shared context at every hop",
    )
    parser.add_argument(
        "--context-policy",
        default="weighted",
        choices=["weighted", "best"],
        help="Selection policy for shared-context choice at every hop",
    )
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--seer-model", default="openai/gpt-4o-mini")
    parser.add_argument(
        "--seer-prompt-style",
        default="xml_fewshot",
        choices=["xml", "xml_strict", "xml_fewshot", "xml_v2", "xml_v2b", "xml_v3_hybrid", "xml_optimized"],
    )
    parser.add_argument(
        "--seer-cache-dir",
        default=None,
        help="Cache dir for Seer metric scoring (default: <output>.seer_cache)",
    )
    parser.add_argument(
        "--requirements",
        default=None,
        help="JSON requirements map keyed by qid for mmr/decomp_binary selection",
    )
    parser.add_argument(
        "--auto-requirements-out",
        default=None,
        help="Path to write auto-generated requirements map for mmr/decomp_binary",
    )
    parser.add_argument(
        "--requirements-model",
        default="openai/gpt-4o-mini",
        help="OpenRouter model used for auto requirements generation",
    )
    parser.add_argument("--requirements-concurrency", type=int, default=20)
    parser.add_argument("--jina-model", default="jina-reranker-v3")
    parser.add_argument("--jina-concurrency", type=int, default=16)
    parser.add_argument(
        "--jina-cache-dir",
        default=None,
        help="Cache dir for Jina reranker calls (default: <output>.jina_cache)",
    )
    parser.add_argument("--decomp-threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--inference-url",
        default=None,
        help="LLM inference server URL (/v1/chat_batch). If set, skip OpenRouter.",
    )
    parser.add_argument(
        "--remote-url",
        default=None,
        help="Remote ColBERT retriever URL (/api/search). If set, skip local index loading.",
    )
    parser.add_argument("--index-path", default=None, help="Local ColBERT index path")
    parser.add_argument("--collection-path", default=None, help="Local collection dir path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate config, inputs, prompt variants, and external-service setup without retrieval or generation.",
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    config = yaml.safe_load(Path(args.config).read_text())
    raw_dir = Path(config["raw_dir"])
    logger = get_logger("dpo_rollouts", Path("logs/dpo_rollouts.log"))

    if args.prompt_variants:
        print(f"Loading prompt variants from {args.prompt_variants}...", flush=True)
    else:
        print("Using built-in default prompt variant.", flush=True)
    prompt_variants = load_prompt_variants(args.prompt_variants)
    print(f"  {len(prompt_variants)} prompt variants loaded", flush=True)

    print(f"Loading variant rollouts from {args.input}...", flush=True)
    rollouts = read_jsonl(args.input)
    print(f"  {len(rollouts)} rollout records", flush=True)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rollout in rollouts:
        base_qid = str(rollout.get("base_qid") or rollout.get("qid") or "")
        groups[base_qid].append(rollout)

    all_keys = list(groups.keys())
    if args.offset > 0:
        all_keys = all_keys[args.offset :]
        print(f"  Offset: skipping first {args.offset} questions", flush=True)
    if args.limit is not None:
        all_keys = all_keys[: args.limit]
    groups = {k: groups[k] for k in all_keys}

    print(f"  {len(groups)} unique questions", flush=True)
    variant_counts = [len(v) for v in groups.values()] if groups else [0]
    print(
        f"  Variants/question: min={min(variant_counts)}, max={max(variant_counts)}, "
        f"median={sorted(variant_counts)[len(variant_counts)//2]}",
        flush=True,
    )

    output_path = Path(args.output)
    checkpoint_path = output_path.with_suffix(".checkpoint.jsonl")
    cache_path = output_path.with_suffix(".cache.json")

    completed = load_checkpoint(checkpoint_path)
    llm_cache = LLMCache(cache_path)

    remaining_qids = [qid for qid in groups if qid not in completed]
    print(
        f"\nCheckpoint: {len(completed)} completed, {len(remaining_qids)} remaining",
        flush=True,
    )
    hits, misses, cached = llm_cache.stats
    print(f"LLM cache: {cached} entries loaded (hits={hits}, misses={misses})", flush=True)

    if not remaining_qids:
        print("All questions already completed. Writing output...", flush=True)
        records = list(completed.values())
        ensure_dir(output_path.parent)
        with open(output_path, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        print(f"Output: {output_path} ({len(records)} records)")
        return

    if args.remote_url:
        print(f"Using REMOTE retriever: {args.remote_url}", flush=True)
        retriever = None if args.validate_only else RemoteColBERTRetriever(args.remote_url)
    else:
        index_path = Path(args.index_path) if args.index_path else raw_dir / "colbert_index" / "wiki2017"
        collection_path = Path(args.collection_path) if args.collection_path else raw_dir / "colbert_index" / "collection"
        if args.validate_only:
            print(
                f"Would use LOCAL ColBERT assets: index={index_path} collection={collection_path}",
                flush=True,
            )
            retriever = None
        else:
            print("Loading LOCAL ColBERT index...", flush=True)
            retriever = init_retriever_with_retry(index_path, collection_path, logger)

    generation_or_cfg = None
    if args.inference_url:
        print(f"Using LOCAL inference server: {args.inference_url}", flush=True)
    else:
        judges_cfg = yaml.safe_load(Path("configs/judges.yaml").read_text())
        generation_or_cfg = OpenRouterConfig(
            base_url=judges_cfg["openrouter"]["base_url"],
            api_key=os.getenv(judges_cfg["openrouter"]["api_key_env"], ""),
            concurrency=args.concurrency,
            temperature=args.temperature,
        )
        if not generation_or_cfg.api_key and not args.validate_only:
            raise RuntimeError("OPENROUTER_API_KEY is not set")

    seer_cache_dir = args.seer_cache_dir or str(output_path.with_suffix(".seer_cache"))
    jina_cache_dir = args.jina_cache_dir or str(output_path.with_suffix(".jina_cache"))

    seer_cfg: OpenRouterConfig | None = None
    if args.context_select == "seer":
        judges_cfg = yaml.safe_load(Path("configs/judges.yaml").read_text())
        seer_cfg = OpenRouterConfig(
            base_url=judges_cfg["openrouter"]["base_url"],
            api_key=os.getenv(judges_cfg["openrouter"]["api_key_env"], ""),
            concurrency=args.concurrency,
            temperature=args.temperature,
        )
        if not seer_cfg.api_key and not args.validate_only:
            raise RuntimeError("OPENROUTER_API_KEY is required for --context-select seer")

    requirements_by_qid: dict[str, list[str]] | None = None
    if args.context_select in {"mmr", "decomp_binary"}:
        req_path: Path
        if args.requirements:
            req_path = Path(args.requirements)
            if not req_path.exists():
                raise FileNotFoundError(f"Requirements file not found: {req_path}")
            requirements_by_qid = load_requirements_map(req_path)
        else:
            auto_req_path = Path(
                args.auto_requirements_out or output_path.with_suffix(".requirements.json")
            )
            if auto_req_path.exists():
                requirements_by_qid = load_requirements_map(auto_req_path)
            else:
                judges_cfg = yaml.safe_load(Path("configs/judges.yaml").read_text())
                req_cfg = OpenRouterConfig(
                    base_url=judges_cfg["openrouter"]["base_url"],
                    api_key=os.getenv(judges_cfg["openrouter"]["api_key_env"], ""),
                    concurrency=args.requirements_concurrency,
                    temperature=0.0,
                )
                if not req_cfg.api_key and not args.validate_only:
                    raise RuntimeError(
                        "OPENROUTER_API_KEY is required to auto-generate requirements"
                    )
                if args.validate_only:
                    print(
                        f"Would auto-generate requirements to {auto_req_path} using {args.requirements_model}",
                        flush=True,
                    )
                    requirements_by_qid = {}
                else:
                    question_map = {
                        qid: str(groups[qid][0].get("question", "")).strip()
                        for qid in groups
                        if groups[qid]
                    }
                    print(
                        f"Auto-generating requirements for {len(question_map)} questions...",
                        flush=True,
                    )
                    requirements_by_qid = asyncio.run(
                        generate_requirements_map(
                            question_map,
                            model=args.requirements_model,
                            openrouter_config=req_cfg,
                            cache_dir=auto_req_path.parent / f"{auto_req_path.stem}.cache",
                            output_path=auto_req_path,
                        )
                    )
            req_path = Path(args.requirements) if args.requirements else auto_req_path
        print(f"Requirements map: {req_path} ({len(requirements_by_qid)} questions)", flush=True)
        if not args.validate_only or args.requirements or auto_req_path.exists():
            missing_qids = [
                qid
                for qid in groups
                if not _requirements_for_question(
                    requirements_map=requirements_by_qid,
                    qid=qid,
                    question=str(groups[qid][0].get("question", "")) if groups[qid] else "",
                )
            ]
            if missing_qids:
                raise RuntimeError(
                    f"Requirements missing for {len(missing_qids)} questions; first={missing_qids[0]}"
                )

    jina_api_key = None
    if args.context_select in {"mmr", "decomp_binary"}:
        jina_api_key = os.getenv("JINA_AI_API_KEY", "")
        if not jina_api_key and not args.validate_only:
            raise RuntimeError("JINA_AI_API_KEY is required for mmr/decomp_binary context selection")

    print("\nPipeline configuration:", flush=True)
    print(f"  Questions: {len(remaining_qids)} remaining / {len(groups)} total", flush=True)
    print(f"  Prompt variants: {len(prompt_variants)}", flush=True)
    print(f"  Context select: {args.context_policy}_{args.context_select}", flush=True)
    if args.context_select == "seer":
        print(
            f"  Seer scoring: model={args.seer_model}, prompt={args.seer_prompt_style}, cache={seer_cache_dir}",
            flush=True,
        )
    if args.context_select in {"mmr", "decomp_binary"}:
        print(
            f"  Jina scoring: model={args.jina_model}, concurrency={args.jina_concurrency}, cache={jina_cache_dir}",
            flush=True,
        )

    if args.validate_only:
        if not args.inference_url:
            judges_cfg = yaml.safe_load(Path("configs/judges.yaml").read_text())
            key_env = judges_cfg["openrouter"]["api_key_env"]
            print(
                f"  OpenRouter key env {key_env}: {'set' if os.getenv(key_env) else 'missing'}",
                flush=True,
            )
        if args.context_select in {"mmr", "decomp_binary"}:
            print(
                f"  JINA_AI_API_KEY: {'set' if os.getenv('JINA_AI_API_KEY') else 'missing'}",
                flush=True,
            )
        print("Validation-only check completed.", flush=True)
        return

    all_records = dict(completed)
    total_errors = 0

    batches = [
        remaining_qids[i : i + args.batch_size]
        for i in range(0, len(remaining_qids), args.batch_size)
    ]

    pbar = tqdm(batches, desc="Batches", unit="batch")
    for batch_idx, batch_qids in enumerate(pbar, start=1):
        print(
            f"\n=== Batch {batch_idx}/{len(batches)} "
            f"({len(batch_qids)} questions) ===",
            flush=True,
        )

        records, errors = process_batch(
            batch_qids=batch_qids,
            groups=groups,
            prompt_variants=prompt_variants,
            retriever=retriever,
            or_config=generation_or_cfg,
            model=args.model,
            llm_cache=llm_cache,
            context_select=args.context_select,
            context_policy=args.context_policy,
            top_k=args.top_k_per_hop,
            base_seed=args.seed,
            concurrency=args.concurrency,
            requirements_by_qid=requirements_by_qid,
            seer_cfg=seer_cfg,
            seer_model=args.seer_model,
            seer_prompt_style=args.seer_prompt_style,
            seer_cache_dir=seer_cache_dir,
            jina_api_key=jina_api_key,
            jina_model=args.jina_model,
            jina_cache_dir=jina_cache_dir,
            jina_concurrency=args.jina_concurrency,
            decomp_threshold=args.decomp_threshold,
            inference_url=args.inference_url,
            temperature=args.temperature,
        )

        for record in records:
            all_records[str(record["qid"])] = record

        total_errors += errors

        save_checkpoint(checkpoint_path, all_records)
        llm_cache.save()
        _, _, cache_entries = llm_cache.stats

        pbar.set_postfix({
            "done": len(all_records),
            "errors": total_errors,
            "cache": cache_entries,
        })

        print(
            f"Batch complete: +{len(records)} records, "
            f"errors={errors}, total={len(all_records)}",
            flush=True,
        )

    print("\nWriting final output...", flush=True)
    final_records = list(all_records.values())
    ensure_dir(output_path.parent)
    with open(output_path, "w") as f:
        for record in final_records:
            f.write(json.dumps(record) + "\n")

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    hits, misses, cached = llm_cache.stats
    print(f"\nDone. Output: {output_path}", flush=True)
    print(f"  Records: {len(final_records)}", flush=True)
    print(f"  Errors: {total_errors}", flush=True)
    print(f"  Cache: {cached} entries (hits={hits}, misses={misses})", flush=True)


if __name__ == "__main__":
    main()
