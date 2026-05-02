#!/usr/bin/env python3
"""Canonical SFT evaluator for batched query generation + ColBERT retrieval.

The evaluator now:
- Loads questions using dataset adapters (same normalization path as rollout generation)
- Uses separate endpoints for inference and retrieval where supported
- Supports variable hop datasets (e.g., MuSiQue 2/3/4 hops)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
import yaml
from dotenv import load_dotenv
import numpy as np

from seer.datasets import get_dataset_adapter
from seer.datasets.schema import NormalizedQuestion
from seer.retrieval_utils import (
    ColBERTRetriever,
    extract_gold_titles,
    init_retriever_with_retry,
)
from seer.rollout_metrics import ap_score, recall_score
from seer.util_io import ensure_dir, read_jsonl, write_jsonl
from seer.util_log import get_logger

load_dotenv()


SEARCH_QUERY_MARKER = "[[ ## search_query ## ]]"
COMPLETED_MARKER = "[[ ## completed ## ]]"
NARRATIVE_PHRASES = (
    "there is no",
    "no information",
    "the context",
    "the provided context",
    "the question asks",
    "search query should",
    "however",
)


def _normalize_dataset_name(dataset: str) -> str:
    dataset = (dataset or "").strip().lower()
    if dataset == "hotpot":
        return "hotpot_fullwiki"
    return dataset


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


def _dedupe_titles_preserve_order(titles: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for title in titles:
        norm = str(title or "").strip()
        if not norm:
            continue
        key = norm.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(norm)
    return deduped


def parse_dspy_output(response: str) -> str:
    text = str(response or "").strip()
    if not text:
        raise ValueError("Empty query-generation output")

    if SEARCH_QUERY_MARKER in text:
        payload = text.split(SEARCH_QUERY_MARKER, 1)[1]
        if COMPLETED_MARKER in payload:
            payload = payload.split(COMPLETED_MARKER, 1)[0].strip()
    else:
        payload = text

    payload = payload.strip("`")
    payload = payload.strip('"').strip("'").strip()
    payload = re.sub(r"\s+", " ", payload).strip()

    if not payload:
        raise ValueError("Parsed query is empty")

    if "[[ ##" in payload or "## ]]" in payload:
        raise ValueError(f"Residual markers in parsed query: {payload[:120]!r}")
    if _looks_narrative_like_query(payload):
        raise ValueError(f"Narrative query-generation output rejected: {payload[:200]!r}")
    return payload


def parse_dspy_output_tolerant(response: str) -> str:
    """Best-effort parser used after strict parse retries are exhausted."""
    text = str(response or "").strip()
    if not text:
        return "unknown"

    if SEARCH_QUERY_MARKER in text:
        payload = text.split(SEARCH_QUERY_MARKER, 1)[1]
        if COMPLETED_MARKER in payload:
            payload = payload.split(COMPLETED_MARKER, 1)[0].strip()
    else:
        payload = text

    payload = payload.strip("`")
    payload = payload.strip('"').strip("'").strip()
    payload = re.sub(r"\s+", " ", payload).strip()
    payload = payload.split("[[ ##", 1)[0].strip()
    payload = payload.split("## ]]", 1)[0].strip()
    return payload or "unknown"


class BatchInferenceClient:
    """Client for server-side batched generation (`/v1/generate_batch`)."""

    def __init__(self, url: str, timeout: float = 300.0):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def generate_batch(
        self,
        questions: list[str],
        contexts: list[str],
        max_retries: int = 3,
        return_raw: bool = False,
    ) -> list[str] | tuple[list[str], list[str], list[dict]]:
        instruction = (
            "Generate a search query to find information needed to answer "
            "the following question. Use the provided context if available."
        )
        requests_list = [
            {
                "instruction": instruction,
                "input_text": f"Context: {ctx}\\n\\nQuestion: {q}",
                "max_new_tokens": 128,
                "temperature": 0.0,
            }
            for q, ctx in zip(questions, contexts)
        ]

        payload = {"requests": requests_list}
        try:
            response = requests.post(
                f"{self.url}/v1/generate_batch",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            raw_responses = response.json()["responses"]
            parsed: list[str | None] = [None] * len(raw_responses)
            errors: list[dict] = [{"error": None, "fallback": False} for _ in raw_responses]
            failed_idxs: list[int] = []
            for i, raw in enumerate(raw_responses):
                try:
                    parsed[i] = parse_dspy_output(raw)
                except ValueError as e:
                    failed_idxs.append(i)
                    errors[i] = {"error": str(e), "fallback": False}

            if failed_idxs and max_retries > 0:
                retry_questions = [questions[i] for i in failed_idxs]
                retry_contexts = [contexts[i] for i in failed_idxs]
                retry_parsed, retry_raw, retry_errors = self.generate_batch(
                    retry_questions,
                    retry_contexts,
                    max_retries=max_retries - 1,
                    return_raw=True,
                )
                for j, orig_idx in enumerate(failed_idxs):
                    parsed[orig_idx] = retry_parsed[j]
                    raw_responses[orig_idx] = retry_raw[j]
                    errors[orig_idx] = retry_errors[j]

            for i, value in enumerate(parsed):
                if value is None:
                    parsed[i] = parse_dspy_output_tolerant(raw_responses[i])
                    errors[i] = {
                        "error": errors[i]["error"] or "strict_parse_failed",
                        "fallback": True,
                    }

            parsed_out = [str(p) for p in parsed]
            if return_raw:
                return parsed_out, raw_responses, errors
            return parsed_out
        except (requests.HTTPError, requests.Timeout, requests.ConnectionError) as e:
            is_timeout = isinstance(e, requests.Timeout)
            is_connection_error = isinstance(e, requests.ConnectionError)
            is_oom = isinstance(e, requests.HTTPError) and e.response is not None and e.response.status_code == 500
            if (is_timeout or is_oom or is_connection_error) and len(questions) > 1 and max_retries > 0:
                mid = len(questions) // 2
                left = self.generate_batch(
                    questions[:mid],
                    contexts[:mid],
                    max_retries=max_retries - 1,
                    return_raw=return_raw,
                )
                right = self.generate_batch(
                    questions[mid:],
                    contexts[mid:],
                    max_retries=max_retries - 1,
                    return_raw=return_raw,
                )
                if return_raw:
                    lp, lr, le = left
                    rp, rr, re = right
                    return lp + rp, lr + rr, le + re
                return left + right
            if is_connection_error and len(questions) == 1 and max_retries > 0:
                return self.generate_batch(
                    questions,
                    contexts,
                    max_retries=max_retries - 1,
                    return_raw=return_raw,
                )

            if is_timeout:
                raise RuntimeError(
                    f"Batch inference timed out at batch size {len(questions)}. "
                    f"Try smaller --batch-size / lower --colbert-threads."
                ) from e
            raise


class OpenRouterInferenceClient:
    """Client for OpenRouter (`/v1/chat/completions`) with async batching."""

    INSTRUCTION = (
        "Generate a search query to find information needed to answer "
        "the following question. Use the provided context if available. "
        "Output ONLY the search query text, nothing else."
    )
    REASONING_MODELS = {"openai/gpt-5", "openai/gpt-5.2", "openai/o1", "openai/o1-mini", "openai/o3-mini"}

    def __init__(
        self,
        model: str,
        api_key: str,
        concurrency: int = 20,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.concurrency = concurrency
        self.timeout = timeout
        self.max_tokens = 4096 if model in self.REASONING_MODELS else 128
        if model in self.REASONING_MODELS:
            print(f"  Reasoning model detected — using max_tokens={self.max_tokens}")

    def _build_messages(self, question: str, context: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.INSTRUCTION},
            {"role": "user", "content": f"Context: {context}\\n\\nQuestion: {question}"},
        ]

    async def _call_one(self, semaphore, httpx_client, question, context, idx):
        import httpx

        payload = {
            "model": self.model,
            "messages": self._build_messages(question, context),
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
        }
        async with semaphore:
            last_error: Exception | None = None
            for attempt in range(3):
                try:
                    response = await httpx_client.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        json=payload,
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        timeout=self.timeout,
                    )
                    response.raise_for_status()
                    payload = response.json()
                    text = (payload["choices"][0]["message"]["content"] or "").strip()
                    if not text and attempt < 2:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    if not text:
                        raise RuntimeError("OpenRouter returned empty output")
                    return idx, text
                except (httpx.HTTPStatusError, httpx.TimeoutException) as e:
                    last_error = e
                    if attempt < 2:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        raise RuntimeError(
                            f"OpenRouter call failed after retries for item {idx}"
                        ) from last_error

    async def _generate_batch_async(self, questions, contexts):
        import httpx

        sem = asyncio.Semaphore(self.concurrency)
        async with httpx.AsyncClient() as client:
            tasks = [
                self._call_one(sem, client, q, c, i)
                for i, (q, c) in enumerate(zip(questions, contexts))
            ]
            outputs = await asyncio.gather(*tasks)
        outputs.sort(key=lambda item: item[0])
        return [text for _, text in outputs]

    def generate_batch(
        self,
        questions: list[str],
        contexts: list[str],
        max_retries: int = 3,
        return_raw: bool = False,
    ) -> list[str] | tuple[list[str], list[str], list[dict]]:
        raw_responses = asyncio.run(self._generate_batch_async(questions, contexts))
        parsed: list[str | None] = [None] * len(raw_responses)
        errors: list[dict] = [{"error": None, "fallback": False} for _ in raw_responses]
        failed_idxs: list[int] = []

        for i, text in enumerate(raw_responses):
            try:
                parsed[i] = parse_dspy_output(text)
            except ValueError as e:
                failed_idxs.append(i)
                errors[i] = {"error": str(e), "fallback": False}

        if failed_idxs and max_retries > 0:
            retry_questions = [questions[i] for i in failed_idxs]
            retry_contexts = [contexts[i] for i in failed_idxs]
            retry_parsed, retry_raw, retry_errors = self.generate_batch(
                retry_questions,
                retry_contexts,
                max_retries=max_retries - 1,
                return_raw=True,
            )
            for j, orig_idx in enumerate(failed_idxs):
                parsed[orig_idx] = retry_parsed[j]
                raw_responses[orig_idx] = retry_raw[j]
                errors[orig_idx] = retry_errors[j]

        for i, value in enumerate(parsed):
            if value is None:
                parsed[i] = parse_dspy_output_tolerant(raw_responses[i])
                errors[i] = {
                    "error": errors[i]["error"] or "strict_parse_failed",
                    "fallback": True,
                }

        parsed_out = [str(p) for p in parsed]
        if return_raw:
            return parsed_out, raw_responses, errors
        return parsed_out


class LocalRetriever:
    """Local ColBERT retriever backed by local index files."""

    def __init__(self, raw_dir: Path, logger=None):
        self.logger = logger
        self.index_path = raw_dir / "colbert_index" / "wiki2017"
        self.collection_path = raw_dir / "colbert_index" / "collection"
        self._retriever = init_retriever_with_retry(self.index_path, self.collection_path, logger)

    def retrieve(self, query: str, top_k: int = 10, exclude_pids: set[int] | None = None):
        exclude = set(exclude_pids or set())
        return self._retriever.retrieve(query, top_k=top_k, exclude_pids=exclude)


class RemoteRetriever:
    """ColBERT HTTP retrieval via `serve_colbert_min.py` API."""

    def __init__(self, url: str, timeout: float = 30.0):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def retrieve(self, query: str, top_k: int = 10, exclude_pids: set[int] | None = None):
        params = {"query": query, "k": top_k + len(exclude_pids or [])}
        response = requests.get(f"{self.url}/api/search", params=params, timeout=self.timeout)
        response.raise_for_status()
        topk = response.json().get("topk", [])

        out = []
        seen: set[int] = set()
        exclude = set(exclude_pids or [])
        for row in topk:
            pid = int(row.get("pid", -1))
            if pid < 0 or pid in exclude or pid in seen:
                continue
            out.append({
                "doc_id": pid,
                "title": row.get("title", ""),
                "text": row.get("text", ""),
                "score": float(row.get("score", 0.0)),
            })
            seen.add(pid)
            if len(out) >= top_k:
                break
        return out


def _rollout_record_from_normalized(record: dict[str, Any], dataset_name: str) -> dict[str, Any]:
    """Project a normalized dataset row into rollout/eval question schema."""
    normalized = NormalizedQuestion.from_record(record)
    adapter = get_dataset_adapter(dataset_name)
    return {
        "_id": normalized.qid,
        "qid": normalized.qid,
        "question": normalized.question,
        "answer": normalized.answer,
        "gold_titles": adapter.extract_gold_titles(record),
        "dataset": dataset_name,
        "num_hops": record.get("num_hops", 2),
        "question_decomposition": record.get("question_decomposition"),
        "decomposition": record.get("decomposition"),
        "split": record.get("split", "unknown"),
        "is_answerable": record.get("is_answerable", True),
    }


def _drop_empty(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


def _clean_dataset_metadata(question: dict[str, Any], dataset: str) -> dict[str, Any]:
    adapter = get_dataset_adapter(_normalize_dataset_name(dataset))
    out = dict(question)
    out["gold_titles"] = adapter.extract_gold_titles(question)
    if "num_hops" not in out:
        out["num_hops"] = adapter.infer_num_hops(question.get("qid", ""), question)
    return out


def _load_hotpot_questions(raw_dir: Path, split: str, limit: int | None) -> list[dict[str, Any]]:
    raw_hotpot = raw_dir / "hotpot_fullwiki"
    if split == "dev":
        path = raw_hotpot / "hotpot_dev_fullwiki_v1.json"
    elif split == "train":
        sample_path = raw_hotpot / "hotpot_train_sample_10k.json"
        full_path = raw_hotpot / "hotpot_train_v1.1.json"
        if limit and limit > 10000:
            path = full_path
        else:
            path = sample_path if sample_path.exists() else full_path
    elif split == "test":
        path = raw_hotpot / "hotpot_test_fullwiki_v1.json"
    else:
        raise ValueError(f"Unknown split: {split}")

    if not path.exists():
        raise FileNotFoundError(f"Hotpot data not found at {path}")

    with open(path) as f:
        rows = json.load(f)
    if limit is not None:
        rows = rows[:limit]

    adapter = get_dataset_adapter("hotpot_fullwiki")
    normalized = [adapter.normalize_to_record(item, split=split) for item in rows]
    return [_drop_empty(_rollout_record_from_normalized(item, "hotpot_fullwiki")) for item in normalized]


def _load_musique_questions(config: dict[str, Any], split: str, limit: int | None) -> list[dict[str, Any]]:
    # Keep same source path as rollout/generation by reusing normalized artifacts.
    norm_path = Path(config["norm_dir"]) / "musique.jsonl"
    if not norm_path.exists():
        raise FileNotFoundError(f"MuSiQue normalized data not found at {norm_path}")
    rows = list(read_jsonl(norm_path))
    if limit is not None:
        rows = rows[:limit]
    adapter = get_dataset_adapter("musique")
    normalized = [adapter.normalize_to_record(item, split=split) for item in rows]
    return [_drop_empty(_rollout_record_from_normalized(item, "musique")) for item in normalized]


def load_questions(
    config: dict[str, Any],
    dataset: str,
    split: str,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    dataset = _normalize_dataset_name(dataset)
    raw_dir = Path(config["raw_dir"])
    if dataset == "hotpot_fullwiki":
        return _load_hotpot_questions(raw_dir, split, limit)
    if dataset == "musique":
        return _load_musique_questions(config, split, limit)
    raise ValueError(f"Unsupported dataset: {dataset}")


def _infer_num_hops(question: dict[str, Any], dataset: str) -> int:
    qid = str(question.get("qid", question.get("_id", "")))
    adapter = get_dataset_adapter(_normalize_dataset_name(dataset))
    return int(adapter.infer_num_hops(qid, question))


def _question_context_text(retrieved: list[dict[str, Any]]) -> str:
    if not retrieved:
        return "N/A"
    parts = []
    for i, doc in enumerate(retrieved, 1):
        title = doc.get("title", "")
        text = doc.get("text", "")
        parts.append(f"[{i}] «{title}: {text}»")
    return "\\n".join(parts)


def retrieve_batch_parallel(
    retriever: LocalRetriever | RemoteRetriever,
    queries: list[str],
    exclude_pids_list: list[set[int]],
    top_k: int,
    max_workers: int = 8,
    serialize: bool = False,
) -> list[list[dict[str, Any]]]:
    if not queries:
        return []
    if serialize:
        return [retriever.retrieve(q, top_k=top_k, exclude_pids=exclude_pids_list[i]) for i, q in enumerate(queries)]

    def _one(idx: int):
        return idx, retriever.retrieve(queries[idx], top_k=top_k, exclude_pids=exclude_pids_list[idx])

    out = [None] * len(queries)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_one, i): i for i in range(len(queries))}
        for fut in as_completed(futures):
            idx, payload = fut.result()
            out[idx] = payload
    return out


def evaluate_chunk(
    questions: list[dict[str, Any]],
    inference_client: BatchInferenceClient | OpenRouterInferenceClient,
    retriever: LocalRetriever | RemoteRetriever,
    top_k_per_hop: int,
    colbert_threads: int,
    serialize_retrieval: bool,
    trace_dir: Path | None,
    trace_include_context: bool,
    trace_include_doc_text: bool,
    chunk_idx: int,
    dataset: str,
    num_hops: int | None = None,
    logger=None,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, list[float]]]]:
    qids = [q.get("_id", q.get("qid", "")) for q in questions]
    all_retrievals = {qid: [] for qid in qids}
    seen_pids = {qid: set() for qid in qids}
    seen_titles = {qid: set() for qid in qids}
    hop_details = {qid: [] for qid in qids}

    hop_targets = (
        [int(num_hops)] * len(questions)
        if num_hops is not None
        else [int(_infer_num_hops(q, dataset)) for q in questions]
    )
    max_hops = max(hop_targets) if hop_targets else 0
    hop_metrics: dict[int, dict[str, list[float]]] = {
        h: {"recall": [], "ap": []} for h in range(1, max_hops + 1)
    }

    for hop in range(1, max_hops + 1):
        active_idxs = [i for i, target in enumerate(hop_targets) if target >= hop]
        if not active_idxs:
            break

        print(f"  Hop {hop}: Generating queries...")
        if logger:
            logger.info("Hop %s: Generating queries...", hop)
        context_rows = []
        batch_questions = []
        batch_contexts = []
        trace_rows: list[dict[str, Any]] = []

        for idx in active_idxs:
            qid = qids[idx]
            question_text = questions[idx].get("question", "")
            context_str = _question_context_text(all_retrievals[qid])
            batch_questions.append(question_text)
            batch_contexts.append(context_str)
            if trace_dir is not None:
                input_text = f"Context: {context_str}\\n\\nQuestion: {question_text}"
                trace_rows.append(
                    {
                        "qid": qid,
                        "hop": hop,
                        "chunk_idx": chunk_idx,
                        "question": question_text,
                        "instruction": "Generate a search query to find information needed to answer the following question. Use the provided context if available.",
                        "input_text": input_text if trace_include_context else None,
                        "input_text_len": len(input_text),
                        "context_len": len(context_str),
                        "top_k_per_hop": top_k_per_hop,
                    }
                )

        if trace_dir is not None:
            queries, raw_responses, inference_errors = inference_client.generate_batch(
                batch_questions,
                batch_contexts,
                return_raw=True,
            )
        else:
            queries = inference_client.generate_batch(batch_questions, batch_contexts)
            raw_responses = inference_errors = None

        print(f"  Hop {hop}: Retrieving passages...")
        if logger:
            logger.info("Hop %s: Retrieving passages...", hop)
        exclude_lists = [seen_pids[qids[idx]] for idx in active_idxs]
        retrieval_results = retrieve_batch_parallel(
            retriever,
            queries,
            exclude_lists,
            top_k_per_hop,
            max_workers=colbert_threads,
            serialize=serialize_retrieval,
        )

        for local_pos, idx in enumerate(active_idxs):
            qid = qids[idx]
            query = queries[local_pos]
            passages = retrieval_results[local_pos]

            for p in passages:
                title = str(p.get("title", "")).strip()
                title_key = title.lower()
                if (
                    p.get("doc_id") not in seen_pids[qid]
                    and title
                    and title_key not in seen_titles[qid]
                ):
                    seen_pids[qid].add(p.get("doc_id"))
                    seen_titles[qid].add(title_key)
                    p["hop"] = hop
                    all_retrievals[qid].append(p)

            gold_titles = extract_gold_titles(questions[idx], dataset=dataset)
            current_titles = _dedupe_titles_preserve_order(
                [p.get("title", "") for p in all_retrievals[qid]]
            )
            recall = recall_score(current_titles, gold_titles)
            ap = ap_score(current_titles, gold_titles)
            hop_metrics[hop]["recall"].append(recall)
            hop_metrics[hop]["ap"].append(ap)
            hop_details[qid].append(
                {
                    "hop": hop,
                    "query": query,
                    "recall_after_hop": recall,
                    "ap_after_hop": ap,
                }
            )

            if trace_dir is not None:
                trace_rows[local_pos]["raw_response"] = raw_responses[local_pos]
                trace_rows[local_pos]["parsed_query"] = queries[local_pos]
                trace_rows[local_pos]["inference_error"] = inference_errors[local_pos]
                trace_rows[local_pos]["retrieved_count"] = len(passages)
                trace_rows[local_pos]["exclude_pids_count"] = len(exclude_lists[local_pos])
                trace_rows[local_pos]["retrieved_doc_ids"] = [p.get("doc_id") for p in passages]
                trace_rows[local_pos]["retrieved_titles"] = [p.get("title") for p in passages]
                trace_rows[local_pos]["retrieved_scores"] = [p.get("score") for p in passages]
                if trace_include_doc_text:
                    trace_rows[local_pos]["retrieved_texts"] = [p.get("text") for p in passages]

        if trace_dir is not None:
            trace_path = trace_dir / f"trace_hop{hop}.jsonl"
            for row in trace_rows:
                with open(trace_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\\n")

        hop_recall = np.mean(hop_metrics[hop]["recall"]) * 100 if hop_metrics[hop]["recall"] else 0.0
        hop_ap = np.mean(hop_metrics[hop]["ap"]) * 100 if hop_metrics[hop]["ap"] else 0.0
        print(f"  Hop {hop} - Recall: {hop_recall:.2f}%, AP: {hop_ap:.2f}%")
        if logger:
            logger.info("Hop %s - Recall: %.2f%%, AP: %.2f%%", hop, hop_recall, hop_ap)

    rollouts: list[dict[str, Any]] = []
    for idx, q in enumerate(questions):
        qid = qids[idx]
        gold_titles = extract_gold_titles(q, dataset=dataset)
        titles = _dedupe_titles_preserve_order([p.get("title", "") for p in all_retrievals[qid]])
        record = {
            "qid": qid,
            "question": q.get("question", ""),
            "answer": q.get("answer", ""),
            "gold_titles": gold_titles,
            "dataset": dataset,
            "num_hops": int(_infer_num_hops(q, dataset)),
            "retrieved": all_retrievals[qid],
            "hop_details": hop_details[qid],
            "gold_coverage": recall_score(titles, gold_titles),
        }
        record.update(_clean_dataset_metadata(q, dataset))
        rollouts.append(record)

    return rollouts, hop_metrics


def _build_output_path(config: dict[str, Any], args: argparse.Namespace) -> Path:
    if args.output:
        return Path(args.output)
    rollouts_dir = Path(config.get("rollouts_dir", "data/rollouts"))
    suffix = f"_{args.limit}" if args.limit else ""
    return rollouts_dir / f"{args.dataset}_{args.split}_eval{suffix}.jsonl"


def _build_checkpoint_path(output_path: Path) -> Path:
    return output_path.parent / f"{output_path.stem}.checkpoint.jsonl"


def _build_partial_path(output_path: Path) -> Path:
    return output_path.parent / f"{output_path.stem}.partial.jsonl"


def _resolve_inference_client(args: argparse.Namespace):
    if args.openrouter_model:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY required when --openrouter-model is set")
        return OpenRouterInferenceClient(
            model=args.openrouter_model,
            api_key=api_key,
            concurrency=args.openrouter_concurrency,
            timeout=args.timeout,
        )
    if args.inference_url:
        return BatchInferenceClient(args.inference_url, timeout=args.timeout)
    raise ValueError("Must pass --inference-url or --openrouter-model")


def _resolve_retriever(config: dict[str, Any], args: argparse.Namespace, logger=None):
    if args.retrieval_url:
        return RemoteRetriever(args.retrieval_url, timeout=args.timeout)
    return LocalRetriever(Path(config["raw_dir"]), logger=logger)


def _merge_metrics(base: dict[int, dict[str, list[float]]], chunk_metrics: dict[int, dict[str, list[float]]]) -> None:
    for hop, values in chunk_metrics.items():
        base.setdefault(hop, {"recall": [], "ap": []})
        base[hop]["recall"].extend(values["recall"])
        base[hop]["ap"].extend(values["ap"])


def _rebuild_metrics_from_rollouts(rollouts: list[dict[str, Any]]) -> dict[int, dict[str, list[float]]]:
    metrics: dict[int, dict[str, list[float]]] = {}
    for row in rollouts:
        for hop_row in row.get("hop_details", []) or []:
            try:
                hop = int(hop_row.get("hop"))
                recall = float(hop_row.get("recall_after_hop"))
                ap = float(hop_row.get("ap_after_hop"))
            except (TypeError, ValueError):
                continue
            metrics.setdefault(hop, {"recall": [], "ap": []})
            metrics[hop]["recall"].append(recall)
            metrics[hop]["ap"].append(ap)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Batched SFT evaluator.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", default="hotpot", choices=["hotpot", "musique"])
    parser.add_argument("--split", default="dev")
    parser.add_argument("--limit", type=int, default=0, help="Limit questions (0 = all)")
    parser.add_argument("--top-k-per-hop", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--colbert-threads", type=int, default=8)
    parser.add_argument("--serialize-retrieval", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--trace-dir")
    parser.add_argument("--trace-include-context", action="store_true")
    parser.add_argument("--trace-include-doc-text", action="store_true")
    parser.add_argument("--openrouter-model")
    parser.add_argument("--openrouter-concurrency", type=int, default=20)
    parser.add_argument("--inference-url", help="Inference service URL (for /v1/generate_batch)")
    parser.add_argument("--retrieval-url", help="ColBERT service URL (for /api/search)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    config = yaml.safe_load(Path(args.config).read_text())
    logs_dir = Path(config["logs_dir"])
    logger = get_logger("sft-eval", logs_dir / "evaluate_batched.log")

    dataset = _normalize_dataset_name(args.dataset)
    questions = load_questions(config, dataset, args.split, args.limit if args.limit > 0 else None)
    print(f"Loaded {len(questions)} questions from {dataset} {args.split}")

    inference_client = _resolve_inference_client(args)
    if isinstance(inference_client, BatchInferenceClient):
        if args.inference_url:
            print(f"Using inference server at {args.inference_url}")
        logger.info("Using server inference")
    else:
        print(f"Using OpenRouter model: {args.openrouter_model}")
        logger.info("Using OpenRouter model: %s", args.openrouter_model)

    retriever = _resolve_retriever(config, args, logger=logger)
    if args.retrieval_url:
        print(f"Using remote ColBERT index at {args.retrieval_url}")
        logger.info("Using remote ColBERT at %s", args.retrieval_url)
    else:
        print("Using local ColBERT index")

    output_path = _build_output_path(config, args)
    checkpoint_path = _build_checkpoint_path(output_path)
    partial_path = _build_partial_path(output_path)

    start_chunk = 0
    all_rollouts: list[dict[str, Any]] = []
    all_metrics: dict[int, dict[str, list[float]]] = {}

    if args.resume:
        checkpoint_rollouts: list[dict[str, Any]] = []
        checkpoint_metrics: dict[int, dict[str, list[float]]] = {}
        checkpoint_start_chunk: int | None = None

        if checkpoint_path.exists():
            with open(checkpoint_path) as f:
                payload = json.load(f)
            checkpoint_start_chunk = int(payload.get("last_chunk", -1)) + 1
            checkpoint_rollouts = payload.get("rollouts", [])
            checkpoint_metrics = payload.get("metrics", {})
            if checkpoint_metrics:
                checkpoint_metrics = {int(k): v for k, v in checkpoint_metrics.items()}

        partial_rollouts: list[dict[str, Any]] = []
        partial_start_chunk: int | None = None
        partial_metrics: dict[int, dict[str, list[float]]] = {}
        if partial_path.exists():
            partial_rollouts = read_jsonl(partial_path)
            if partial_rollouts:
                partial_start_chunk = len(partial_rollouts) // args.batch_size
                partial_metrics = _rebuild_metrics_from_rollouts(partial_rollouts)

        if partial_rollouts and len(partial_rollouts) > len(checkpoint_rollouts):
            all_rollouts = partial_rollouts
            all_metrics = partial_metrics
            start_chunk = partial_start_chunk or 0
            print(
                f"Resuming from partial at chunk {start_chunk + 1} "
                f"({len(all_rollouts)} completed rollouts)"
            )
        elif checkpoint_start_chunk is not None:
            start_chunk = checkpoint_start_chunk
            all_rollouts = checkpoint_rollouts
            all_metrics = checkpoint_metrics
            print(f"Resuming from chunk {start_chunk + 1}")
        elif partial_rollouts:
            all_rollouts = partial_rollouts
            all_metrics = partial_metrics
            start_chunk = partial_start_chunk or 0
            print(
                f"Resuming from partial at chunk {start_chunk + 1} "
                f"({len(all_rollouts)} completed rollouts)"
            )

    trace_dir = Path(args.trace_dir) if args.trace_dir else None
    trace_include_context = args.trace_include_context or args.trace_dir is not None
    trace_include_doc_text = args.trace_include_doc_text
    if trace_dir is not None:
        ensure_dir(trace_dir)

    num_chunks = (len(questions) + args.batch_size - 1) // args.batch_size
    start_time = time.time()

    for chunk_idx in range(start_chunk, num_chunks):
        start = chunk_idx * args.batch_size
        end = min(start + args.batch_size, len(questions))
        chunk = questions[start:end]

        print(f"\\n{'=' * 60}")
        print(f"Chunk {chunk_idx + 1}/{num_chunks}: questions {start}-{end-1}")
        print(f"{'=' * 60}")
        logger.info("%s", "=" * 60)
        logger.info("Chunk %s/%s: questions %s-%s", chunk_idx + 1, num_chunks, start, end - 1)
        logger.info("%s", "=" * 60)

        chunk_rollouts, chunk_metrics = evaluate_chunk(
            questions=chunk,
            inference_client=inference_client,
            retriever=retriever,
            top_k_per_hop=args.top_k_per_hop,
            colbert_threads=args.colbert_threads,
            serialize_retrieval=args.serialize_retrieval,
            trace_dir=trace_dir,
            trace_include_context=trace_include_context,
            trace_include_doc_text=trace_include_doc_text,
            chunk_idx=chunk_idx,
            dataset=dataset,
            logger=logger,
        )
        all_rollouts.extend(chunk_rollouts)
        _merge_metrics(all_metrics, chunk_metrics)

        write_jsonl(partial_path, all_rollouts)
        if (chunk_idx + 1) % args.checkpoint_interval == 0:
            checkpoint_payload = {
                "last_chunk": chunk_idx,
                "total_chunks": num_chunks,
                "rollouts": all_rollouts,
                "metrics": all_metrics,
                "timestamp": time.time(),
            }
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint_payload, f)
            print(f"  ✓ Checkpoint saved ({len(all_rollouts)} rollouts)")
            logger.info("Checkpoint saved at chunk %s", chunk_idx + 1)

    elapsed = time.time() - start_time

    ensure_dir(output_path.parent)
    write_jsonl(output_path, all_rollouts)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    if partial_path.exists():
        partial_path.unlink()

    print("\\n" + "=" * 60)
    print("Evaluation Complete!")
    print(f"Time: {elapsed:.2f}s ({len(questions) / elapsed:.2f} questions/sec)")
    print(f"Results: {output_path}")
    print("=" * 60)

    for hop in sorted(all_metrics):
        hop_recall = np.mean(all_metrics[hop]["recall"]) * 100 if all_metrics[hop]["recall"] else 0.0
        hop_ap = np.mean(all_metrics[hop]["ap"]) * 100 if all_metrics[hop]["ap"] else 0.0
        print(f"Hop {hop} - Recall: {hop_recall:.2f}%, AP: {hop_ap:.2f}%")
        logger.info("Hop %s - Recall: %.2f%%, AP: %.2f%%", hop, hop_recall, hop_ap)


if __name__ == "__main__":
    main()
