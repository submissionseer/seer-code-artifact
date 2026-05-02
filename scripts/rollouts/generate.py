"""
Generate multi-hop retrieval rollouts using ColBERTv2 over Wikipedia.

This implements the standard multi-hop retrieval pipeline:
- HotpotQA: 2 hops (fixed)
- MuSiQue: 2-4 hops (variable, based on QID)

For each hop after the first, an LLM generates a follow-up query
conditioned on the question and previously retrieved context.
"""
from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import os
import random
import re
import time
from array import array
from pathlib import Path
from typing import Optional
import threading

import httpx
import numpy as np
import transformers
import yaml
from tqdm import tqdm

from seer.datasets import get_dataset_adapter
from seer.datasets.schema import NormalizedQuestion
from seer.util_io import ensure_dir, read_jsonl, write_jsonl
from seer.util_log import get_logger
from seer.prompt_variants import prompts_from_dspy_state
from seer.util_text import safe_format_prompt
from seer.openrouter import normalize_model_slug
from seer.retrieval_utils import (
    extract_gold_titles as shared_extract_gold_titles,
    get_num_hops as shared_get_num_hops,
)

if not hasattr(transformers.modeling_utils.PreTrainedModel, "all_tied_weights_keys"):
    def _get_all_tied_weights_keys(self):
        value = getattr(self, "_tied_weights_keys", None)
        return value or {}

    def _set_all_tied_weights_keys(self, value):
        setattr(self, "_tied_weights_keys", value or {})

    transformers.modeling_utils.PreTrainedModel.all_tied_weights_keys = property(
        _get_all_tied_weights_keys,
        _set_all_tied_weights_keys,
    )


# ============================================
# Query Generation (Same for ALL hops, matching LeReT)
# ============================================

# LeReT-style query generation: same signature for all hops
# Input: (context, question) -> search_query
# - Hop 1: context is empty
# - Hop 2+: context is accumulated passages from previous hops

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
    """Extract a single query from model output.

    Required format:
    - DSPy markers: [[ ## search_query ## ]] ... [[ ## completed ## ]]
      Some fine-tuned policies omit the trailing completed marker while still
      returning a clean search-query payload. In that case, accept the
      remainder of the response as the candidate query.

    Raises:
      ValueError if output is missing required marker or query payload.
    """
    text = str(response or "").strip()
    if not text:
        raise ValueError("Empty query-generation response")

    if SEARCH_QUERY_MARKER not in text:
        raise ValueError(
            "Missing required '[[ ## search_query ## ]]' marker in query-generation output: "
            f"{text[:200]!r}"
        )

    candidate = text.split(SEARCH_QUERY_MARKER, 1)[1]
    if COMPLETED_MARKER in candidate:
        candidate = candidate.split(COMPLETED_MARKER, 1)[0]
    elif "[[ ##" in candidate and "## ]]" in candidate:
        raise ValueError(
            "Missing required '[[ ## completed ## ]]' marker in query-generation output: "
            f"{text[:200]!r}"
        )
    candidate = candidate.strip()
    if not candidate:
        raise ValueError("DSPy output missing search_query payload")
    query = _normalize_query_text(candidate)
    if "[[ ##" in query or "## ]]" in query:
        raise ValueError(f"Residual markers in parsed query: {query[:120]!r}")
    if _looks_narrative_like_query(query):
        raise ValueError(f"Narrative query-generation output rejected: {query[:200]!r}")
    return query


def _normalize_query_text(candidate: str) -> str:
    query = str(candidate or "").strip()
    # Strip common wrappers from quoted/code-formatted answers.
    query = query.strip("`")
    query = query.strip('"').strip("'").strip()
    query = re.sub(r"\s+", " ", query).strip()
    if not query:
        raise ValueError("Parsed query is empty")
    return query


def _fallback_query_from_question(question: str) -> str:
    query = _normalize_query_text(question)
    if not query:
        raise ValueError("Unable to build fallback query from empty question")
    return query


class QueryGenerator:
    """Generate search queries using an LLM (same model for all hops, LeReT-style)."""

    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-4o-mini",
        logger=None,
        cache_path: Path | None = None,
        prompt_variants: list[str] | None = None,
        use_cache: bool = True,
        temperature: float = 0.3,
        inference_url: str | None = None,
        request_timeout_s: float = 45.0,
        max_retries: int = 4,
        retry_backoff_s: float = 1.5,
        max_output_tokens: int = 512,
    ):
        self.api_key = api_key
        raw_model = model.replace("openrouter/", "", 1)
        self.model = normalize_model_slug(raw_model)
        self.logger = logger
        self.inference_url = inference_url
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.cache_path = cache_path
        self.use_cache = use_cache
        self.temperature = temperature
        self.request_timeout_s = max(1.0, float(request_timeout_s))
        self.max_retries = max(1, int(max_retries))
        self.retry_backoff_s = max(0.0, float(retry_backoff_s))
        self.max_output_tokens = max(32, int(max_output_tokens))
        # Cache key includes model identity to prevent cross-model contamination
        self._cache_model_id = inference_url or self.model
        self.cache = self._load_cache()
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_lock = threading.Lock()
        self.prompt_variants = [p for p in (prompt_variants or []) if p.strip()]
    
    def _load_cache(self) -> dict:
        """Load cache from disk if available."""
        if not self.use_cache:
            return {}
        if self.cache_path and self.cache_path.exists():
            try:
                with open(self.cache_path) as f:
                    cache = json.load(f)
                if self.logger:
                    self.logger.info("Loaded %d cached LLM responses from %s", len(cache), self.cache_path)
                return cache
            except Exception as e:
                if self.logger:
                    self.logger.warning("Failed to load cache: %s", e)
        return {}
    
    def save_cache(self):
        """Save cache to disk."""
        if self.use_cache and self.cache_path:
            with self._cache_lock:
                cache_copy = dict(self.cache)
                cache_hits = self._cache_hits
                cache_misses = self._cache_misses
            ensure_dir(self.cache_path.parent)
            with open(self.cache_path, 'w') as f:
                json.dump(cache_copy, f)
            if self.logger:
                self.logger.info("Saved %d cached responses (hits: %d, misses: %d)", 
                               len(cache_copy), cache_hits, cache_misses)
    
    def generate_query(
        self, 
        question: str, 
        context: list[dict],
        prompt_template: str | None = None,
        max_context_chars: int = 3000
    ) -> str:
        """
        Generate a search query based on question and current context.
        
        This is the same function for ALL hops (matching LeReT):
        - Hop 1: context=[] (empty)
        - Hop 2+: context=accumulated passages from previous hops
        """
        # Format context (empty string for hop 1)
        context_str = self._format_context(context, max_context_chars)
        prompt_template = self._select_prompt_template(prompt_template)
        
        # Check cache
        cache_key = None
        if self.use_cache:
            cache_key = hashlib.md5(f"{self._cache_model_id}||{question}||{context_str}||{prompt_template}".encode()).hexdigest()
            with self._cache_lock:
                if cache_key in self.cache:
                    self._cache_hits += 1
                    return self.cache[cache_key]
                self._cache_misses += 1
        
        prompt = self._render_prompt(prompt_template, question, context_str)

        for attempt in range(1, self.max_retries + 1):
            try:
                if self.inference_url:
                    # Local inference server (on-policy generation)
                    # Use chat_batch endpoint (correctly extracts generated tokens)
                    response = httpx.post(
                        f"{self.inference_url}/v1/chat_batch",
                        json={
                            "requests": [{
                                "messages": [{"role": "user", "content": prompt}],
                                "max_new_tokens": 128,
                                "temperature": self.temperature,
                            }],
                        },
                        timeout=self.request_timeout_s,
                    )
                    response.raise_for_status()
                    raw = response.json()["responses"][0].strip()
                    query = parse_dspy_output(raw)
                else:
                    # OpenRouter (off-policy generation)
                    response = httpx.post(
                        self.base_url,
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": self.model,
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": self.max_output_tokens,
                            "temperature": self.temperature,
                        },
                        timeout=self.request_timeout_s,
                    )
                    response.raise_for_status()
                    result = response.json()
                    raw = result["choices"][0]["message"]["content"].strip()
                    query = parse_dspy_output(raw)

                # Cache result
                if self.use_cache and cache_key is not None:
                    with self._cache_lock:
                        self.cache[cache_key] = query
                return query
            except Exception as e:
                retryable = _is_retryable_http_error(e) or _is_retryable_query_parse_error(e)
                is_last_attempt = attempt >= self.max_retries
                if not retryable or is_last_attempt:
                    if _is_retryable_query_parse_error(e):
                        fallback_query = _fallback_query_from_question(question)
                        if self.logger:
                            self.logger.warning(
                                "Query generation fell back to question text after parse failure: %s",
                                e,
                            )
                        if self.use_cache and cache_key is not None:
                            with self._cache_lock:
                                self.cache[cache_key] = fallback_query
                        return fallback_query
                    if self.logger:
                        self.logger.exception("Query generation failed (fail-fast): %s", e)
                    raise RuntimeError("Query generation failed") from e
                sleep_s = self.retry_backoff_s * (2 ** (attempt - 1))
                if self.logger:
                    self.logger.warning(
                        "Transient query-generation error on attempt %d/%d: %s; retrying in %.1fs",
                        attempt,
                        self.max_retries,
                        e,
                        sleep_s,
                    )
                if sleep_s > 0:
                    time.sleep(sleep_s)

        raise RuntimeError("Query generation failed")

    def _format_context(self, context: list[dict], max_chars: int) -> str:
        """Format retrieved passages as context string (LeReT-style)."""
        if not context:
            return "N/A"  # LeReT uses "N/A" for empty context
        
        lines = []
        total_chars = 0
        for i, passage in enumerate(context):
            title = passage.get("title", "")
            text = passage.get("text", "")
            line = f"[{i+1}] «{title}: {text}»"  # LeReT uses « » delimiters
            
            if total_chars + len(line) > max_chars:
                break
            lines.append(line)
            total_chars += len(line)
        
        return "\n".join(lines) if lines else "N/A"

    def _select_prompt_template(self, prompt_template: str | None) -> str:
        if prompt_template:
            return prompt_template
        if self.prompt_variants:
            return random.choice(self.prompt_variants)
        return QUERY_GENERATION_PROMPT

    def _render_prompt(self, template: str, question: str, context_str: str) -> str:
        rendered = safe_format_prompt(template, question=question, context=context_str)
        if rendered.strip():
            return rendered
        return QUERY_GENERATION_PROMPT.format(question=question, context=context_str)


def _is_retryable_http_error(exc: Exception) -> bool:
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code if exc.response is not None else None
        return code in {408, 409, 425, 429, 500, 502, 503, 504}
    return False


def _is_retryable_query_parse_error(exc: Exception) -> bool:
    return isinstance(exc, ValueError)


# ============================================
# ColBERT Retriever
# ============================================

class ColBERTRetriever:
    """ColBERTv2 retriever using pre-built Wikipedia index."""
    
    def __init__(self, index_path: Path, collection_path: Path, logger):
        self.logger = logger
        self.index_path = index_path
        self.collection_path = collection_path
        self.collection_tsv = self.collection_path / "collection.tsv"
        self.collection_file = None
        self.offsets = array("Q")  # 1-based pid -> byte offset
        self._load_collection_offsets()
        
        # Resolve symlinks to get absolute path (ColBERT needs this)
        resolved_index_path = index_path.resolve()
        logger.info("Loading ColBERTv2 index from %s...", resolved_index_path)
        
        try:
            from colbert import Searcher
            from colbert.infra import ColBERTConfig
            
            # Initialize ColBERT searcher with resolved absolute path
            self.searcher = Searcher(
                index=str(resolved_index_path),
                config=ColBERTConfig(
                    doc_maxlen=256,
                    query_maxlen=32,
                )
            )
            logger.info("ColBERTv2 searcher loaded successfully")
            
        except Exception as e:
            logger.error("Failed to load ColBERTv2: %s", e)
            raise
    
    def _load_collection_offsets(self) -> None:
        """Load byte offsets for collection.tsv to avoid full in-memory passages."""
        self.logger.info("Indexing Wikipedia collection offsets from %s...", self.collection_tsv)
        self.collection_file = open(self.collection_tsv, "r", encoding="utf-8")
        _ = self.collection_file.readline()
        offset = self.collection_file.tell()
        line = self.collection_file.readline()
        while line:
            self.offsets.append(offset)
            offset = self.collection_file.tell()
            line = self.collection_file.readline()
        self.collection_file.seek(0)
        self.logger.info("Indexed %d passage offsets", len(self.offsets))

    def _get_entry(self, pid: int) -> tuple[str, str] | None:
        if not self.collection_file or pid <= 0 or pid > len(self.offsets):
            return None
        self.collection_file.seek(self.offsets[pid - 1])
        line = self.collection_file.readline()
        if not line:
            return None
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3:
            return None
        try:
            line_pid = int(parts[0])
        except ValueError:
            return None
        if line_pid != pid:
            return None
        text = parts[1]
        title = parts[2]
        return text, title

    def retrieve(self, query: str, top_k: int = 10, exclude_pids: set = None) -> list[dict]:
        """Retrieve top-k passages for a query, optionally excluding already-seen passages."""
        try:
            # Retrieve more if we need to exclude some
            fetch_k = top_k + len(exclude_pids) if exclude_pids else top_k
            fetch_k = min(fetch_k, 100)  # Cap at 100
            
            results = self.searcher.search(query, k=fetch_k)
            
            passages = []
            for pid, rank, score in zip(*results):
                # Skip if already retrieved
                if exclude_pids and pid in exclude_pids:
                    continue
                
                entry = self._get_entry(int(pid))
                if not entry:
                    continue
                text, title = entry
                
                passages.append({
                    "doc_id": int(pid),
                    "title": title.strip(),
                    "text": text.strip(),
                    "score": float(score),
                })
                
                if len(passages) >= top_k:
                    break
            
            return passages
            
        except Exception as e:
            self.logger.exception("Local retrieval failed for query '%s': %s", query[:50], e)
            raise RuntimeError("Local retrieval failed") from e


class RemoteColBERTRetriever:
    """HTTP client for ColBERT search server (serve_colbert_min.py)."""

    def __init__(
        self,
        base_url: str,
        logger,
        timeout: float = 30.0,
        max_retries: int = 4,
        retry_backoff_s: float = 1.0,
    ):
        self.logger = logger
        self.base_url = str(base_url).rstrip("/")
        self.timeout = float(timeout)
        self.max_retries = max(1, int(max_retries))
        self.retry_backoff_s = max(0.0, float(retry_backoff_s))
        self.client = httpx.Client(
            timeout=self.timeout,
            limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
        )

    def healthcheck(self) -> None:
        resp = self.client.get(f"{self.base_url}/health")
        resp.raise_for_status()
        payload = resp.json()
        if not isinstance(payload, dict) or not payload.get("ok", False):
            raise RuntimeError(f"Unexpected /health response from {self.base_url}: {payload}")

    def retrieve(self, query: str, top_k: int = 10, exclude_pids: set = None) -> list[dict]:
        fetch_k = top_k + len(exclude_pids) if exclude_pids else top_k
        fetch_k = min(fetch_k, 100)

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.get(
                    f"{self.base_url}/api/search",
                    params={"query": query, "k": fetch_k},
                )
                resp.raise_for_status()
                payload = resp.json()
                if not isinstance(payload, dict) or "topk" not in payload:
                    raise ValueError(f"Unexpected /api/search payload: {payload!r}")
                topk_rows = payload.get("topk", [])
                if not isinstance(topk_rows, list):
                    raise ValueError(f"Unexpected topk payload type: {type(topk_rows)!r}")

                passages = []
                for row in topk_rows:
                    if not isinstance(row, dict):
                        continue
                    pid_raw = row.get("pid", row.get("doc_id"))
                    try:
                        pid = int(pid_raw)
                    except (TypeError, ValueError):
                        continue
                    if exclude_pids and pid in exclude_pids:
                        continue
                    passages.append(
                        {
                            "doc_id": pid,
                            "title": str(row.get("title", "")).strip(),
                            "text": str(row.get("text", "")).strip(),
                            "score": float(row.get("score", 0.0)),
                        }
                    )
                    if len(passages) >= top_k:
                        break
                return passages
            except Exception as e:
                retryable = _is_retryable_http_error(e)
                is_last_attempt = attempt >= self.max_retries
                if not retryable or is_last_attempt:
                    self.logger.exception("Remote retrieval failed for query '%s': %s", query[:50], e)
                    raise RuntimeError("Remote retrieval failed") from e
                sleep_s = self.retry_backoff_s * (2 ** (attempt - 1))
                self.logger.warning(
                    "Transient remote retrieval error on attempt %d/%d for query '%s': %s; retrying in %.1fs",
                    attempt,
                    self.max_retries,
                    query[:50],
                    e,
                    sleep_s,
                )
                if sleep_s > 0:
                    time.sleep(sleep_s)

        raise RuntimeError("Remote retrieval failed")

    def close(self) -> None:
        self.client.close()


def _init_retriever_with_retry(
    index_path: Path,
    collection_path: Path,
    logger,
    max_attempts: int = 3,
    backoff_seconds: int = 20,
) -> ColBERTRetriever:
    for attempt in range(1, max_attempts + 1):
        try:
            return ColBERTRetriever(index_path, collection_path, logger)
        except httpx.ReadTimeout:
            if attempt >= max_attempts:
                logger.exception("ColBERT init failed after %d attempts", attempt)
                raise
            logger.warning(
                "ColBERT init read timeout (attempt %d/%d). Retrying in %ds.",
                attempt,
                max_attempts,
                backoff_seconds,
            )
            time.sleep(backoff_seconds)


def _init_remote_retriever(
    remote_url: str,
    logger,
    max_attempts: int = 3,
    backoff_seconds: int = 3,
    timeout_s: float = 90.0,
    max_retries: int = 6,
    retry_backoff_s: float = 2.0,
) -> RemoteColBERTRetriever:
    retriever = RemoteColBERTRetriever(
        remote_url,
        logger,
        timeout=timeout_s,
        max_retries=max_retries,
        retry_backoff_s=retry_backoff_s,
    )
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            retriever.healthcheck()
            logger.info("Connected to remote ColBERT retriever at %s", retriever.base_url)
            return retriever
        except Exception as exc:
            last_error = exc
            if attempt >= max_attempts:
                logger.exception("Remote ColBERT healthcheck failed after %d attempts", attempt)
                raise
            logger.warning(
                "Remote ColBERT healthcheck failed (attempt %d/%d): %s. Retrying in %ds.",
                attempt,
                max_attempts,
                exc,
                backoff_seconds,
            )
            time.sleep(backoff_seconds)
    assert last_error is not None
    raise last_error


def _build_retriever(
    *,
    raw_dir: Path,
    logger,
    remote_url: str | None,
    remote_timeout_s: float = 90.0,
    remote_max_retries: int = 6,
    remote_retry_backoff_s: float = 2.0,
):
    if remote_url:
        logger.info("Using REMOTE ColBERT retriever: %s", remote_url)
        return _init_remote_retriever(
            remote_url,
            logger,
            timeout_s=remote_timeout_s,
            max_retries=remote_max_retries,
            retry_backoff_s=remote_retry_backoff_s,
        )

    # Local ColBERT fallback path (legacy/local workflows)
    index_path = raw_dir / "colbert_index" / "wiki2017"
    collection_path = raw_dir / "colbert_index" / "collection"

    if not index_path.exists():
        raise FileNotFoundError(f"ColBERTv2 index not found at {index_path}")
    if not collection_path.exists():
        raise FileNotFoundError(f"Wikipedia collection not found at {collection_path}")

    os.environ.setdefault("HF_HUB_READ_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_CONNECT_TIMEOUT", "30")
    return _init_retriever_with_retry(index_path, collection_path, logger)


# ============================================
# Multi-Hop Rollout Generation
# ============================================

def _rollout_dataset_for_helpers(dataset_name: str) -> str:
    normalized = (dataset_name or "").strip().lower()
    if normalized in {"hotpot", "hotpot_fullwiki"}:
        return "hotpot_fullwiki"
    return normalized


def compute_gold_coverage(retrieved: list[dict], gold_titles: list[str]) -> dict:
    """Compute how many gold passages were retrieved."""
    if not gold_titles:
        return {"coverage": 1.0, "hit": 0, "total": 0}
    
    gold_titles_lower = {t.lower().strip() for t in gold_titles}
    retrieved_titles_lower = {r["title"].lower().strip() for r in retrieved if r.get("title")}
    
    hits = gold_titles_lower & retrieved_titles_lower
    
    return {
        "coverage": len(hits) / len(gold_titles_lower) if gold_titles_lower else 1.0,
        "hit": len(hits),
        "total": len(gold_titles_lower),
    }


def generate_multihop_rollout(
    question: dict,
    retriever: ColBERTRetriever,
    query_generator: QueryGenerator,
    dataset_name: str,
    top_k_per_hop: int = 5,
    use_rewriter: bool = True,
    logger=None,
) -> dict:
    """
    Generate a multi-hop retrieval rollout for a single question.
    
    LeReT-style: Same query generator for ALL hops.
    - Hop 1: generate_query(question, context=[])
    - Hop 2+: generate_query(question, context=accumulated_passages)
    
    Args:
        use_rewriter: If True, use query generator for all hops (LeReT-style).
                     If False, use original question for hop 1 only (baseline).
    """
    qid = question.get("_id", question.get("qid", question.get("id", "")))
    base_qid = question.get("base_qid", qid)
    question_text = question.get("question", "")
    answer = question.get("answer", "")
    helper_dataset = _rollout_dataset_for_helpers(dataset_name)
    gold_titles = shared_extract_gold_titles(question, dataset=helper_dataset)
    prompt_variant_id = question.get("prompt_variant_id")
    prompt_variant = question.get("prompt_variant")
    prompt_variant_source = question.get("prompt_variant_source")
    if prompt_variant_id is not None and qid == base_qid:
        qid = f"{base_qid}::v{prompt_variant_id}"
    
    num_hops = shared_get_num_hops(base_qid, helper_dataset, question)
    
    all_retrieved = []  # All passages across all hops
    seen_pids = set()   # For deduplication
    hop_details = []    # Details per hop
    
    for hop in range(1, num_hops + 1):
        # Generate query for this hop (same function for all hops, LeReT-style)
        if use_rewriter:
            # LeReT-style: query generator for ALL hops
            # Hop 1: context is empty [], Hop 2+: context is accumulated passages
            current_query = query_generator.generate_query(
                question_text,
                all_retrieved,  # Empty for hop 1, accumulated for hop 2+
                prompt_template=prompt_variant,
            )
        else:
            # Baseline: original question for hop 1, then query generator for 2+
            if hop == 1:
                current_query = question_text
            else:
                current_query = query_generator.generate_query(
                    question_text,
                    all_retrieved,
                    prompt_template=prompt_variant,
                )
        
        # Retrieve for this hop
        hop_passages = retriever.retrieve(
            current_query, 
            top_k=top_k_per_hop,
            exclude_pids=seen_pids
        )
        
        # Add to cumulative results
        for p in hop_passages:
            if p["doc_id"] not in seen_pids:
                seen_pids.add(p["doc_id"])
                p["hop"] = hop  # Tag which hop retrieved this
                all_retrieved.append(p)
        
        # Compute coverage after this hop
        hop_coverage = compute_gold_coverage(all_retrieved, gold_titles)
        
        hop_details.append({
            "hop": hop,
            "query": current_query,
            "num_retrieved": len(hop_passages),
            "cumulative_retrieved": len(all_retrieved),
            "coverage_after_hop": hop_coverage["coverage"],
        })
    
    # Mark which passages are gold
    gold_titles_lower = {t.lower().strip() for t in gold_titles}
    for p in all_retrieved:
        p["is_gold"] = p.get("title", "").lower().strip() in gold_titles_lower
    
    # Final coverage
    final_coverage = compute_gold_coverage(all_retrieved, gold_titles)
    
    return {
        "dataset": dataset_name,
        "qid": qid,
        "base_qid": base_qid,
        "question": question_text,
        "answer": answer,
        "gold_titles": gold_titles,
        "num_hops": num_hops,
        "prompt_variant_id": prompt_variant_id,
        "prompt_variant_source": prompt_variant_source,
        "retrieved": all_retrieved,
        "hop_details": hop_details,
        "gold_coverage": final_coverage["coverage"],
        "gold_hit": final_coverage["hit"],
        "gold_total": final_coverage["total"],
    }


def _load_prompt_variants(path: str | None) -> list[str]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Prompt variants file not found: {p}")
    data = json.loads(p.read_text())
    if isinstance(data, dict):
        variants = list(data.values())
    elif isinstance(data, list):
        variants = data
    else:
        raise ValueError("Prompt variants must be a JSON list or dict.")
    return [v for v in variants if isinstance(v, str) and v.strip()]


def _expand_questions_with_variants(
    questions: list[dict],
    prompt_variants: list[str],
    variants_per_question: int,
    mode: str,
    seed: int,
    source_label: str,
) -> list[dict]:
    if not prompt_variants:
        return questions
    rng = random.Random(seed)
    expanded = []
    for q in questions:
        base_qid = q.get("qid") or q.get("_id") or q.get("id")
        if mode == "all":
            chosen = list(range(len(prompt_variants)))
        else:
            k = min(max(1, variants_per_question), len(prompt_variants))
            chosen = rng.sample(range(len(prompt_variants)), k=k)
        for idx in chosen:
            q_copy = dict(q)
            q_copy["base_qid"] = base_qid
            q_copy["prompt_variant_id"] = idx
            q_copy["prompt_variant"] = prompt_variants[idx]
            q_copy["prompt_variant_source"] = source_label
            q_copy["qid"] = f"{base_qid}::v{idx}"
            expanded.append(q_copy)
    return expanded


def _parse_hop_mix(spec: str | None) -> dict[int, float]:
    """Parse hop mix spec like '2=69.91,3=23.02,4=7.07'."""
    if not spec:
        return {}

    out: dict[int, float] = {}
    for part in spec.split(","):
        token = part.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"Invalid --hop-mix token (expected hop=value): {token!r}")
        hop_s, value_s = token.split("=", 1)
        try:
            hop = int(hop_s.strip())
        except ValueError as exc:
            raise ValueError(f"Invalid hop in --hop-mix: {hop_s!r}") from exc
        if hop not in {2, 3, 4}:
            raise ValueError(f"Unsupported hop in --hop-mix: {hop} (allowed: 2,3,4)")
        try:
            value = float(value_s.strip())
        except ValueError as exc:
            raise ValueError(f"Invalid value in --hop-mix token: {token!r}") from exc
        if value < 0:
            raise ValueError(f"Negative value in --hop-mix token: {token!r}")
        out[hop] = value

    if not out:
        raise ValueError("--hop-mix was provided but no valid tokens were parsed")
    if sum(out.values()) <= 0:
        raise ValueError("--hop-mix values must sum to > 0")
    return out


def _allocate_by_weights(total: int, weights: dict[int, float]) -> dict[int, int]:
    if total <= 0:
        raise ValueError(f"total must be > 0, got {total}")
    weight_sum = sum(weights.values())
    if weight_sum <= 0:
        raise ValueError("weights must sum to > 0")

    raw = {hop: (total * (weights[hop] / weight_sum)) for hop in weights}
    base = {hop: int(raw[hop]) for hop in raw}
    remainder = total - sum(base.values())

    if remainder > 0:
        rank = sorted(
            ((raw[hop] - base[hop], hop) for hop in raw),
            key=lambda x: (-x[0], x[1]),
        )
        for i in range(remainder):
            base[rank[i % len(rank)][1]] += 1
    return base


def _infer_question_hops(question: dict, dataset_name: str) -> int:
    helper_dataset = _rollout_dataset_for_helpers(dataset_name)
    qid = (
        question.get("base_qid")
        or question.get("qid")
        or question.get("_id")
        or question.get("id")
        or ""
    )
    return shared_get_num_hops(str(qid), helper_dataset, question)


def _sample_questions_by_hop_mix(
    questions: list[dict],
    *,
    dataset_name: str,
    hop_mix: dict[int, float],
    total_questions: int,
    seed: int,
    logger,
) -> list[dict]:
    if total_questions > len(questions):
        raise ValueError(
            f"--hop-mix-total ({total_questions}) exceeds available questions ({len(questions)})"
        )

    pool: dict[int, list[tuple[int, dict]]] = {2: [], 3: [], 4: []}
    for idx, q in enumerate(questions):
        hop = _infer_question_hops(q, dataset_name)
        if hop in pool:
            pool[hop].append((idx, q))

    targets = _allocate_by_weights(total_questions, hop_mix)
    for hop, target in sorted(targets.items()):
        available = len(pool.get(hop, []))
        if available < target:
            raise ValueError(
                f"Insufficient {hop}-hop questions for --hop-mix: need {target}, available {available}"
            )

    rng = random.Random(seed)
    selected_indices: list[int] = []
    for hop in sorted(targets):
        target = targets[hop]
        if target <= 0:
            continue
        chosen = rng.sample(pool[hop], k=target)
        selected_indices.extend(i for i, _ in chosen)

    selected_indices.sort()
    selected = [questions[i] for i in selected_indices]

    counts = {hop: targets.get(hop, 0) for hop in sorted(targets)}
    logger.info(
        "Applied hop-mix sampling: total=%d, selected=%s, seed=%d",
        total_questions,
        counts,
        seed,
    )
    return selected


def generate_rollouts(
    questions: list[dict],
    retriever: ColBERTRetriever,
    query_generator: QueryGenerator,
    dataset_name: str,
    top_k_per_hop: int,
    use_rewriter: bool,
    logger,
    num_threads: int = 1,
    checkpoint_path: Path | None = None,
    checkpoint_interval: int = 100,
) -> list[dict]:
    """Generate multi-hop retrieval rollouts for all questions.
    
    Supports checkpointing for resumable runs.
    """
    
    if num_threads > 1:
        return generate_rollouts_parallel(
            questions, retriever, query_generator, dataset_name,
            top_k_per_hop, use_rewriter, logger, num_threads,
            checkpoint_path, checkpoint_interval
        )
    
    # Sequential processing (fallback)
    rollouts = []
    total_coverage = 0.0
    
    pbar = tqdm(
        questions, 
        desc=f"{dataset_name} (sequential)",
        unit="q",
        ncols=100,
    )
    
    for q in pbar:
        rollout = generate_multihop_rollout(
            q, retriever, query_generator, dataset_name, top_k_per_hop, 
            use_rewriter=use_rewriter, logger=None
        )
        rollouts.append(rollout)
        total_coverage += rollout["gold_coverage"]
        
        # Update progress bar
        avg_cov = total_coverage / len(rollouts)
        pbar.set_postfix(cov=f"{avg_cov*100:.1f}%")
    
    pbar.close()
    avg_coverage = total_coverage / len(questions) if questions else 0
    logger.info("Final average gold coverage: %.1f%%", avg_coverage * 100)
    
    return rollouts


def generate_rollouts_parallel(
    questions: list[dict],
    retriever: ColBERTRetriever,
    query_generator: QueryGenerator,
    dataset_name: str,
    top_k_per_hop: int,
    use_rewriter: bool,
    logger,
    num_threads: int = 16,
    checkpoint_path: Path | None = None,
    checkpoint_interval: int = 100,
) -> list[dict]:
    """Generate multi-hop retrieval rollouts in parallel (LeReT-style).
    
    Supports checkpointing for resumable runs.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    logger.info("Using %d threads for parallel processing", num_threads)
    
    desired_qids = set()
    for q in questions:
        qid = q.get("qid") or q.get("_id")
        base_qid = q.get("base_qid") or qid
        prompt_variant_id = q.get("prompt_variant_id")
        if prompt_variant_id is not None and qid == base_qid:
            qid = f"{base_qid}::v{prompt_variant_id}"
        desired_qids.add(qid)

    # Load checkpoint if available
    completed_ids = set()
    rollouts_dict = {}  # qid -> rollout for deduplication
    
    if checkpoint_path and checkpoint_path.exists():
        try:
            with open(checkpoint_path) as f:
                for line in f:
                    r = json.loads(line)
                    qid = r.get("qid") or r.get("_id")
                    base_qid = r.get("base_qid") or qid
                    prompt_variant_id = r.get("prompt_variant_id")
                    if prompt_variant_id is not None and qid == base_qid:
                        qid = f"{base_qid}::v{prompt_variant_id}"
                        r["qid"] = qid
                    if qid not in desired_qids:
                        continue
                    rollouts_dict[qid] = r
                    completed_ids.add(qid)
            logger.info("Resuming from checkpoint: %d questions already completed", len(completed_ids))
        except Exception as e:
            logger.warning("Failed to load checkpoint: %s", e)
    
    # Filter to only uncompleted questions
    remaining_questions = []
    for q in questions:
        qid = q.get("qid") or q.get("_id")
        base_qid = q.get("base_qid") or qid
        prompt_variant_id = q.get("prompt_variant_id")
        if prompt_variant_id is not None and qid == base_qid:
            qid = f"{base_qid}::v{prompt_variant_id}"
        if qid not in completed_ids:
            remaining_questions.append(q)
    
    if not remaining_questions:
        logger.info("All questions already completed!")
        return list(rollouts_dict.values())
    
    logger.info("Processing %d remaining questions (%d already done)", 
               len(remaining_questions), len(completed_ids))
    
    start_time = time.time()
    newly_completed = 0
    total_coverage = sum(r["gold_coverage"] for r in rollouts_dict.values())
    
    def process_question(q):
        qid = q.get("qid") or q.get("_id")
        try:
            rollout = generate_multihop_rollout(
                q, retriever, query_generator, dataset_name, top_k_per_hop,
                use_rewriter=use_rewriter, logger=None  # Don't log per-question
            )
        except Exception as exc:
            return qid, None, exc
        return qid, rollout, None
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(process_question, q): q for q in remaining_questions}
        
        # Progress bar with live stats
        pbar = tqdm(
            as_completed(futures), 
            total=len(remaining_questions),
            desc=f"{dataset_name}",
            unit="q",
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] cov:{postfix}",
            initial=0,
        )
        
        for future in pbar:
            try:
                qid, rollout, error = future.result()
                if error:
                    logger.exception("Rollout failed for qid=%s", qid, exc_info=error)
                    if checkpoint_path:
                        _save_checkpoint(checkpoint_path, list(rollouts_dict.values()), logger)
                        query_generator.save_cache()
                    raise RuntimeError(f"Rollout failed for qid={qid}") from error
                rollouts_dict[qid] = rollout
                total_coverage += rollout["gold_coverage"]
                newly_completed += 1
                
                # Update progress bar with live coverage
                total_done = len(completed_ids) + newly_completed
                avg_cov = total_coverage / total_done if total_done > 0 else 0
                pbar.set_postfix_str(f"{avg_cov*100:.1f}%")
                
                # Checkpoint every N completions
                if checkpoint_path and newly_completed % checkpoint_interval == 0:
                    _save_checkpoint(checkpoint_path, list(rollouts_dict.values()), logger)
                    # Also save LLM cache
                    query_generator.save_cache()
                
                # Log every 500 for file-based monitoring
                if newly_completed % 500 == 0:
                    elapsed = time.time() - start_time
                    qps = newly_completed / elapsed if elapsed > 0 else 0
                    logger.info(
                        "Progress: %d/%d (%.1f q/s), avg coverage: %.1f%%",
                        total_done, len(questions), qps, avg_cov * 100
                    )
                
            except Exception as e:
                logger.error("Error processing question: %s", e)
                if checkpoint_path:
                    _save_checkpoint(checkpoint_path, list(rollouts_dict.values()), logger)
                    query_generator.save_cache()
                raise
        
        pbar.close()
    
    # Final checkpoint
    if checkpoint_path:
        _save_checkpoint(checkpoint_path, list(rollouts_dict.values()), logger)
        query_generator.save_cache()
    
    rollouts = list(rollouts_dict.values())
    avg_coverage = total_coverage / len(rollouts) if rollouts else 0
    elapsed = time.time() - start_time
    logger.info(
        "Final: %d rollouts, avg coverage: %.1f%%, %.2f q/s (this run)",
        len(rollouts),
        avg_coverage * 100,
        newly_completed / elapsed if elapsed > 0 else 0,
    )
    return rollouts


def _save_checkpoint(checkpoint_path: Path, rollouts: list[dict], logger):
    """Save checkpoint to disk."""
    ensure_dir(checkpoint_path.parent)
    with open(checkpoint_path, 'w') as f:
        for r in rollouts:
            f.write(json.dumps(r) + '\n')
    logger.info("Checkpoint saved: %d rollouts to %s", len(rollouts), checkpoint_path)

    return rollouts


# ============================================
# Batched Inference + Parallel ColBERT
# ============================================

def _get_qid(question: dict) -> tuple[str, str]:
    """Extract (qid, base_qid) from a question dict, including variant suffix."""
    qid = question.get("_id", question.get("qid", question.get("id", "")))
    base_qid = question.get("base_qid", qid)
    prompt_variant_id = question.get("prompt_variant_id")
    if prompt_variant_id is not None and qid == base_qid:
        qid = f"{base_qid}::v{prompt_variant_id}"
    return qid, base_qid


def _batch_chat_inference(
    inference_url: str,
    prompts: list[str],
    temperature: float,
    max_new_tokens: int = 128,
    timeout: float = 45.0,
    max_retries: int = 3,
    wait_retries: int = 1,
    wait_seconds: float = 30.0,
    logger=None,
) -> list[str]:
    """Send a batch of prompts to the inference server via /v1/chat_batch.

    On 500 errors (typically OOM), splits the batch in half and retries.
    On connection/timeout errors, waits and retries up to wait_retries times.
    Never falls back to original query text — raises if all retries exhausted.
    """
    if not prompts:
        return []

    requests_list = [
        {
            "messages": [{"role": "user", "content": prompt}],
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        }
        for prompt in prompts
    ]

    try:
        response = httpx.post(
            f"{inference_url}/v1/chat_batch",
            json={"requests": requests_list},
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()["responses"]
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 500 and len(prompts) > 1 and max_retries > 0:
            if logger:
                logger.warning("OOM with batch size %d, splitting and retrying...", len(prompts))
            mid = len(prompts) // 2
            left = _batch_chat_inference(
                inference_url, prompts[:mid], temperature,
                max_new_tokens, timeout, max_retries - 1, wait_retries, wait_seconds, logger,
            )
            right = _batch_chat_inference(
                inference_url, prompts[mid:], temperature,
                max_new_tokens, timeout, max_retries - 1, wait_retries, wait_seconds, logger,
            )
            return left + right
        raise
    except httpx.ConnectTimeout:
        # Server temporarily unreachable — wait and retry with same batch (splitting won't help)
        if wait_retries > 0:
            if logger:
                logger.warning(
                    "ConnectTimeout (batch=%d), waiting %.0fs and retrying (%d retries left)...",
                    len(prompts), wait_seconds, wait_retries,
                )
            time.sleep(wait_seconds)
            return _batch_chat_inference(
                inference_url, prompts, temperature,
                max_new_tokens, timeout, max_retries, wait_retries - 1, wait_seconds, logger,
            )
        raise
    except httpx.TimeoutException:
        # Read/write timeout — try splitting first, then fall back to wait-and-retry
        if len(prompts) > 1 and max_retries > 0:
            if logger:
                logger.warning("Timeout with batch size %d, splitting and retrying...", len(prompts))
            mid = len(prompts) // 2
            left = _batch_chat_inference(
                inference_url, prompts[:mid], temperature,
                max_new_tokens, timeout, max_retries - 1, wait_retries, wait_seconds, logger,
            )
            right = _batch_chat_inference(
                inference_url, prompts[mid:], temperature,
                max_new_tokens, timeout, max_retries - 1, wait_retries, wait_seconds, logger,
            )
            return left + right
        if wait_retries > 0:
            if logger:
                logger.warning(
                    "Timeout on batch size 1, waiting %.0fs and retrying (%d retries left)...",
                    wait_seconds, wait_retries,
                )
            time.sleep(wait_seconds)
            return _batch_chat_inference(
                inference_url, prompts, temperature,
                max_new_tokens, timeout, max_retries, wait_retries - 1, wait_seconds, logger,
            )
        if logger:
            logger.warning(
                "Timeout on batch size 1 after retries exhausted; returning empty response "
                "so the caller can fall back to question text."
            )
        return ["" for _ in prompts]


def _retrieve_batch(
    retriever: ColBERTRetriever,
    queries: list[str],
    exclude_pids_list: list[set],
    top_k: int,
) -> list[list[dict]]:
    """Run ColBERT retrieval for multiple queries sequentially.

    ColBERTRetriever uses a shared file handle (collection_file.seek()) which
    is NOT thread-safe — parallel retrieval causes seek races that return
    fewer than k results. Always run sequentially.
    """
    results = []
    for i in range(len(queries)):
        passages = retriever.retrieve(
            queries[i], top_k=top_k, exclude_pids=exclude_pids_list[i]
        )
        results.append(passages)
    return results


def generate_rollouts_batched(
    questions: list[dict],
    retriever: ColBERTRetriever,
    query_generator: QueryGenerator,
    dataset_name: str,
    top_k_per_hop: int,
    use_rewriter: bool,
    logger,
    batch_size: int = 50,
    checkpoint_path: Path | None = None,
) -> list[dict]:
    """Generate rollouts with GPU-batched inference and parallel ColBERT retrieval.

    Instead of processing questions one at a time with threading, processes
    chunks of questions through the full multi-hop pipeline:
      Hop 1: Batch GPU inference → Parallel ColBERT retrieval
      Hop 2: Batch GPU inference (with hop 1 context) → Parallel ColBERT retrieval
      → Checkpoint

    This is significantly faster when using a local inference server because
    the GPU processes all queries in a single forward pass.
    """
    # Build desired QID set and load checkpoint
    rollouts_dict = {}
    completed_qids = set()

    desired_qids = set()
    for q in questions:
        qid, _ = _get_qid(q)
        desired_qids.add(qid)

    if checkpoint_path and checkpoint_path.exists():
        try:
            with open(checkpoint_path) as f:
                for line in f:
                    r = json.loads(line)
                    rqid = r.get("qid", "")
                    if rqid in desired_qids:
                        rollouts_dict[rqid] = r
                        completed_qids.add(rqid)
            logger.info("Resuming from checkpoint: %d questions already completed", len(completed_qids))
        except Exception as e:
            logger.warning("Failed to load checkpoint: %s", e)

    # Filter to remaining questions
    remaining = [q for q in questions if _get_qid(q)[0] not in completed_qids]

    if not remaining:
        logger.info("All questions already completed!")
        return list(rollouts_dict.values())

    logger.info(
        "Processing %d remaining questions (%d done) in chunks of %d",
        len(remaining), len(completed_qids), batch_size,
    )

    start_time = time.time()
    num_chunks = (len(remaining) + batch_size - 1) // batch_size
    newly_completed = 0
    total_coverage = sum(r["gold_coverage"] for r in rollouts_dict.values())

    for chunk_idx in range(num_chunks):
        chunk = remaining[chunk_idx * batch_size : (chunk_idx + 1) * batch_size]
        chunk_qids = [_get_qid(q)[0] for q in chunk]

        # Per-question state
        all_retrieved = {qid: [] for qid in chunk_qids}
        seen_pids = {qid: set() for qid in chunk_qids}
        hop_details_map = {qid: [] for qid in chunk_qids}

        helper_dataset = _rollout_dataset_for_helpers(dataset_name)
        num_hops = shared_get_num_hops(
            chunk[0].get("base_qid", chunk[0].get("_id", "")),
            helper_dataset,
            chunk[0],
        )

        for hop in range(1, num_hops + 1):
            # --- Render prompts and check cache ---
            prompts_to_infer = []  # (chunk_index, prompt)
            query_results = {}     # chunk_index -> query string
            cache_keys = {}        # chunk_index -> cache_key

            for i, q in enumerate(chunk):
                qid = chunk_qids[i]
                question_text = q.get("question", "")

                if not use_rewriter and hop == 1:
                    query_results[i] = question_text
                    continue

                context = all_retrieved[qid]
                context_str = query_generator._format_context(context, 3000)
                template = query_generator._select_prompt_template(q.get("prompt_variant"))
                prompt = query_generator._render_prompt(template, question_text, context_str)

                # Check cache
                if query_generator.use_cache:
                    cache_key = hashlib.md5(
                        f"{query_generator._cache_model_id}||{question_text}||{context_str}||{template}".encode()
                    ).hexdigest()
                    cache_keys[i] = cache_key
                    with query_generator._cache_lock:
                        if cache_key in query_generator.cache:
                            query_results[i] = query_generator.cache[cache_key]
                            query_generator._cache_hits += 1
                            continue
                        query_generator._cache_misses += 1

                prompts_to_infer.append((i, prompt))

            # --- Batch inference for uncached queries ---
            if prompts_to_infer:
                indices, prompts = zip(*prompts_to_infer)
                raw_responses = _batch_chat_inference(
                    query_generator.inference_url,
                    list(prompts),
                    query_generator.temperature,
                    logger=logger,
                )
                for j, idx in enumerate(indices):
                    try:
                        query = parse_dspy_output(raw_responses[j].strip())
                    except ValueError as exc:
                        fallback_query = _fallback_query_from_question(
                            chunk[idx].get("question", "")
                        )
                        logger.warning(
                            "Batch query-generation parse failure for qid=%s; "
                            "falling back to question text. Error=%s",
                            chunk_qids[idx],
                            exc,
                        )
                        query = fallback_query
                    query_results[idx] = query
                    if query_generator.use_cache and idx in cache_keys:
                        with query_generator._cache_lock:
                            query_generator.cache[cache_keys[idx]] = query

            queries = [query_results[i] for i in range(len(chunk))]

            # --- ColBERT retrieval (serial — file handle not thread-safe) ---
            exclude_pids = [seen_pids[qid] for qid in chunk_qids]
            batch_passages = _retrieve_batch(
                retriever, queries, exclude_pids, top_k_per_hop,
            )

            # --- Update state ---
            for i, q in enumerate(chunk):
                qid = chunk_qids[i]
                for p in batch_passages[i]:
                    if p["doc_id"] not in seen_pids[qid]:
                        seen_pids[qid].add(p["doc_id"])
                        p["hop"] = hop
                        all_retrieved[qid].append(p)

                gold_titles = shared_extract_gold_titles(q, dataset=helper_dataset)
                hop_coverage = compute_gold_coverage(all_retrieved[qid], gold_titles)
                hop_details_map[qid].append({
                    "hop": hop,
                    "query": queries[i],
                    "num_retrieved": len(batch_passages[i]),
                    "cumulative_retrieved": len(all_retrieved[qid]),
                    "coverage_after_hop": hop_coverage["coverage"],
                })

            logger.info(
                "Chunk %d/%d, Hop %d: %d inferred, %d cached",
                chunk_idx + 1, num_chunks, hop,
                len(prompts_to_infer), len(chunk) - len(prompts_to_infer),
            )

        # Build rollout records for this chunk
        for i, q in enumerate(chunk):
            qid = chunk_qids[i]
            gold_titles = shared_extract_gold_titles(q, dataset=helper_dataset)

            gold_titles_lower = {t.lower().strip() for t in gold_titles}
            for p in all_retrieved[qid]:
                p["is_gold"] = p.get("title", "").lower().strip() in gold_titles_lower

            final_coverage = compute_gold_coverage(all_retrieved[qid], gold_titles)

            rollouts_dict[qid] = {
                "dataset": dataset_name,
                "qid": qid,
                "base_qid": q.get("base_qid", _get_qid(q)[1]),
                "question": q.get("question", ""),
                "answer": q.get("answer", ""),
                "gold_titles": gold_titles,
                "num_hops": num_hops,
                "prompt_variant_id": q.get("prompt_variant_id"),
                "prompt_variant_source": q.get("prompt_variant_source"),
                "retrieved": all_retrieved[qid],
                "hop_details": hop_details_map[qid],
                "gold_coverage": final_coverage["coverage"],
                "gold_hit": final_coverage["hit"],
                "gold_total": final_coverage["total"],
            }
            total_coverage += final_coverage["coverage"]

        newly_completed += len(chunk)
        elapsed = time.time() - start_time
        avg_coverage = total_coverage / len(rollouts_dict) if rollouts_dict else 0

        logger.info(
            "Chunk %d/%d done: %d total, avg coverage %.1f%%, %.2f q/s",
            chunk_idx + 1, num_chunks, len(rollouts_dict),
            avg_coverage * 100, newly_completed / elapsed,
        )

        # Checkpoint after every chunk
        if checkpoint_path:
            _save_checkpoint(checkpoint_path, list(rollouts_dict.values()), logger)
            query_generator.save_cache()

    rollouts = list(rollouts_dict.values())
    elapsed = time.time() - start_time
    avg_coverage = total_coverage / len(rollouts) if rollouts else 0
    logger.info(
        "Final: %d rollouts, avg coverage %.1f%%, %.2f q/s",
        len(rollouts), avg_coverage * 100,
        newly_completed / elapsed if elapsed > 0 else 0,
    )
    return rollouts


# ============================================
# Data Loaders
# ============================================

def _rollout_question_from_normalized_record(
    record: dict,
    *,
    dataset_name: str,
    adapter,
) -> dict:
    """Project a normalized record into the compact rollout question schema."""
    normalized = NormalizedQuestion.from_record(record)

    out = {
        "_id": normalized.qid,
        "qid": normalized.qid,
        "question": normalized.question,
        "answer": normalized.answer,
        "gold_titles": adapter.extract_gold_titles(record),
        "dataset": dataset_name,
    }
    # Preserve metadata needed by adapter-backed helper paths (MuSiQue hops).
    for key in ("num_hops", "question_decomposition", "decomposition", "split", "is_answerable"):
        if key in record:
            out[key] = record[key]
    return out


def load_hotpot_fullwiki(raw_dir: Path, split: str = "dev", limit: int | None = None) -> list[dict]:
    """Load HotpotQA fullwiki questions as rollout-ready records.
    
    Handles different filename patterns:
    - dev: hotpot_dev_fullwiki_v1.json
    - train: hotpot_train_v1.1.json (full) or hotpot_train_sample_10k.json (sample)
    
    For train split: uses full dataset if limit > 10k, otherwise uses sample for speed.
    """
    hotpot_dir = raw_dir / "hotpot_fullwiki"
    
    if split == "dev":
        path = hotpot_dir / "hotpot_dev_fullwiki_v1.json"
    elif split == "train":
        sample_path = hotpot_dir / "hotpot_train_sample_10k.json"
        full_path = hotpot_dir / "hotpot_train_v1.1.json"
        
        # Use full dataset if limit > 10k or full dataset doesn't have sample
        if limit and limit > 10000:
            path = full_path
        else:
            path = sample_path if sample_path.exists() else full_path
    elif split == "test":
        path = hotpot_dir / "hotpot_test_fullwiki_v1.json"
    else:
        raise ValueError(f"Unknown split: {split}")
    
    if not path.exists():
        raise FileNotFoundError(f"HotpotQA not found at {path}")
    
    with open(path) as f:
        rows = json.load(f)
    if limit is not None:
        rows = rows[:limit]

    adapter = get_dataset_adapter("hotpot_fullwiki")
    normalized_rows = [adapter.normalize_to_record(item, split=split) for item in rows]
    return [
        _rollout_question_from_normalized_record(
            record, dataset_name="hotpot_fullwiki", adapter=adapter
        )
        for record in normalized_rows
    ]


def load_musique_questions(norm_dir: Path) -> list[dict]:
    """Load normalized MuSiQue questions as rollout-ready records."""
    path = norm_dir / "musique.jsonl"
    
    if not path.exists():
        raise FileNotFoundError(f"MuSiQue not found at {path}")
    
    adapter = get_dataset_adapter("musique")
    return [
        _rollout_question_from_normalized_record(item, dataset_name="musique", adapter=adapter)
        for item in read_jsonl(path)
    ]


# ============================================
# Main
# ============================================

def main() -> None:
    parser = argparse.ArgumentParser(description="""
    Generate multi-hop retrieval rollouts using ColBERTv2 over Wikipedia.
    
    LeReT-style: Same query generator for ALL hops.
    - Hop 1: generate_query(question, context=[])
    - Hop 2+: generate_query(question, context=accumulated_passages)
    """)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None, help="Override output file path")
    parser.add_argument("--dataset", required=True, choices=["hotpot", "musique", "both"])
    parser.add_argument("--top-k-per-hop", type=int, default=5, 
                        help="Number of passages to retrieve per hop")
    parser.add_argument("--limit", type=int, default=None, help="Limit questions (for testing)")
    parser.add_argument("--split", default="dev", choices=["dev", "test", "train"])
    parser.add_argument("--model", default="openrouter/meta-llama/llama-3-8b-instruct",
                        help="Model for query generation")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Sampling temperature for query generation")
    parser.add_argument("--no-rewriter", action="store_true",
                        help="Baseline: use original question for hop 1 (no rewriter)")
    parser.add_argument("--prompt-variants", default=None,
                        help="JSON list/dict of prompt templates for diverse query generation")
    parser.add_argument("--dspy-state", default=None,
                        help="DSPy state JSON with bootstrapped demos (for few-shot prompts)")
    parser.add_argument("--variants-per-question", type=int, default=1,
                        help="How many prompt variants to use per question")
    parser.add_argument("--prompt-variant-mode", choices=["sample", "all"], default="sample",
                        help="Use a random sample or all prompt variants per question")
    parser.add_argument("--seed", type=int, default=42, help="Seed for variant sampling")
    parser.add_argument("--no-cache", action="store_true", help="Disable query generator cache")
    parser.add_argument(
        "--llm-cache-path",
        default=None,
        help="Optional path for the query-generator response cache JSON.",
    )
    parser.add_argument("--num-threads", type=int, default=16,
                        help="Number of threads for parallel processing (LeReT uses 32-128)")
    parser.add_argument("--checkpoint-interval", type=int, default=100,
                        help="Save checkpoint every N questions (default: 100)")
    parser.add_argument("--offset", type=int, default=0,
                        help="Skip first N questions (for question slicing)")
    parser.add_argument(
        "--hop-mix",
        default=None,
        help="Hop sampling mix for MuSiQue in hop=value format, e.g. '2=69.91,3=23.02,4=7.07'.",
    )
    parser.add_argument(
        "--hop-mix-total",
        type=int,
        default=None,
        help="Total number of base questions to sample when using --hop-mix. "
             "If omitted, uses --limit; if neither is set, uses all loaded questions.",
    )
    parser.add_argument(
        "--inference-url", default=None,
        help="Local inference server URL (e.g. http://localhost:8000). "
             "When set, uses local model instead of OpenRouter for on-policy generation.",
    )
    parser.add_argument(
        "--remote-url",
        default=None,
        help="Remote ColBERT retrieval server URL (serve_colbert_min.py). "
             "When set, skips local ColBERT index loading and queries /api/search over HTTP.",
    )
    parser.add_argument(
        "--remote-timeout-s",
        type=float,
        default=90.0,
        help="HTTP timeout in seconds for remote ColBERT /health and /api/search requests.",
    )
    parser.add_argument(
        "--remote-max-retries",
        type=int,
        default=6,
        help="Max retry attempts for transient remote ColBERT request failures.",
    )
    parser.add_argument(
        "--remote-retry-backoff-s",
        type=float,
        default=2.0,
        help="Base backoff in seconds for remote ColBERT retries (exponential).",
    )
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Batch size for GPU-batched inference (used with --inference-url)")
    parser.add_argument(
        "--llm-timeout-s",
        type=float,
        default=45.0,
        help="HTTP timeout in seconds for query-generation requests.",
    )
    parser.add_argument(
        "--llm-max-retries",
        type=int,
        default=4,
        help="Max retry attempts for transient query-generation failures (timeouts/429/5xx).",
    )
    parser.add_argument(
        "--llm-retry-backoff-s",
        type=float,
        default=1.5,
        help="Base backoff in seconds for query-generation retries (exponential).",
    )
    parser.add_argument(
        "--llm-max-output-tokens",
        type=int,
        default=512,
        help="Max tokens for OpenRouter query-generation completions.",
    )
    args = parser.parse_args()
    hop_mix = _parse_hop_mix(args.hop_mix)
    
    use_rewriter = not args.no_rewriter
    num_threads = args.num_threads
    
    config = yaml.safe_load(Path(args.config).read_text())
    raw_dir = Path(config["raw_dir"])
    norm_dir = Path(config["norm_dir"])
    rollouts_dir = Path(config.get("rollouts_dir", "data/rollouts"))
    logs_dir = Path(config["logs_dir"])
    
    logger = get_logger("multihop_rollouts", logs_dir / "multihop_rollouts.log")
    error_log_path = logs_dir / "multihop_rollouts_errors.log"
    try:
        ensure_dir(error_log_path.parent)
        error_log_handle = open(error_log_path, "a")
        faulthandler.enable(error_log_handle)
    except Exception as exc:
        logger.warning("Failed to enable faulthandler: %s", exc)
    
    # Load API key (not required when using local inference)
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        env_path = Path(".env")
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("OPENROUTER_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                    break

    if not api_key and not args.inference_url:
        logger.error("OPENROUTER_API_KEY not found. Set it in .env or environment, or use --inference-url.")
        return

    if args.inference_url:
        logger.info("Using LOCAL inference server: %s", args.inference_url)
    
    # Load ColBERT retriever (remote or local)
    try:
        retriever = _build_retriever(
            raw_dir=raw_dir,
            logger=logger,
            remote_url=args.remote_url,
            remote_timeout_s=args.remote_timeout_s,
            remote_max_retries=args.remote_max_retries,
            remote_retry_backoff_s=args.remote_retry_backoff_s,
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return
    except Exception as exc:
        logger.error("Failed to initialize ColBERT retriever: %s", exc)
        return
    
    # Setup cache path for LLM responses
    if args.llm_cache_path:
        llm_cache_path = Path(args.llm_cache_path)
        ensure_dir(llm_cache_path.parent)
    else:
        cache_dir = Path(config.get("cache_dir", "data/cache"))
        ensure_dir(cache_dir)
        llm_cache_path = cache_dir / "query_generator_cache.json"
    
    prompt_variants = _load_prompt_variants(args.prompt_variants)
    prompt_variants_from_file = bool(prompt_variants)
    prompt_variant_source = "prompt_variants" if prompt_variants_from_file else "default"
    if args.dspy_state:
        try:
            dspy_prompts = prompts_from_dspy_state(Path(args.dspy_state))
        except Exception as exc:
            logger.error("Failed to load DSPy state prompts: %s", exc)
            return
        if (
            args.prompt_variant_mode == "all"
            and args.variants_per_question
            and len(dspy_prompts) > args.variants_per_question
        ):
            dspy_prompts = dspy_prompts[:args.variants_per_question]
            logger.info(
                "Capped DSPy prompts to %d for mode=all",
                len(dspy_prompts),
            )
        if dspy_prompts:
            if prompt_variants_from_file:
                prompt_variants.extend(dspy_prompts)
                prompt_variant_source = "mixed"
            else:
                prompt_variants = dspy_prompts
                prompt_variant_source = "dspy_state"
        else:
            logger.warning("No prompt variants extracted from DSPy state: %s", args.dspy_state)
    if prompt_variants:
        logger.info(
            "Loaded %d prompt variants (source=%s, mode=%s, per-question=%d)",
            len(prompt_variants),
            prompt_variant_source,
            args.prompt_variant_mode,
            args.variants_per_question,
        )
    query_generator = QueryGenerator(
        api_key or "",
        args.model,
        logger,
        cache_path=None if args.no_cache else llm_cache_path,
        prompt_variants=prompt_variants,
        use_cache=not args.no_cache,
        temperature=args.temperature,
        inference_url=args.inference_url,
        request_timeout_s=args.llm_timeout_s,
        max_retries=args.llm_max_retries,
        retry_backoff_s=args.llm_retry_backoff_s,
        max_output_tokens=args.llm_max_output_tokens,
    )
    
    ensure_dir(rollouts_dir)
    
    # Log mode
    mode = "rewriter" if use_rewriter else "baseline"
    logger.info("Query generation mode: %s (LeReT-style: same generator for all hops)", mode)
    if use_rewriter:
        logger.info("  - Hop 1: query = generate_query(question, context=[])")
        logger.info("  - Hop 2+: query = generate_query(question, context=accumulated)")
    else:
        logger.info("  - Hop 1: query = original question (no rewrite)")
        logger.info("  - Hop 2+: query = generate_query(question, context=accumulated)")
    
    datasets_to_process = []
    if args.dataset in ["hotpot", "both"]:
        # When using offset, need enough questions to cover offset + limit
        load_limit = (args.offset + args.limit) if (args.offset and args.limit) else args.limit
        datasets_to_process.append(("hotpot_fullwiki", load_hotpot_fullwiki(raw_dir, args.split, load_limit)))
    if args.dataset in ["musique", "both"]:
        datasets_to_process.append(("musique", load_musique_questions(norm_dir)))
    
    for dataset_name, questions in datasets_to_process:
        logger.info("\n=== Processing %s (%d questions) ===", dataset_name, len(questions))

        if hop_mix:
            if dataset_name != "musique":
                raise ValueError("--hop-mix is only supported for --dataset musique")
            if args.offset > 0:
                raise ValueError("--offset cannot be combined with --hop-mix")
            if args.hop_mix_total is not None and args.limit is not None and args.hop_mix_total != args.limit:
                raise ValueError("--hop-mix-total and --limit both set but differ")
            hop_total = args.hop_mix_total if args.hop_mix_total is not None else args.limit
            if hop_total is None:
                hop_total = len(questions)
            questions = _sample_questions_by_hop_mix(
                questions,
                dataset_name=dataset_name,
                hop_mix=hop_mix,
                total_questions=hop_total,
                seed=args.seed,
                logger=logger,
            )
            logger.info("Hop-mix limited to %d questions", len(questions))
        else:
            if args.offset > 0:
                questions = questions[args.offset:]
                logger.info("Offset: skipping first %d questions", args.offset)
            if args.limit:
                questions = questions[:args.limit]
                logger.info("Limited to %d questions", len(questions))
        
        if prompt_variants:
            questions = _expand_questions_with_variants(
                questions,
                prompt_variants,
                args.variants_per_question,
                args.prompt_variant_mode,
                args.seed,
                prompt_variant_source,
            )
            logger.info("Expanded to %d questions with prompt variants", len(questions))
        
        # Checkpoint path for resumable runs
        mode_suffix = "rewriter" if use_rewriter else "baseline"
        variant_suffix = ""
        if prompt_variant_source == "dspy_state":
            variant_suffix = "_fewshot"
        elif prompt_variant_source == "mixed":
            variant_suffix = "_mixed"
        if args.output:
            checkpoint_path = Path(args.output).with_suffix(".checkpoint.jsonl")
        elif dataset_name == "hotpot_fullwiki" and args.split != "dev":
            checkpoint_name = (
                f"hotpot_{args.split}_multihop_k{args.top_k_per_hop}_{mode_suffix}{variant_suffix}.checkpoint.jsonl"
            )
            checkpoint_path = rollouts_dir / checkpoint_name
        else:
            checkpoint_name = f"{dataset_name}_multihop_k{args.top_k_per_hop}_{mode_suffix}{variant_suffix}.checkpoint.jsonl"
            checkpoint_path = rollouts_dir / checkpoint_name
        
        if args.inference_url:
            # Batched path: GPU-batched inference + serial ColBERT
            logger.info(
                "Using BATCHED pipeline: batch_size=%d, serial ColBERT",
                args.batch_size,
            )
            rollouts = generate_rollouts_batched(
                questions, retriever, query_generator, dataset_name,
                args.top_k_per_hop, use_rewriter, logger,
                batch_size=args.batch_size,
                checkpoint_path=checkpoint_path,
            )
        else:
            rollouts = generate_rollouts(
                questions, retriever, query_generator, dataset_name,
                args.top_k_per_hop, use_rewriter, logger, num_threads,
                checkpoint_path, args.checkpoint_interval
            )
        
        # Save final rollouts (include mode and split in filename)
        if args.output:
            output_path = Path(args.output)
        elif dataset_name == "hotpot_fullwiki" and args.split != "dev":
            output_name = f"hotpot_{args.split}_multihop_k{args.top_k_per_hop}_{mode_suffix}{variant_suffix}.jsonl"
            output_path = rollouts_dir / output_name
        else:
            output_name = f"{dataset_name}_multihop_k{args.top_k_per_hop}_{mode_suffix}{variant_suffix}.jsonl"
            output_path = rollouts_dir / output_name
        write_jsonl(output_path, rollouts)
        logger.info("Saved %d rollouts to %s", len(rollouts), output_path)
        
        # Remove checkpoint file after successful completion
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info("Removed checkpoint file %s", checkpoint_path)
        
        # Summary stats
        coverages = [r["gold_coverage"] for r in rollouts]
        hop_counts = [r["num_hops"] for r in rollouts]
        
        logger.info("\n--- %s Summary ---", dataset_name)
        logger.info("Total questions: %d", len(rollouts))
        logger.info("Mean gold coverage: %.1f%%", np.mean(coverages) * 100)
        logger.info("Median gold coverage: %.1f%%", np.median(coverages) * 100)
        logger.info("100%% coverage: %d (%.1f%%)",
                   sum(1 for c in coverages if c >= 1.0),
                   sum(1 for c in coverages if c >= 1.0) / len(coverages) * 100)
        logger.info("Avg hops: %.1f", np.mean(hop_counts))
        
        # Per-hop coverage improvement
        if rollouts:
            for hop in range(1, max(hop_counts) + 1):
                hop_coverages = []
                for r in rollouts:
                    for hd in r["hop_details"]:
                        if hd["hop"] == hop:
                            hop_coverages.append(hd["coverage_after_hop"])
                if hop_coverages:
                    logger.info("Coverage after hop %d: %.1f%% (n=%d)",
                               hop, np.mean(hop_coverages) * 100, len(hop_coverages))


if __name__ == "__main__":
    main()
