from __future__ import annotations

import asyncio
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from seer.util_hash import sha256_text
from seer.util_io import ensure_dir, read_json, write_json


@dataclass
class OpenRouterConfig:
    base_url: str
    api_key: str
    concurrency: int = 10
    temperature: float = 0.0


MODEL_ALIASES = {
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "claude-3-haiku": "anthropic/claude-3-haiku",
    "gemini": "google/gemini-1.5-flash",
}


def normalize_model_slug(model: str) -> str:
    # Allow callers to pass a convenience prefix like:
    #   openrouter/meta-llama/llama-3-8b-instruct
    # OpenRouter API expects provider/model without that prefix.
    if isinstance(model, str) and model.startswith("openrouter/"):
        model = model[len("openrouter/") :]
    if "/" in model:
        return model
    return MODEL_ALIASES.get(model, model)


def _cache_path(cache_dir: str | Path, cache_key: str) -> Path:
    return Path(cache_dir) / f"{cache_key}.json"


def cached_completion(
    cache_dir: str | Path,
    cache_key: str,
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    path = _cache_path(cache_dir, cache_key)
    if path.exists():
        try:
            return read_json(path)
        except Exception:
            # Corrupt cache file — delete and re-fetch
            path.unlink(missing_ok=True)
            return None
    return None


def save_cached_completion(cache_dir: str | Path, cache_key: str, payload: dict[str, Any]) -> None:
    import json as _json
    path = _cache_path(cache_dir, cache_key)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        _json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))


def build_cache_key(model: str, system_prompt: str, user_prompt: str, question: str, context: str) -> str:
    normalized_model = normalize_model_slug(model)
    return sha256_text(normalized_model, system_prompt, user_prompt, question, context)


async def _post_completion(
    client: httpx.AsyncClient,
    config: OpenRouterConfig,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int | None = None,
    logprobs: bool = False,
    top_logprobs: int | None = None,
) -> dict[str, Any]:
    normalized_model = normalize_model_slug(model)
    payload = {
        "model": normalized_model,
        "messages": messages,
        "temperature": config.temperature,
        "provider": {"sort": "throughput"},
    }
    if logprobs:
        payload["logprobs"] = True
        if top_logprobs is not None:
            payload["top_logprobs"] = top_logprobs
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    retries = 5
    for attempt in range(retries):
        response = await client.post(
            f"{config.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {config.api_key}"},
            json=payload,
            timeout=120,
        )
        if response.status_code in {429, 500, 502, 503, 504} and attempt < retries - 1:
            await asyncio.sleep(2**attempt + random.random())
            continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError("OpenRouter request failed after retries")


async def run_completions(
    config: OpenRouterConfig,
    cache_dir: str | Path,
    requests: list[dict[str, Any]],
    logprobs: bool = False,
    top_logprobs: int | None = None,
    cache_logprobs: bool = False,
) -> list[dict[str, Any]]:
    from tqdm.asyncio import tqdm
    semaphore = asyncio.Semaphore(config.concurrency)

    # Use a single shared client with connection pooling for all requests.
    # HTTP/2 + keep-alive avoids per-request TCP/TLS overhead.
    limits = httpx.Limits(
        max_connections=config.concurrency + 10,
        max_keepalive_connections=config.concurrency,
    )
    async with httpx.AsyncClient(http2=False, limits=limits) as client:

        async def _bounded(request: dict[str, Any]) -> dict[str, Any]:
            cache_key = request["cache_key"]
            cached = cached_completion(cache_dir, cache_key, request)
            if cached:
                return cached

            async with semaphore:
                try:
                    response = await _post_completion(
                        client,
                        config,
                        request["model"],
                        request["messages"],
                        max_tokens=request.get("max_tokens"),
                        logprobs=logprobs,
                        top_logprobs=top_logprobs,
                    )
                    normalized_model = normalize_model_slug(request["model"])
                    # Strip logprobs from cached response to save disk space
                    # (~80KB per file → ~5KB without logprobs)
                    # Unless cache_logprobs=True (for short responses needing logprob analysis)
                    save_response = response
                    if not cache_logprobs and isinstance(response, dict) and "choices" in response:
                        save_response = {**response}
                        save_response["choices"] = [
                            {k: v for k, v in c.items() if k != "logprobs"}
                            for c in response["choices"]
                        ]
                    record = {
                        "model": normalized_model,
                        "response": save_response,
                        "timestamp": datetime.utcnow().isoformat(),
                        "meta": request.get("meta", {}),
                    }
                    save_cached_completion(cache_dir, cache_key, record)
                    # Return original response (with logprobs) to caller
                    if save_response is not response:
                        return {**record, "response": response}
                    return record
                except Exception as e:
                    print(f"\nError in task {request.get('meta', {}).get('qid')}: {e}")
                    return {
                        "error": str(e),
                        "meta": request.get("meta", {}),
                        "cache_key": cache_key
                    }

        tasks = [_bounded(req) for req in requests]
        results = await tqdm.gather(*tasks, desc="OpenRouter Completions")
        return list(results)


def load_openrouter_config(config: dict[str, Any]) -> OpenRouterConfig:
    api_key = os.getenv(config["api_key_env"], "")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    return OpenRouterConfig(
        base_url=config["base_url"],
        api_key=api_key,
        concurrency=config.get("concurrency", 10),
        temperature=config.get("temperature", 0.0),
    )
