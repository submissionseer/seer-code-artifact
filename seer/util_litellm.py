from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable


def strip_usage_param(request: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of the request without usage fields."""
    if "usage" not in request:
        return request
    sanitized = dict(request)
    sanitized.pop("usage", None)
    return sanitized


def strip_usage_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of kwargs without usage fields."""
    if "usage" not in kwargs:
        return kwargs
    sanitized = dict(kwargs)
    sanitized.pop("usage", None)
    return sanitized


def _log_usage_payload(label: str, payload: dict[str, Any]) -> None:
    if os.environ.get("SEER_DEBUG_USAGE") != "1":
        return
    if "usage" not in payload:
        return
    path = Path("logs/usage_request_debug.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"label": label, "keys": list(payload.keys())}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def wrap_litellm_completion(
    fn: Callable[[dict[str, Any], int, dict[str, Any] | None], Any],
) -> Callable[[dict[str, Any], int, dict[str, Any] | None], Any]:
    def wrapped(request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None = None):
        _log_usage_payload("dspy_request", request)
        sanitized = strip_usage_param(request)
        return fn(request=sanitized, num_retries=num_retries, cache=cache)

    return wrapped


def wrap_litellm_call(fn: Callable[..., Any]) -> Callable[..., Any]:
    def wrapped(*args, **kwargs):
        _log_usage_payload("litellm_kwargs", kwargs)
        sanitized = strip_usage_kwargs(kwargs)
        return fn(*args, **sanitized)

    return wrapped


def patch_dspy_litellm_usage() -> None:
    from dspy.clients import lm as dspy_lm

    if getattr(dspy_lm, "_seer_usage_patch", False):
        return

    dspy_lm.litellm_completion = wrap_litellm_completion(dspy_lm.litellm_completion)
    dspy_lm.litellm_text_completion = wrap_litellm_completion(dspy_lm.litellm_text_completion)
    dspy_lm.litellm_responses_completion = wrap_litellm_completion(dspy_lm.litellm_responses_completion)
    dspy_lm.alitellm_completion = wrap_litellm_completion(dspy_lm.alitellm_completion)
    dspy_lm.alitellm_text_completion = wrap_litellm_completion(dspy_lm.alitellm_text_completion)
    dspy_lm.alitellm_responses_completion = wrap_litellm_completion(dspy_lm.alitellm_responses_completion)

    dspy_lm._seer_usage_patch = True


def patch_litellm_usage() -> None:
    import litellm

    if getattr(litellm, "_seer_usage_patch", False):
        return

    for name in ("completion", "acompletion", "text_completion", "atext_completion"):
        fn = getattr(litellm, name, None)
        if fn is not None:
            setattr(litellm, name, wrap_litellm_call(fn))

    litellm._seer_usage_patch = True
