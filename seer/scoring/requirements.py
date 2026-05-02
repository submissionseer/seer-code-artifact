from __future__ import annotations

import re
from pathlib import Path

from seer.openrouter import OpenRouterConfig, normalize_model_slug, run_completions
from seer.util_hash import sha256_text
from seer.util_io import read_json, write_json


REQ_SYSTEM_PROMPT = """You are a careful evaluator. Given a question, identify the minimal set of disjoint information requirements needed to fully answer it.

Rules:
- List requirements as f1), f2), f3), etc. with no gaps.
- Each requirement should describe ONE specific piece of information needed.
- Requirements should be minimal and non-overlapping.
- For multi-hop questions, include intermediate requirements (e.g., "The director of Inception" before "The nationality of the director of Inception").
- Do NOT reference any passages or documents — describe information needs abstractly.
- Output ONLY the numbered requirements, nothing else."""

REQ_USER_TEMPLATE = """Question: {question}

List the minimal disjoint information requirements:"""

_REQ_LINE_RE = re.compile(r"^f(\d+)\)\s*(.+)$")


def parse_requirements(text: str) -> list[str]:
    reqs: list[str] = []
    for line in str(text or "").strip().splitlines():
        line = line.strip()
        match = _REQ_LINE_RE.match(line)
        if match:
            reqs.append(match.group(2).strip())
    return reqs


def load_requirements_map(path: str | Path) -> dict[str, list[str]]:
    payload = read_json(path)
    out: dict[str, list[str]] = {}
    for key, values in payload.items():
        if isinstance(values, list):
            out[str(key)] = [str(v).strip() for v in values if str(v).strip()]
    return out


async def generate_requirements_map(
    questions_by_key: dict[str, str],
    *,
    model: str,
    openrouter_config: OpenRouterConfig,
    cache_dir: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, list[str]]:
    requests = []
    normalized = normalize_model_slug(model)
    for key, question in questions_by_key.items():
        q = str(question or "").strip()
        if not q:
            continue
        user_prompt = REQ_USER_TEMPLATE.format(question=q)
        cache_key = sha256_text(
            normalized,
            REQ_SYSTEM_PROMPT,
            user_prompt,
            q,
            "",
            "requirements_generation",
        )
        requests.append(
            {
                "model": model,
                "messages": [
                    {"role": "system", "content": REQ_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "cache_key": cache_key,
                "meta": {"key": str(key), "question": q},
            }
        )

    results = await run_completions(openrouter_config, cache_dir, requests)

    requirements: dict[str, list[str]] = {}
    for request, result in zip(requests, results):
        key = str(request["meta"]["key"])
        response = result.get("response", {})
        choices = response.get("choices", []) if isinstance(response, dict) else []
        content = ""
        if choices:
            content = str(choices[0].get("message", {}).get("content", ""))
        requirements[key] = parse_requirements(content)

    if output_path:
        write_json(output_path, requirements)

    return requirements
