from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from seer.openrouter import (
    OpenRouterConfig,
    build_cache_key,
    run_completions,
)
from seer.parse_seer_xml import parse_seer_xml
from seer.seer_metrics import compute_seer_metrics


def format_passages_for_seer(passages: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for i, passage in enumerate(passages, start=1):
        title = str(passage.get("title", "No Title"))
        text = str(passage.get("text", ""))
        chunks.append(f"p{i}: [{title}] {text}")
    return "\n\n".join(chunks)


def _prompt_bundle(prompt_style: str) -> tuple[str, str]:
    if prompt_style == "xml_strict":
        from seer import prompt_seer_xml_strict as prompt_mod
    elif prompt_style == "xml_fewshot":
        from seer import prompt_seer_xml_fewshot as prompt_mod
    elif prompt_style == "xml_v2":
        from seer import prompt_seer_xml_v2 as prompt_mod
    elif prompt_style == "xml_v2b":
        from seer import prompt_seer_xml_v2b as prompt_mod
    elif prompt_style == "xml_v3_hybrid":
        from seer import prompt_seer_xml_v3_hybrid as prompt_mod
    elif prompt_style == "xml_optimized":
        from seer import prompt_seer_xml_optimized as prompt_mod
    else:
        from seer import prompt_seer_xml as prompt_mod
    return prompt_mod.SYSTEM_PROMPT, prompt_mod.USER_PROMPT_TEMPLATE


def _extract_choice_content(result: dict[str, Any]) -> str:
    response = result.get("response", {})
    if not isinstance(response, dict):
        return ""
    choices = response.get("choices", [])
    if not choices:
        return ""
    return str(choices[0].get("message", {}).get("content", ""))


def seer_ap_from_rollout_xml(
    *,
    seer_label: str | None,
    retrieved: list[dict[str, Any]],
    hop_num: int,
) -> float:
    if not seer_label:
        return 0.0
    parsed = parse_seer_xml(str(seer_label))
    if not parsed.valid:
        return 0.0

    allowed_pids = {
        i + 1
        for i, passage in enumerate(retrieved)
        if int(passage.get("hop", 0) or 0) <= int(hop_num)
    }
    filtered_present_map: dict[str, list[int]] = {}
    for fid, pids in parsed.present_map.items():
        filtered = [pid for pid in pids if pid in allowed_pids]
        if filtered:
            filtered_present_map[fid] = filtered

    parsed_filtered = replace(parsed, present_map=filtered_present_map)
    metrics = compute_seer_metrics(parsed_filtered, num_passages=len(allowed_pids))
    return float(metrics.get("seer_ap", 0.0))


async def score_seer_candidates_batch(
    *,
    question: str,
    candidates_passages: list[list[dict[str, Any]]],
    model: str,
    prompt_style: str,
    openrouter_config: OpenRouterConfig,
    cache_dir: str | Path,
    request_meta: dict[str, Any] | None = None,
) -> list[float]:
    system_prompt, user_template = _prompt_bundle(prompt_style)
    requests = []
    for idx, passages in enumerate(candidates_passages):
        passages_str = format_passages_for_seer(passages)
        user_prompt = user_template.format(question=question, passages=passages_str)
        cache_key = build_cache_key(model, system_prompt, user_prompt, question, passages_str)
        meta = {"candidate_idx": idx}
        if request_meta:
            meta.update(request_meta)
        requests.append(
            {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "cache_key": cache_key,
                "meta": meta,
            }
        )

    results = await run_completions(openrouter_config, cache_dir, requests)
    scores: list[float] = []
    for passages, result in zip(candidates_passages, results):
        raw = _extract_choice_content(result)
        parsed = parse_seer_xml(raw)
        metrics = compute_seer_metrics(parsed, num_passages=len(passages))
        scores.append(float(metrics.get("seer_ap", 0.0)))
    return scores
