from __future__ import annotations

from typing import Any

from .mmr import score_mmr_and_binary_candidates_batch


def decomp_binary_ap_from_rollout(rollout: dict[str, Any], hop_idx: int) -> float:
    hop_num = hop_idx + 1
    keyed = rollout.get(f"decomp_binary_ap_hop{hop_num}")
    if keyed is not None:
        return float(keyed or 0.0)

    details = rollout.get("hop_details", [])
    if len(details) >= hop_num:
        detail = details[hop_idx]
        for key in ("decomp_binary_ap", "decomp_ap", "decomp_binary_score"):
            if key in detail:
                return float(detail.get(key) or 0.0)
    return 0.0


async def score_decomp_binary_candidates_batch(
    *,
    requirements: list[str],
    candidates_passages: list[list[dict[str, Any]]],
    jina_api_key: str,
    cache_dir: str,
    jina_model: str = "jina-reranker-v3",
    concurrency: int = 20,
    threshold: float = 0.5,
) -> list[float]:
    _, binary_scores = await score_mmr_and_binary_candidates_batch(
        requirements=requirements,
        candidates_passages=candidates_passages,
        jina_api_key=jina_api_key,
        cache_dir=cache_dir,
        jina_model=jina_model,
        concurrency=concurrency,
        threshold=threshold,
    )
    return binary_scores
