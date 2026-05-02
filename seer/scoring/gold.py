from __future__ import annotations

from typing import Any

from seer.rollout_metrics import _normalize_title


def _bounded_ap_score(retrieved_titles: list[str], gold_titles: list[str]) -> float:
    """Average precision with duplicate-gold-hit suppression (bounded in [0, 1])."""
    gold = {_normalize_title(t) for t in gold_titles if t}
    if not gold:
        return 0.0

    hits = 0
    score = 0.0
    seen_gold: set[str] = set()
    for idx, title in enumerate(retrieved_titles, start=1):
        norm = _normalize_title(title)
        if norm in gold and norm not in seen_gold:
            seen_gold.add(norm)
            hits += 1
            score += hits / idx
    return score / len(gold)


def score_gold_candidate(
    *,
    retrieved_passages: list[dict[str, Any]],
    gold_titles: list[str],
    prior_titles: list[str] | None = None,
) -> dict[str, float]:
    titles = [str(p.get("title", "")) for p in retrieved_passages]
    cumulative_titles = list(prior_titles or []) + titles
    return {
        "gold_ap": float(_bounded_ap_score(titles, gold_titles)),
        "gold_ap_cumulative": float(_bounded_ap_score(cumulative_titles, gold_titles)),
    }


def score_gold_candidates(
    *,
    candidates: list[dict[str, Any]],
    gold_titles: list[str],
    prior_titles: list[str] | None = None,
) -> list[dict[str, float]]:
    return [
        score_gold_candidate(
            retrieved_passages=list(c.get("retrieved", []) or []),
            gold_titles=gold_titles,
            prior_titles=prior_titles,
        )
        for c in candidates
    ]


def score_key_for_metric(metric: str, hop_num: int) -> str:
    if metric == "gold":
        return "gold_ap" if hop_num == 1 else "gold_ap_cumulative"
    if metric == "seer":
        return "seer_ap"
    if metric == "mmr":
        return "mmr_ap"
    if metric == "decomp_binary":
        return "decomp_binary_ap"
    raise ValueError(f"Unsupported metric: {metric}")
