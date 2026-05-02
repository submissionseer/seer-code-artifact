from __future__ import annotations

import math
from typing import Any

from seer.parse_seer_xml import SeerParseResult


def _compute_seer_ap(present_map: dict[str, list[int]], num_passages: int, total_req: int) -> float:
    """Requirement-level Average Precision.

    Treats requirements as the 'relevant items to find' (analogous to gold
    documents in standard IR AP).  For each requirement, finds the earliest
    passage rank that satisfies it, then computes AP with
    denominator = total_requirements.

    When multiple requirements share the same first-satisfying passage,
    precision is capped at 1.0 (analogous to tied relevant docs).
    """
    if total_req == 0:
        return 1.0
    if num_passages <= 0:
        return 0.0

    # For each requirement, find the earliest passage that satisfies it
    from collections import defaultdict
    req_first_rank: dict[str, int] = {}
    for fid, pids in present_map.items():
        valid = [p for p in pids if isinstance(p, int) and 1 <= p <= num_passages]
        if valid:
            req_first_rank[fid] = min(valid)

    if not req_first_rank:
        return 0.0

    # Group requirements by first-satisfying rank
    rank_groups: dict[int, int] = defaultdict(int)
    for rank in req_first_rank.values():
        rank_groups[rank] += 1

    # Compute AP: walk ranks in order, accumulate satisfied requirements
    ap_sum = 0.0
    reqs_found = 0
    for rank in sorted(rank_groups):
        n_new = rank_groups[rank]
        reqs_found += n_new
        precision = min(reqs_found / rank, 1.0)
        ap_sum += precision * n_new

    return ap_sum / total_req


def compute_seer_metrics(parsed: SeerParseResult, num_passages: int) -> dict[str, Any]:
    if not parsed.valid:
        return {
            "seer_recall": 0.0,
            "irrelevance_rate": 1.0,
            "seer_f1": 0.0,
            "coverage_ndcg": 0.0,
            "seer_ap": 0.0,
            "valid": False,
        }

    total_req = len(parsed.requirements)
    present_ids = {f"f{idx+1}" for idx in range(total_req)} & set(parsed.present_map.keys())
    recall = len(present_ids) / max(total_req, 1)
    support_passages = {pid for pids in parsed.present_map.values() for pid in pids}
    support_concentration = len(support_passages) / max(num_passages, 1)
    irrelevance_rate = 1 - support_concentration
    if recall + (1 - irrelevance_rate) == 0:
        f1 = 0.0
    else:
        f1 = 2 * recall * (1 - irrelevance_rate) / (recall + (1 - irrelevance_rate))

    gains = []
    seen = set()
    for rank in range(1, num_passages + 1):
        new = 0
        for fid, pids in parsed.present_map.items():
            if fid in seen:
                continue
            if rank in pids:
                new += 1
                seen.add(fid)
        gains.append(new / max(total_req, 1))

    dcg = sum(gain / math.log2(idx + 2) for idx, gain in enumerate(gains))
    ideal = sorted(gains, reverse=True)
    idcg = sum(gain / math.log2(idx + 2) for idx, gain in enumerate(ideal))
    coverage_ndcg = dcg / idcg if idcg else 0.0
    seer_ap = _compute_seer_ap(parsed.present_map, num_passages, total_req)

    return {
        "seer_recall": recall,
        "irrelevance_rate": irrelevance_rate,
        "seer_f1": f1,
        "coverage_ndcg": coverage_ndcg,
        "seer_ap": seer_ap,
        "valid": True,
    }
