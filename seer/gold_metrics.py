from __future__ import annotations

from typing import Iterable


def supporting_recall_at_k(supporting_ids: Iterable[str], ranked_ids: Iterable[str], k: int) -> float:
    supporting_set = set(supporting_ids)
    if not supporting_set:
        return 0.0
    topk = set(list(ranked_ids)[:k])
    return len(supporting_set & topk) / len(supporting_set)


def supporting_hit_all_at_k(supporting_ids: Iterable[str], ranked_ids: Iterable[str], k: int) -> int:
    supporting_set = set(supporting_ids)
    if not supporting_set:
        return 0
    topk = set(list(ranked_ids)[:k])
    return int(supporting_set.issubset(topk))
