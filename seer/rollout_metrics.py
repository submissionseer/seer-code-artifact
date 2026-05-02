from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable
import json


def _normalize_title(title: str) -> str:
    return title.strip().lower()


def ap_score(retrieved_titles: Iterable[str], gold_titles: Iterable[str]) -> float:
    gold = {_normalize_title(t) for t in gold_titles if t}
    if not gold:
        return 0.0
    hits = 0
    score = 0.0
    seen_hits: set[str] = set()
    for idx, title in enumerate(retrieved_titles, start=1):
        normalized_title = _normalize_title(title)
        if normalized_title in gold and normalized_title not in seen_hits:
            hits += 1
            score += hits / idx
            seen_hits.add(normalized_title)
    return score / len(gold)


def recall_score(retrieved_titles: Iterable[str], gold_titles: Iterable[str]) -> float:
    gold = {_normalize_title(t) for t in gold_titles if t}
    if not gold:
        return 0.0
    retrieved = {_normalize_title(t) for t in retrieved_titles if t}
    return len(gold & retrieved) / len(gold)


def _titles_for_hop(
    record: dict,
    hop: int,
    top_k: int | None = None,
    cumulative: bool = True,
) -> list[str]:
    retrieved = record.get("retrieved", [])
    if cumulative:
        retrieved = [p for p in retrieved if (p.get("hop") or 1) <= hop]
        if top_k is not None:
            retrieved = retrieved[: top_k * hop]
    else:
        retrieved = [p for p in retrieved if (p.get("hop") or 1) == hop]
        if top_k is not None:
            retrieved = retrieved[:top_k]
    return [p.get("title", "") for p in retrieved]


def _summarize_records(records: list[dict], top_k: int | None = None) -> dict:
    if not records:
        return {"total": 0, "per_hop": {}}

    per_hop = defaultdict(lambda: {"recall": [], "ap": []})
    for record in records:
        gold_titles = record.get("gold_titles", [])
        num_hops = record.get("num_hops", 2)
        for hop in range(1, num_hops + 1):
            titles = _titles_for_hop(record, hop, top_k=top_k, cumulative=True)
            per_hop[hop]["recall"].append(recall_score(titles, gold_titles))
            per_hop[hop]["ap"].append(ap_score(titles, gold_titles))

    summary = {"total": len(records), "per_hop": {}}
    for hop, values in per_hop.items():
        summary["per_hop"][hop] = {
            "recall": sum(values["recall"]) / len(values["recall"]),
            "ap": sum(values["ap"]) / len(values["ap"]),
        }
    return summary


def summarize_rollouts(path: Path, top_k: int | None = None) -> dict:
    rollouts = []
    with path.open() as handle:
        for line in handle:
            rollouts.append(json.loads(line))
    return _summarize_records(rollouts, top_k=top_k)


def summarize_by_variant(path: Path, top_k: int | None = None) -> dict[int, dict]:
    buckets: dict[int, list[dict]] = defaultdict(list)
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            variant_id = record.get("prompt_variant_id")
            if variant_id is None:
                continue
            buckets[int(variant_id)].append(record)

    per_variant = {}
    for variant_id, records in buckets.items():
        per_variant[variant_id] = _summarize_records(records, top_k=top_k)
    return per_variant


def summarize_best_variant(per_variant: dict[int, dict], hop: int = 2) -> tuple[int | None, dict]:
    best_id = None
    best_summary = {}
    best_ap = -1.0
    for variant_id, summary in per_variant.items():
        hop_summary = summary.get("per_hop", {}).get(hop, {})
        ap = hop_summary.get("ap", -1.0)
        if ap > best_ap:
            best_ap = ap
            best_id = variant_id
            best_summary = summary
    return best_id, best_summary
