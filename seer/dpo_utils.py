from __future__ import annotations

import re
from typing import Any


_HOP_KEY_RE = re.compile(r"^hop(\d+)$")


def iter_hop_records(rollout: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    """Return ordered hop records from either canonical or legacy DPO payloads."""
    hops = rollout.get("hops")
    items: list[tuple[int, dict[str, Any]]] = []

    if isinstance(hops, list):
        for idx, hop_record in enumerate(hops, start=1):
            if not isinstance(hop_record, dict):
                continue
            hop_num = hop_record.get("hop", idx)
            try:
                hop_num = int(hop_num)
            except (TypeError, ValueError):
                hop_num = idx
            items.append((hop_num, hop_record))

    if not items:
        for key, value in rollout.items():
            match = _HOP_KEY_RE.match(str(key))
            if not match or not isinstance(value, dict):
                continue
            items.append((int(match.group(1)), value))

    items.sort(key=lambda x: x[0])
    return items


def hop_record_map(rollout: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {hop_num: hop_record for hop_num, hop_record in iter_hop_records(rollout)}


def selected_candidate_idx_from_context_source(hop_record: dict[str, Any]) -> int | None:
    context_source = hop_record.get("context_source") or {}
    idx = context_source.get("selected_candidate_idx")
    if idx is None:
        idx = context_source.get("selected_idx")  # legacy field
    if idx is None:
        return None
    try:
        return int(idx)
    except (TypeError, ValueError):
        return None


def selected_context_docs_upto_hop(
    rollout: dict[str, Any],
    hop_num: int,
) -> list[dict[str, Any]]:
    """Return selected shared-context docs accumulated up to hop_num-1."""
    if hop_num <= 1:
        return []

    hmap = hop_record_map(rollout)
    docs: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    for next_hop in range(2, hop_num + 1):
        prev_hop = next_hop - 1
        prev_record = hmap.get(prev_hop)
        next_record = hmap.get(next_hop)
        if not prev_record or not next_record:
            continue

        selected_idx = selected_candidate_idx_from_context_source(next_record)
        if selected_idx is None:
            continue
        candidates = prev_record.get("candidates") or []
        if not (0 <= selected_idx < len(candidates)):
            continue

        retrieved = candidates[selected_idx].get("retrieved") or []
        for passage in retrieved:
            pid = passage.get("doc_id")
            if pid is not None:
                try:
                    key = ("pid", int(pid))
                except (TypeError, ValueError):
                    key = ("pid_raw", str(pid))
            else:
                key = (
                    "text",
                    str(passage.get("title", "")),
                    str(passage.get("text", "")),
                )
            if key in seen:
                continue
            seen.add(key)
            docs.append(passage)

    return docs

