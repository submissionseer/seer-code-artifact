#!/usr/bin/env python3
"""Validate DPO rollout JSONL quality with strict structural and data checks."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Allow running as `python scripts/dpo/validate.py` without editable install.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from seer.util_io import read_jsonl


def _is_finite_score(v: Any) -> bool:
    try:
        f = float(v)
    except Exception:
        return False
    return math.isfinite(f) and 0.0 <= f <= 1.0


def _metric_score(candidate: dict[str, Any], mode: str, metric_key: str | None) -> Any:
    if metric_key is None:
        return None
    if mode == "decomp_binary":
        score = candidate.get("decomp_binary_ap")
        if score is None:
            score = candidate.get("decomposed_binary_ap")
        return score
    return candidate.get(metric_key)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate DPO rollout output JSONL")
    parser.add_argument("--input", required=True, help="Path to DPO rollout JSONL")
    parser.add_argument(
        "--mode",
        default="gold",
        choices=[
            "gold",
            "mmr",
            "raw_jina_maxmean",
            "raw_jina_max",
            "seer",
            "decomp_binary",
            "uniform",
        ],
        help="Context selection mode used during generation",
    )
    parser.add_argument("--expected-variants", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-zero-retrieved-rate", type=float, default=0.0)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.input))
    if not rows:
        print("FAIL: no rows")
        raise SystemExit(1)

    required_row_keys = {"qid", "question", "gold_titles", "num_hops", "hops"}
    metric_key = {
        "gold": "gold_ap_cumulative",
        "mmr": "mmr_ap",
        "raw_jina_maxmean": "raw_jina_maxmean_top3",
        "raw_jina_max": "raw_jina_max",
        "seer": "seer_ap",
        "decomp_binary": "decomp_binary_ap",
        "uniform": None,
    }[args.mode]

    missing_row_core = 0
    bad_num_hops = 0
    missing_context_source = 0
    bad_selected_index = 0
    bad_prevhop_ref = 0
    empty_query = 0
    marker_query = 0
    narrative_like_query = 0
    invalid_gold_scores = 0
    invalid_metric_scores = 0
    zero_retrieved = 0
    bad_variant_count = 0

    hop_counter: Counter[int] = Counter()
    cand_counter: Counter[int] = Counter()
    retr_counter: Counter[int] = Counter()

    selected_metric_scores: list[float] = []
    selected_metric_best: list[float] = []
    selected_gold_scores: list[float] = []
    selected_gold_best: list[float] = []

    for row in rows:
        if any(k not in row for k in required_row_keys):
            missing_row_core += 1
            continue

        hops = row.get("hops") or []
        num_hops = int(row.get("num_hops") or 0)
        if num_hops != len(hops) or num_hops <= 0:
            bad_num_hops += 1
        hop_counter[num_hops] += 1

        for hop_idx, hop in enumerate(hops, start=1):
            candidates = hop.get("candidates") or []
            cand_counter[len(candidates)] += 1
            if len(candidates) != args.expected_variants:
                bad_variant_count += 1

            for c in candidates:
                query = str(c.get("query", "")).strip()
                if not query:
                    empty_query += 1
                if "[[ ##" in query or "## ]]" in query:
                    marker_query += 1

                q_lower = query.lower()
                if len(q_lower.split()) >= 10 and any(
                    p in q_lower
                    for p in (
                        "there is no",
                        "no information",
                        "provided context",
                        "the question asks",
                        "search query should",
                    )
                ):
                    narrative_like_query += 1

                retrieved = c.get("retrieved") or []
                retr_len = len(retrieved)
                retr_counter[retr_len] += 1
                if retr_len == 0:
                    zero_retrieved += 1
                if retr_len > args.top_k:
                    zero_retrieved += 1  # treat as structural retrieval error

                if not _is_finite_score(c.get("gold_ap")):
                    invalid_gold_scores += 1
                if not _is_finite_score(c.get("gold_ap_cumulative")):
                    invalid_gold_scores += 1

                metric_value = _metric_score(c, args.mode, metric_key)
                if metric_key is not None and not _is_finite_score(metric_value):
                    invalid_metric_scores += 1

            if hop_idx >= 2:
                context_source = hop.get("context_source") or {}
                if not context_source:
                    missing_context_source += 1
                    continue

                selected_idx = context_source.get("selected_candidate_idx")
                selection_hop = context_source.get("selection_hop")
                if selection_hop != hop_idx - 1:
                    bad_prevhop_ref += 1
                    continue
                if not isinstance(selected_idx, int):
                    bad_selected_index += 1
                    continue

                prev_candidates = (hops[selection_hop - 1].get("candidates") or [])
                if not (0 <= selected_idx < len(prev_candidates)):
                    bad_selected_index += 1
                    continue

                score_key = context_source.get("selected_score_key")
                if score_key:
                    vals = [float(c.get(score_key)) for c in prev_candidates if _is_finite_score(c.get(score_key))]
                    if vals:
                        selected_metric_scores.append(float(prev_candidates[selected_idx].get(score_key)))
                        selected_metric_best.append(max(vals))

                g_vals = [
                    float(c.get("gold_ap_cumulative"))
                    for c in prev_candidates
                    if _is_finite_score(c.get("gold_ap_cumulative"))
                ]
                if g_vals:
                    selected_gold_scores.append(float(prev_candidates[selected_idx].get("gold_ap_cumulative")))
                    selected_gold_best.append(max(g_vals))

    total_candidates = sum(cand_count * n for n, cand_count in cand_counter.items())
    total_candidates = max(total_candidates, 1)

    zero_retrieved_rate = zero_retrieved / total_candidates

    print(f"rows={len(rows)}")
    print(f"hop_dist={dict(sorted(hop_counter.items()))}")
    print(f"candidates_per_hop_dist={dict(sorted(cand_counter.items()))}")
    print(f"retrieved_len_dist_top={retr_counter.most_common(6)}")
    print(f"missing_row_core={missing_row_core}")
    print(f"bad_num_hops={bad_num_hops}")
    print(f"bad_variant_count={bad_variant_count}")
    print(f"empty_query={empty_query}")
    print(f"marker_query={marker_query}")
    print(f"narrative_like_query={narrative_like_query}")
    print(f"invalid_gold_scores={invalid_gold_scores}")
    if metric_key is not None:
        print(f"invalid_{metric_key}_scores={invalid_metric_scores}")
    print(f"missing_context_source={missing_context_source}")
    print(f"bad_prevhop_ref={bad_prevhop_ref}")
    print(f"bad_selected_index={bad_selected_index}")
    print(f"zero_retrieved={zero_retrieved} ({zero_retrieved_rate:.6f})")

    if selected_metric_scores:
        mean_selected = sum(selected_metric_scores) / len(selected_metric_scores)
        mean_best = sum(selected_metric_best) / len(selected_metric_best)
        print(f"selected_metric_mean={mean_selected:.4f}")
        print(f"oracle_metric_mean={mean_best:.4f}")
        print(f"metric_gap_to_oracle={mean_best - mean_selected:.4f}")

    if selected_gold_scores:
        mean_selected_gold = sum(selected_gold_scores) / len(selected_gold_scores)
        mean_best_gold = sum(selected_gold_best) / len(selected_gold_best)
        print(f"selected_gold_cum_mean={mean_selected_gold:.4f}")
        print(f"oracle_gold_cum_mean={mean_best_gold:.4f}")
        print(f"gold_gap_to_oracle={mean_best_gold - mean_selected_gold:.4f}")

    fail_reasons = []
    if missing_row_core:
        fail_reasons.append("missing_row_core")
    if bad_num_hops:
        fail_reasons.append("bad_num_hops")
    if bad_variant_count:
        fail_reasons.append("bad_variant_count")
    if empty_query:
        fail_reasons.append("empty_query")
    if marker_query:
        fail_reasons.append("marker_query")
    if invalid_gold_scores:
        fail_reasons.append("invalid_gold_scores")
    if invalid_metric_scores:
        fail_reasons.append("invalid_metric_scores")
    if missing_context_source:
        fail_reasons.append("missing_context_source")
    if bad_prevhop_ref:
        fail_reasons.append("bad_prevhop_ref")
    if bad_selected_index:
        fail_reasons.append("bad_selected_index")
    if zero_retrieved_rate > args.max_zero_retrieved_rate:
        fail_reasons.append("zero_retrieved_rate")

    if fail_reasons:
        print(f"STATUS=FAIL reasons={','.join(fail_reasons)}")
        raise SystemExit(1)

    print("STATUS=PASS")


if __name__ == "__main__":
    main()
