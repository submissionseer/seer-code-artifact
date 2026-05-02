#!/usr/bin/env python3
"""Analyze DPO rollout + pair artifacts for structural and training-signal quality."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from seer.dpo_utils import iter_hop_records
from seer.util_io import read_jsonl


SEARCH_QUERY_MARKER = "[[ ## search_query ## ]]"
COMPLETED_MARKER = "[[ ## completed ## ]]"


def _pct(num: int, den: int) -> float:
    return 0.0 if den == 0 else 100.0 * num / den


def _q(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * p))
    return float(ordered[idx])


def parse_dspy_query(text: str) -> tuple[str | None, str | None]:
    raw = str(text or "").strip()
    if not raw:
        return None, "empty"
    if SEARCH_QUERY_MARKER not in raw or COMPLETED_MARKER not in raw:
        return None, "missing_marker"
    body = raw.split(SEARCH_QUERY_MARKER, 1)[1].split(COMPLETED_MARKER, 1)[0].strip()
    if not body:
        return None, "empty_payload"
    first = [ln.strip() for ln in body.splitlines() if ln.strip()]
    if not first:
        return None, "empty_payload"
    return first[0], None


def metric_key_for_mode(mode: str, hop_num: int) -> str | None:
    if mode == "gold":
        return "gold_ap" if hop_num == 1 else "gold_ap_cumulative"
    if mode == "mmr":
        return "mmr_ap"
    if mode == "raw_jina_maxmean":
        return "raw_jina_maxmean_top3"
    if mode == "raw_jina_max":
        return "raw_jina_max"
    if mode == "seer":
        return "seer_ap"
    if mode == "decomp_binary":
        return "decomp_binary_ap"
    return None


def metric_value_for_candidate(candidate: dict[str, Any], mode: str, hop_num: int) -> float | None:
    key = metric_key_for_mode(mode, hop_num)
    if key is None:
        return None
    if key == "decomp_binary_ap":
        value = candidate.get("decomp_binary_ap")
        if value is None:
            value = candidate.get("decomposed_binary_ap")
        return value
    return candidate.get(key)


def selected_gold_from_rollout(row: dict[str, Any], hop_num: int) -> float | None:
    if hop_num <= 1:
        return None
    hops = row.get("hops") or []
    if hop_num > len(hops):
        return None
    hop = hops[hop_num - 1]
    source = hop.get("context_source") or {}
    idx = source.get("selected_candidate_idx")
    sel_hop = source.get("selection_hop")
    if not isinstance(idx, int) or not isinstance(sel_hop, int):
        return None
    if not (1 <= sel_hop <= len(hops)):
        return None
    prev_cands = (hops[sel_hop - 1].get("candidates") or [])
    if not (0 <= idx < len(prev_cands)):
        return None
    v = prev_cands[idx].get("gold_ap_cumulative")
    try:
        return float(v)
    except Exception:
        return None


def load_rollout_selected_gold(path: Path) -> dict[tuple[str, int], float]:
    mapping: dict[tuple[str, int], float] = {}
    for row in read_jsonl(path):
        qid = row.get("qid")
        if not qid:
            continue
        for hop_num, _ in iter_hop_records(row):
            v = selected_gold_from_rollout(row, hop_num)
            if v is not None:
                mapping[(str(qid), int(hop_num))] = v
    return mapping


def analyze_rollout(path: Path, mode: str) -> dict[str, Any]:
    rows = read_jsonl(path)
    hop_dist: Counter[int] = Counter()
    cand_dist: Counter[int] = Counter()
    retr_dist: Counter[int] = Counter()
    selected_metric = []
    oracle_metric = []
    selected_gold = []
    oracle_gold = []
    qid_hops = set()

    for row in rows:
        hops = row.get("hops") or []
        hop_dist[len(hops)] += 1
        for hop_num, hop in iter_hop_records(row):
            qid = row.get("qid")
            if qid:
                qid_hops.add((str(qid), int(hop_num)))

            cands = hop.get("candidates") or []
            cand_dist[len(cands)] += 1
            mk = metric_key_for_mode(mode, hop_num)
            vals_metric = []
            vals_gold = []
            for c in cands:
                retrieved = c.get("retrieved") or []
                retr_dist[len(retrieved)] += 1
                if mk is not None:
                    try:
                        val = metric_value_for_candidate(c, mode, hop_num)
                        vals_metric.append(float(val))
                    except Exception:
                        pass
                try:
                    gk = "gold_ap" if hop_num == 1 else "gold_ap_cumulative"
                    vals_gold.append(float(c.get(gk)))
                except Exception:
                    pass

            if hop_num >= 2:
                source = hop.get("context_source") or {}
                idx = source.get("selected_candidate_idx")
                sel_hop = source.get("selection_hop")
                if isinstance(idx, int) and isinstance(sel_hop, int) and sel_hop == hop_num - 1:
                    prev = (hops[sel_hop - 1].get("candidates") or [])
                    if 0 <= idx < len(prev):
                        if mk is not None:
                            try:
                                selected_metric.append(float(metric_value_for_candidate(prev[idx], mode, sel_hop)))
                            except Exception:
                                pass
                        try:
                            selected_gold.append(float(prev[idx].get("gold_ap_cumulative")))
                        except Exception:
                            pass

                        if mk is not None:
                            vals = []
                            for c in prev:
                                try:
                                    vals.append(float(metric_value_for_candidate(c, mode, sel_hop)))
                                except Exception:
                                    pass
                            if vals:
                                oracle_metric.append(max(vals))
                        gvals = []
                        for c in prev:
                            try:
                                gvals.append(float(c.get("gold_ap_cumulative")))
                            except Exception:
                                pass
                        if gvals:
                            oracle_gold.append(max(gvals))

    return {
        "rows": len(rows),
        "hop_dist": dict(sorted(hop_dist.items())),
        "candidates_per_hop_dist": dict(sorted(cand_dist.items())),
        "retrieved_len_dist_top": retr_dist.most_common(6),
        "selected_metric_mean": statistics.fmean(selected_metric) if selected_metric else 0.0,
        "oracle_metric_mean": statistics.fmean(oracle_metric) if oracle_metric else 0.0,
        "selected_gold_mean": statistics.fmean(selected_gold) if selected_gold else 0.0,
        "oracle_gold_mean": statistics.fmean(oracle_gold) if oracle_gold else 0.0,
        "qid_hops": qid_hops,
    }


def analyze_pairs(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    missing_core = 0
    bad_meta = 0
    bad_marker = 0
    empty_query = 0
    identical_queries = 0
    nonpositive_gap = 0

    hop_dist: Counter[int] = Counter()
    scoring_dist: Counter[str] = Counter()
    pairs_per_qid: defaultdict[str, int] = defaultdict(int)
    pairs_per_qid_hop: defaultdict[tuple[str, int], int] = defaultdict(int)
    gaps = []
    chosen_scores = []
    rejected_scores = []
    qid_hops = set()

    for row in rows:
        if any(k not in row for k in ("prompt", "chosen", "rejected", "meta")):
            missing_core += 1
            continue
        meta = row.get("meta") or {}
        qid = meta.get("qid")
        hop = meta.get("hop")
        if not qid or not isinstance(hop, int):
            bad_meta += 1
            continue

        qid = str(qid)
        hop_dist[int(hop)] += 1
        scoring_dist[str(meta.get("scoring", "unknown"))] += 1
        pairs_per_qid[qid] += 1
        pairs_per_qid_hop[(qid, int(hop))] += 1
        qid_hops.add((qid, int(hop)))

        c_q, c_err = parse_dspy_query(str(row.get("chosen", "")))
        r_q, r_err = parse_dspy_query(str(row.get("rejected", "")))
        if c_err or r_err:
            bad_marker += 1
            if c_err == "empty" or r_err == "empty":
                empty_query += 1
            continue
        if (c_q or "").strip() == (r_q or "").strip():
            identical_queries += 1

        try:
            cs = float(meta.get("chosen_score"))
            rs = float(meta.get("rejected_score"))
            chosen_scores.append(cs)
            rejected_scores.append(rs)
            gap = cs - rs
            gaps.append(gap)
            if gap <= 0:
                nonpositive_gap += 1
        except Exception:
            bad_meta += 1

    pairs_per_qid_vals = list(pairs_per_qid.values())
    return {
        "rows": len(rows),
        "missing_core": missing_core,
        "bad_meta": bad_meta,
        "bad_marker": bad_marker,
        "empty_query": empty_query,
        "identical_queries": identical_queries,
        "nonpositive_gap": nonpositive_gap,
        "hop_dist": dict(sorted(hop_dist.items())),
        "scoring_dist": dict(scoring_dist),
        "unique_qids": len(pairs_per_qid),
        "unique_qid_hops": len(qid_hops),
        "pairs_per_qid_min": min(pairs_per_qid_vals) if pairs_per_qid_vals else 0,
        "pairs_per_qid_median": statistics.median(pairs_per_qid_vals) if pairs_per_qid_vals else 0,
        "pairs_per_qid_p95": _q([float(v) for v in pairs_per_qid_vals], 0.95),
        "pairs_per_qid_max": max(pairs_per_qid_vals) if pairs_per_qid_vals else 0,
        "chosen_score_mean": statistics.fmean(chosen_scores) if chosen_scores else 0.0,
        "rejected_score_mean": statistics.fmean(rejected_scores) if rejected_scores else 0.0,
        "gap_mean": statistics.fmean(gaps) if gaps else 0.0,
        "gap_min": min(gaps) if gaps else 0.0,
        "gap_p25": _q(gaps, 0.25),
        "gap_p50": _q(gaps, 0.50),
        "gap_p75": _q(gaps, 0.75),
        "gap_p95": _q(gaps, 0.95),
        "gap_max": max(gaps) if gaps else 0.0,
        "qid_hops": qid_hops,
    }


def print_summary(name: str, data: dict[str, Any]) -> None:
    print(f"\n== {name} ==")
    for k, v in data.items():
        if k == "qid_hops":
            continue
        print(f"{k}={v}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze DPO rollout/pair data quality")
    parser.add_argument("--pairs", required=True, help="DPO pairs JSONL")
    parser.add_argument("--rollout", default=None, help="Rollout JSONL used to create pairs")
    parser.add_argument(
        "--mode",
        default="mmr",
        choices=[
            "gold",
            "mmr",
            "raw_jina_maxmean",
            "raw_jina_max",
            "seer",
            "decomp_binary",
            "uniform",
        ],
        help="Rollout context-selection mode (for rollout stats)",
    )
    parser.add_argument(
        "--compare-rollout",
        default=None,
        help="Optional comparison rollout to compute selected gold deltas by (qid,hop)",
    )
    args = parser.parse_args()

    pair_stats = analyze_pairs(Path(args.pairs))
    print_summary("Pairs", pair_stats)

    fail_reasons = []
    if pair_stats["missing_core"] > 0:
        fail_reasons.append("missing_core")
    if pair_stats["bad_meta"] > 0:
        fail_reasons.append("bad_meta")
    if pair_stats["bad_marker"] > 0:
        fail_reasons.append("bad_marker")
    if pair_stats["identical_queries"] > 0:
        fail_reasons.append("identical_queries")
    if pair_stats["nonpositive_gap"] > 0:
        fail_reasons.append("nonpositive_gap")

    if args.rollout:
        rollout_stats = analyze_rollout(Path(args.rollout), args.mode)
        print_summary("Rollout", rollout_stats)
        covered = len(pair_stats["qid_hops"] & rollout_stats["qid_hops"])
        print(f"pair_qid_hop_coverage={covered}/{len(rollout_stats['qid_hops'])} ({_pct(covered, len(rollout_stats['qid_hops'])):.2f}%)")

        if args.compare_rollout:
            lhs = load_rollout_selected_gold(Path(args.rollout))
            rhs = load_rollout_selected_gold(Path(args.compare_rollout))
            deltas = []
            under = equal = over = 0
            for key in set(lhs.keys()) & set(rhs.keys()):
                d = lhs[key] - rhs[key]
                deltas.append(d)
                if d < 0:
                    under += 1
                elif d > 0:
                    over += 1
                else:
                    equal += 1
            print(
                "selected_gold_delta "
                f"(left-right): n={len(deltas)} under={under} ({_pct(under,len(deltas)):.2f}%) "
                f"equal={equal} ({_pct(equal,len(deltas)):.2f}%) over={over} ({_pct(over,len(deltas)):.2f}%) "
                f"mean={statistics.fmean(deltas) if deltas else 0.0:.4f} "
                f"p25={_q(deltas,0.25):.4f} p50={_q(deltas,0.50):.4f} p75={_q(deltas,0.75):.4f}"
            )

    if fail_reasons:
        print(f"STATUS=FAIL reasons={','.join(fail_reasons)}")
        raise SystemExit(1)
    print("STATUS=PASS")


if __name__ == "__main__":
    main()
