#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _parse_name_path(items: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected NAME=PATH, got: {item}")
        name, path = item.split("=", 1)
        out[name.strip()] = Path(path.strip())
    return out


def _parse_aliases(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected OLD=NEW alias, got: {item}")
        old, new = item.split("=", 1)
        out[old.strip()] = new.strip()
    return out


def _resolve_hops(row: dict[str, Any]) -> int | None:
    num_hops = row.get("num_hops")
    if isinstance(num_hops, int):
        return num_hops
    hop_details = row.get("hop_details", []) or []
    if isinstance(hop_details, list) and hop_details:
        hops = []
        for h in hop_details:
            if isinstance(h, dict):
                try:
                    hops.append(int(h.get("hop", 0) or 0))
                except Exception:
                    pass
        if hops:
            return max(hops)
    retrieved = row.get("retrieved", []) or []
    if isinstance(retrieved, list) and retrieved:
        hops = []
        for p in retrieved:
            if isinstance(p, dict):
                try:
                    hops.append(int(p.get("hop", 1) or 1))
                except Exception:
                    pass
        if hops:
            return max(hops)
    return None


@dataclass
class ArmPoint:
    qid: str
    score: float
    final_ap: float
    num_hops: int | None
    em: float
    f1: float
    unans: float


def _load_scored(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        qid = str(row.get("qid", "")).strip()
        if not qid:
            continue
        score = row.get("mmr_ap_final", row.get("final_ap", 0.0))
        try:
            score_f = float(score)
        except Exception:
            score_f = 0.0
        try:
            final_ap = float(row.get("final_ap", np.nan))
        except Exception:
            final_ap = float("nan")
        out[qid] = {
            "score": score_f,
            "final_ap": final_ap,
            "num_hops": _resolve_hops(row),
        }
    return out


def _load_reader_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        qid = str(row.get("qid", "")).strip()
        if not qid:
            continue
        pred = str(row.get("predicted_answer", row.get("pred", "")))
        is_unans = bool(row.get("is_unanswerable", pred.strip().upper() == "UNANSWERABLE"))
        out[qid] = {
            "em": float(row.get("em", 0.0)),
            "f1": float(row.get("f1", 0.0)),
            "is_unanswerable": is_unans,
            "pred": pred,
        }
    return out


def _load_reader_compare(path: Path, aliases: dict[str, str]) -> dict[str, dict[str, dict[str, Any]]]:
    payload = json.loads(path.read_text())
    meta = payload.get("meta", {})
    name_a = aliases.get(str(meta.get("name_a", "arm_a")), str(meta.get("name_a", "arm_a")))
    name_b = aliases.get(str(meta.get("name_b", "arm_b")), str(meta.get("name_b", "arm_b")))

    out: dict[str, dict[str, dict[str, Any]]] = {name_a: {}, name_b: {}}
    for row in payload.get("paired_examples", []) or []:
        if not isinstance(row, dict):
            continue
        qid = str(row.get("qid", "")).strip()
        if not qid:
            continue
        pred_a = str(row.get("pred_a", ""))
        pred_b = str(row.get("pred_b", ""))
        out[name_a][qid] = {
            "em": float(row.get("em_a", 0.0)),
            "f1": float(row.get("f1_a", 0.0)),
            "is_unanswerable": pred_a.strip().upper() == "UNANSWERABLE",
            "pred": pred_a,
        }
        out[name_b][qid] = {
            "em": float(row.get("em_b", 0.0)),
            "f1": float(row.get("f1_b", 0.0)),
            "is_unanswerable": pred_b.strip().upper() == "UNANSWERABLE",
            "pred": pred_b,
        }
    return out


def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        rank = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = rank
        i = j + 1
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    x = x.astype(float)
    y = y.astype(float)
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    return _pearson(_rankdata(x), _rankdata(y))


def _roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(int)
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata(scores.astype(float))
    sum_pos = float(ranks[labels == 1].sum())
    return (sum_pos - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)


def _metric_summary(points: list[ArmPoint], threshold_percents: list[int]) -> dict[str, Any]:
    score = np.array([p.score for p in points], dtype=float)
    final_ap = np.array([p.final_ap for p in points], dtype=float)
    em = np.array([p.em for p in points], dtype=float)
    f1 = np.array([p.f1 for p in points], dtype=float)
    unans = np.array([p.unans for p in points], dtype=float)
    em_fail = 1.0 - em
    hops = np.array([p.num_hops if p.num_hops is not None else -1 for p in points], dtype=int)

    n = int(len(points))
    out: dict[str, Any] = {
        "n": n,
        "score_mean": float(np.mean(score)),
        "score_median": float(np.median(score)),
        "score_p10": float(np.quantile(score, 0.10)),
        "score_p90": float(np.quantile(score, 0.90)),
        "em": float(np.mean(em)),
        "f1": float(np.mean(f1)),
        "unanswerable_rate": float(np.mean(unans)),
        "em_fail_rate": float(np.mean(em_fail)),
        "corr_score_final_ap_pearson": _pearson(score, final_ap),
        "corr_score_final_ap_spearman": _spearman(score, final_ap),
        "corr_score_em_pearson": _pearson(score, em),
        "corr_score_f1_pearson": _pearson(score, f1),
        "corr_score_unanswerable_pearson": _pearson(score, unans),
        "auc_em_fail_from_low_score": _roc_auc(-score, em_fail),
        "auc_unanswerable_from_low_score": _roc_auc(-score, unans),
    }

    k = max(1, int(0.2 * n))
    order = np.argsort(score)
    low_idx = order[:k]
    high_idx = order[-k:]
    out["bottom20_top20"] = {
        "k": k,
        "bottom20_em": float(np.mean(em[low_idx])),
        "top20_em": float(np.mean(em[high_idx])),
        "bottom20_f1": float(np.mean(f1[low_idx])),
        "top20_f1": float(np.mean(f1[high_idx])),
        "bottom20_unans": float(np.mean(unans[low_idx])),
        "top20_unans": float(np.mean(unans[high_idx])),
        "bottom20_em_fail": float(np.mean(em_fail[low_idx])),
        "top20_em_fail": float(np.mean(em_fail[high_idx])),
    }

    threshold_metrics: dict[str, Any] = {}
    for p in threshold_percents:
        q = p / 100.0
        thr = float(np.quantile(score, q))
        low = score <= thr
        high = ~low
        if not np.any(high):
            high = score >= thr
            low = ~high
        low_em_fail = float(np.mean(em_fail[low])) if np.any(low) else float("nan")
        high_em_fail = float(np.mean(em_fail[high])) if np.any(high) else float("nan")
        low_unans = float(np.mean(unans[low])) if np.any(low) else float("nan")
        high_unans = float(np.mean(unans[high])) if np.any(high) else float("nan")
        threshold_metrics[str(p)] = {
            "threshold": thr,
            "n_low": int(np.sum(low)),
            "n_high": int(np.sum(high)),
            "low_em_fail_rate": low_em_fail,
            "high_em_fail_rate": high_em_fail,
            "low_unans_rate": low_unans,
            "high_unans_rate": high_unans,
            "low_em": float(np.mean(em[low])) if np.any(low) else float("nan"),
            "high_em": float(np.mean(em[high])) if np.any(high) else float("nan"),
            "lift_em_fail_low_over_high": low_em_fail / high_em_fail if high_em_fail > 0 else float("nan"),
            "lift_unans_low_over_high": low_unans / high_unans if high_unans > 0 else float("nan"),
        }
    out["thresholds"] = threshold_metrics

    by_hop: dict[str, Any] = {}
    for hop in sorted(h for h in set(hops.tolist()) if h >= 0):
        mask = hops == hop
        if int(np.sum(mask)) < 2:
            continue
        by_hop[str(hop)] = {
            "n": int(np.sum(mask)),
            "em": float(np.mean(em[mask])),
            "f1": float(np.mean(f1[mask])),
            "unanswerable_rate": float(np.mean(unans[mask])),
            "score_mean": float(np.mean(score[mask])),
            "corr_score_em_pearson": _pearson(score[mask], em[mask]),
            "corr_score_final_ap_pearson": _pearson(score[mask], final_ap[mask]),
        }
    out["by_hop"] = by_hop
    return out


def _bootstrap_ci(
    points: list[ArmPoint],
    samples: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    n = len(points)
    if n == 0:
        return {}
    rng = np.random.default_rng(seed)
    score = np.array([p.score for p in points], dtype=float)
    final_ap = np.array([p.final_ap for p in points], dtype=float)
    em = np.array([p.em for p in points], dtype=float)
    unans = np.array([p.unans for p in points], dtype=float)
    em_fail = 1.0 - em

    out = {
        "auc_em_fail_from_low_score": [],
        "auc_unanswerable_from_low_score": [],
        "corr_score_em_pearson": [],
        "corr_score_final_ap_pearson": [],
    }
    for _ in range(samples):
        idx = rng.integers(0, n, size=n)
        out["auc_em_fail_from_low_score"].append(_roc_auc(-score[idx], em_fail[idx]))
        out["auc_unanswerable_from_low_score"].append(_roc_auc(-score[idx], unans[idx]))
        out["corr_score_em_pearson"].append(_pearson(score[idx], em[idx]))
        out["corr_score_final_ap_pearson"].append(_pearson(score[idx], final_ap[idx]))

    ci: dict[str, dict[str, float]] = {}
    for k, vals in out.items():
        arr = np.array([v for v in vals if np.isfinite(v)], dtype=float)
        if len(arr) == 0:
            ci[k] = {"lo": float("nan"), "hi": float("nan")}
            continue
        ci[k] = {
            "lo": float(np.quantile(arr, 0.025)),
            "hi": float(np.quantile(arr, 0.975)),
        }
    return ci


def _pairwise(
    arm_points: dict[str, dict[str, ArmPoint]],
    samples: int,
    seed: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    arm_names = list(arm_points.keys())
    for a, b in combinations(arm_names, 2):
        common = sorted(set(arm_points[a].keys()) & set(arm_points[b].keys()))
        if not common:
            continue
        score_a = np.array([arm_points[a][q].score for q in common], dtype=float)
        score_b = np.array([arm_points[b][q].score for q in common], dtype=float)
        em_a = np.array([arm_points[a][q].em for q in common], dtype=float)
        em_b = np.array([arm_points[b][q].em for q in common], dtype=float)
        f1_a = np.array([arm_points[a][q].f1 for q in common], dtype=float)
        f1_b = np.array([arm_points[b][q].f1 for q in common], dtype=float)
        un_a = np.array([arm_points[a][q].unans for q in common], dtype=float)
        un_b = np.array([arm_points[b][q].unans for q in common], dtype=float)

        d_score = score_b - score_a
        d_em = em_b - em_a
        d_f1 = f1_b - f1_a
        d_un = un_b - un_a
        d_em_fail = (1.0 - em_b) - (1.0 - em_a)

        wins_em = {
            f"{b}_better": int(np.sum(d_em > 0)),
            f"{a}_better": int(np.sum(d_em < 0)),
            "equal": int(np.sum(d_em == 0)),
        }
        wins_f1 = {
            f"{b}_better": int(np.sum(d_f1 > 0)),
            f"{a}_better": int(np.sum(d_f1 < 0)),
            "equal": int(np.sum(d_f1 == 0)),
        }

        key = f"{b}_minus_{a}"
        pair = {
            "n": len(common),
            "mean_d_score": float(np.mean(d_score)),
            "mean_d_em": float(np.mean(d_em)),
            "mean_d_f1": float(np.mean(d_f1)),
            "mean_d_unans": float(np.mean(d_un)),
            "mean_d_em_fail": float(np.mean(d_em_fail)),
            "corr_d_score_vs_d_em_pearson": _pearson(d_score, d_em),
            "corr_d_score_vs_d_f1_pearson": _pearson(d_score, d_f1),
            "em_wins": wins_em,
            "f1_wins": wins_f1,
        }

        rng = np.random.default_rng(seed)
        n = len(common)
        bs = {
            "mean_d_em": [],
            "mean_d_f1": [],
            "mean_d_unans": [],
            "mean_d_score": [],
        }
        for _ in range(samples):
            idx = rng.integers(0, n, size=n)
            bs["mean_d_em"].append(float(np.mean(d_em[idx])))
            bs["mean_d_f1"].append(float(np.mean(d_f1[idx])))
            bs["mean_d_unans"].append(float(np.mean(d_un[idx])))
            bs["mean_d_score"].append(float(np.mean(d_score[idx])))
        pair["bootstrap_95"] = {
            name: {
                "lo": float(np.quantile(vals, 0.025)),
                "hi": float(np.quantile(vals, 0.975)),
            }
            for name, vals in bs.items()
        }
        out[key] = pair
    return out


def _build_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    meta = report["meta"]
    lines.append(f"# Monitoring report: {meta['dataset']}")
    lines.append("")
    lines.append(f"- `n_common_qids`: {meta['n_common_qids']}")
    lines.append(f"- `bootstrap_samples`: {meta['bootstrap_samples']}")
    lines.append("")
    lines.append("## Arms")
    lines.append("")
    lines.append("| arm | n | score_mean | EM | F1 | unans | AUC(em_fail|low_score) | AUC(unans|low_score) | corr(score,EM) | corr(score,final_ap) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for arm, m in report["arms"].items():
        lines.append(
            f"| {arm} | {m['n']} | {m['score_mean']:.4f} | {m['em']:.4f} | {m['f1']:.4f} | {m['unanswerable_rate']:.4f} | "
            f"{m['auc_em_fail_from_low_score']:.4f} | {m['auc_unanswerable_from_low_score']:.4f} | "
            f"{m['corr_score_em_pearson']:.4f} | {m['corr_score_final_ap_pearson']:.4f} |"
        )
    lines.append("")

    if report.get("pairwise"):
        lines.append("## Pairwise deltas")
        lines.append("")
        lines.append("| pair (`b_minus_a`) | n | dEM | dF1 | dUnans | dScore |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for name, p in report["pairwise"].items():
            lines.append(
                f"| {name} | {p['n']} | {p['mean_d_em']:.4f} | {p['mean_d_f1']:.4f} | "
                f"{p['mean_d_unans']:.4f} | {p['mean_d_score']:.4f} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Monitoring-signal report from scored rollouts + reader EM outputs.")
    ap.add_argument("--dataset", required=True, help="Dataset label for report metadata.")
    ap.add_argument("--arm", action="append", default=[], help="Scored rollout arm as NAME=PATH (repeat).")
    ap.add_argument("--reader-jsonl", action="append", default=[], help="Reader result JSONL as NAME=PATH (repeat).")
    ap.add_argument("--reader-compare", action="append", default=[], help="Reader paired-compare JSON path (repeat).")
    ap.add_argument("--alias", action="append", default=[], help="Alias mapping OLD=NEW for reader-compare names.")
    ap.add_argument("--bootstrap-samples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threshold-percents", default="10,20,30")
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-md", default=None)
    args = ap.parse_args()

    aliases = _parse_aliases(args.alias)
    arm_paths = _parse_name_path(args.arm)
    reader_jsonl_paths = _parse_name_path(args.reader_jsonl)
    threshold_percents = [int(x.strip()) for x in args.threshold_percents.split(",") if x.strip()]

    reader_by_arm: dict[str, dict[str, dict[str, Any]]] = {}
    for arm, path in reader_jsonl_paths.items():
        reader_by_arm[arm] = _load_reader_jsonl(path)
    for cmp_path in args.reader_compare:
        cmp_payload = _load_reader_compare(Path(cmp_path), aliases=aliases)
        for arm, qmap in cmp_payload.items():
            existing = reader_by_arm.setdefault(arm, {})
            existing.update(qmap)

    scored_by_arm: dict[str, dict[str, dict[str, Any]]] = {
        arm: _load_scored(path) for arm, path in arm_paths.items()
    }

    arm_points_by_name: dict[str, dict[str, ArmPoint]] = {}
    arms_summary: dict[str, Any] = {}
    n_common_qids_global = None
    for arm_name, scored_map in scored_by_arm.items():
        reader_map = reader_by_arm.get(arm_name)
        if not reader_map:
            raise RuntimeError(f"No reader metrics found for arm: {arm_name}")
        common = sorted(set(scored_map.keys()) & set(reader_map.keys()))
        if not common:
            raise RuntimeError(f"No shared qids for arm: {arm_name}")
        points: list[ArmPoint] = []
        for q in common:
            s = scored_map[q]
            r = reader_map[q]
            points.append(
                ArmPoint(
                    qid=q,
                    score=float(s["score"]),
                    final_ap=float(s["final_ap"]),
                    num_hops=s["num_hops"],
                    em=float(r.get("em", 0.0)),
                    f1=float(r.get("f1", 0.0)),
                    unans=1.0 if bool(r.get("is_unanswerable", False)) else 0.0,
                )
            )
        arm_points_by_name[arm_name] = {p.qid: p for p in points}
        summary = _metric_summary(points, threshold_percents=threshold_percents)
        summary["bootstrap_95"] = _bootstrap_ci(
            points,
            samples=args.bootstrap_samples,
            seed=args.seed,
        )
        arms_summary[arm_name] = summary
        if n_common_qids_global is None:
            n_common_qids_global = len(common)
        else:
            n_common_qids_global = min(n_common_qids_global, len(common))

    report = {
        "meta": {
            "dataset": args.dataset,
            "n_common_qids": int(n_common_qids_global or 0),
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
            "inputs": {
                "arms": {k: str(v) for k, v in arm_paths.items()},
                "reader_jsonl": {k: str(v) for k, v in reader_jsonl_paths.items()},
                "reader_compare": [str(Path(p)) for p in args.reader_compare],
                "aliases": aliases,
            },
        },
        "arms": arms_summary,
        "pairwise": _pairwise(
            arm_points_by_name,
            samples=args.bootstrap_samples,
            seed=args.seed,
        ),
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2))

    out_md = Path(args.output_md) if args.output_md else out_json.with_suffix(".md")
    out_md.write_text(_build_markdown(report))

    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
