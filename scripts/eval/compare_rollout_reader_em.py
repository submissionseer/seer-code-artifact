#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import yaml

from seer.datasets import get_dataset_adapter, list_dataset_adapters
from seer.openrouter import OpenRouterConfig
from seer.reader_eval import run_reader_batch_sync
from seer.stats import bootstrap_ci


_SPLIT_ALIASES = {
    "dev": "validation",
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
    "train": "train",
    "test": "test",
}


@dataclass
class AnswerJoinInfo:
    dataset: str
    split: str | None
    source_path: str
    loaded_rows: int
    answers_by_qid: dict[str, str]


def _load_api_key() -> str | None:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if api_key:
        return api_key
    for env_path in (Path(".env"), Path("../.env")):
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("OPENROUTER_API_KEY="):
                    return line.split("=", 1)[1].strip()
    return None


def _canonical_split_name(value: str | None) -> str:
    normalized = str(value or "").strip().lower()
    return _SPLIT_ALIASES.get(normalized, normalized)


def _load_jsonl_map(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            qid = r.get("qid")
            if qid:
                out[qid] = r
    return out


def _load_answers_from_raw_jsonl(raw_path: Path, dataset: str, split: str | None) -> AnswerJoinInfo:
    adapter = get_dataset_adapter(dataset)
    target_split = _canonical_split_name(split) if split else None

    answers_by_qid: dict[str, str] = {}
    loaded_rows = 0
    with raw_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            row_split = _canonical_split_name(item.get("split"))
            if target_split and row_split != target_split:
                continue
            normalized = adapter.normalize_raw(item, split=item.get("split"))
            if not normalized.qid:
                continue
            answers_by_qid[normalized.qid] = normalized.answer
            loaded_rows += 1

    return AnswerJoinInfo(
        dataset=adapter.name,
        split=target_split,
        source_path=str(raw_path),
        loaded_rows=loaded_rows,
        answers_by_qid=answers_by_qid,
    )


def _load_hotpot_fullwiki_answers(raw_dir: Path, split: str | None) -> AnswerJoinInfo:
    target_split = _canonical_split_name(split) if split else "validation"
    if target_split == "test":
        path = raw_dir / "hotpot_fullwiki" / "hotpot_test_fullwiki_v1.json"
    else:
        path = raw_dir / "hotpot_fullwiki" / "hotpot_dev_fullwiki_v1.json"
        target_split = "validation"

    rows = json.loads(path.read_text())
    answers_by_qid: dict[str, str] = {}
    for row in rows:
        qid = str(row.get("_id", row.get("id", row.get("qid", ""))))
        if not qid:
            continue
        answers_by_qid[qid] = str(row.get("answer", ""))

    return AnswerJoinInfo(
        dataset="hotpot",
        split=target_split,
        source_path=str(path),
        loaded_rows=len(answers_by_qid),
        answers_by_qid=answers_by_qid,
    )


def _load_answers_by_qid(raw_dir: Path, dataset: str, split: str | None) -> AnswerJoinInfo:
    adapter = get_dataset_adapter(dataset)
    raw_jsonl = raw_dir / adapter.name / "raw.jsonl"
    if raw_jsonl.exists():
        return _load_answers_from_raw_jsonl(raw_jsonl, adapter.name, split)

    if adapter.name == "hotpot":
        return _load_hotpot_fullwiki_answers(raw_dir, split)

    raise FileNotFoundError(f"Could not locate answer source for dataset={adapter.name} under {raw_dir}")


def _reader_context(record: dict, top_k_per_hop: int) -> list[dict]:
    retrieved = record.get("retrieved", [])
    by_hop: dict[int, list[dict]] = {}
    for passage in retrieved:
        hop = int(passage.get("hop", 1))
        by_hop.setdefault(hop, []).append(passage)
    context: list[dict] = []
    for hop in sorted(by_hop):
        hop_passages = sorted(by_hop[hop], key=lambda p: p.get("score", 0), reverse=True)
        context.extend(hop_passages[:top_k_per_hop])
    return context


def _build_reader_records(
    records: list[dict],
    answers_by_qid: dict[str, str],
    top_k_per_hop: int,
) -> tuple[list[dict], int]:
    payloads = []
    missing_answers = 0
    for record in records:
        qid = record.get("qid", "")
        answer = answers_by_qid.get(qid, "")
        if not answer:
            missing_answers += 1
        payloads.append(
            {
                "qid": qid,
                "question": record.get("question", ""),
                "context": _reader_context(record, top_k_per_hop),
                "gold": {"answer": answer},
            }
        )
    return payloads, missing_answers


def _build_no_context_reader_records(
    records: list[dict],
    answers_by_qid: dict[str, str],
) -> tuple[list[dict], int]:
    payloads = []
    missing_answers = 0
    for record in records:
        qid = record.get("qid", "")
        answer = answers_by_qid.get(qid, "")
        if not answer:
            missing_answers += 1
        payloads.append(
            {
                "qid": qid,
                "question": record.get("question", ""),
                "context": [],
                "gold": {"answer": answer},
            }
        )
    return payloads, missing_answers


def _summarize(results: list[dict]) -> dict:
    if not results:
        return {"n": 0, "em": 0.0, "f1": 0.0}
    return {
        "n": len(results),
        "em": mean(r.get("em", 0.0) for r in results),
        "f1": mean(r.get("f1", 0.0) for r in results),
        "unanswerable_rate": mean(1.0 if r.get("is_unanswerable") else 0.0 for r in results),
    }


def _paired_delta_summary(
    paired: list[dict],
    *,
    delta_em_key: str,
    delta_f1_key: str,
    num_samples: int,
    seed: int,
) -> dict:
    em_values = [float(x.get(delta_em_key, 0.0)) for x in paired]
    f1_values = [float(x.get(delta_f1_key, 0.0)) for x in paired]
    em_lower, em_upper = bootstrap_ci(em_values, num_samples=num_samples, seed=seed)
    f1_lower, f1_upper = bootstrap_ci(f1_values, num_samples=num_samples, seed=seed)
    return {
        "n": len(paired),
        "mean_delta_em": mean(em_values) if em_values else 0.0,
        "mean_delta_f1": mean(f1_values) if f1_values else 0.0,
        "em_ci_95": [em_lower, em_upper],
        "f1_ci_95": [f1_lower, f1_upper],
        "em_ci_excludes_zero": bool(em_lower > 0 or em_upper < 0),
        "f1_ci_excludes_zero": bool(f1_lower > 0 or f1_upper < 0),
    }


def _load_checkpoint_results(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = row.get("qid")
            if qid:
                out[qid] = row
    return out


def _append_checkpoint_results(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _write_json_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def _run_reader_in_batches(
    config_obj: OpenRouterConfig,
    cache_dir: str,
    model: str,
    records: list[dict],
    batch_size: int,
    label: str,
    checkpoint_results_path: Path | None = None,
    checkpoint_progress_path: Path | None = None,
    resume_from_checkpoints: bool = True,
    checkpoint_every_batches: int = 1,
) -> list[dict]:
    results_by_qid: dict[str, dict] = {}
    if checkpoint_results_path and resume_from_checkpoints:
        results_by_qid = _load_checkpoint_results(checkpoint_results_path)
        resumed = sum(1 for r in records if r.get("qid") in results_by_qid)
        if resumed:
            print(f"{label}: resumed {resumed}/{len(records)} from {checkpoint_results_path}", flush=True)
    pending_records = [r for r in records if r.get("qid") not in results_by_qid]
    completed = len(records) - len(pending_records)
    total = len(records)
    if checkpoint_progress_path:
        _write_json_checkpoint(
            checkpoint_progress_path,
            {"label": label, "completed": completed, "total": total, "done": completed >= total},
        )
    batches_since_flush = 0
    staged_results: list[dict] = []
    for start in range(0, len(pending_records), batch_size):
        batch = pending_records[start : start + batch_size]
        print(f"{label}: reader {completed + len(batch)}/{total}", flush=True)
        batch_results = run_reader_batch_sync(
            config_obj,
            cache_dir=cache_dir,
            model=model,
            records=batch,
        )
        for row in batch_results:
            qid = row.get("qid")
            if qid:
                results_by_qid[qid] = row
        staged_results.extend(batch_results)
        completed += len(batch)
        batches_since_flush += 1
        if checkpoint_results_path and batches_since_flush >= checkpoint_every_batches:
            _append_checkpoint_results(checkpoint_results_path, staged_results)
            staged_results = []
            batches_since_flush = 0
        if checkpoint_progress_path and batches_since_flush == 0:
            _write_json_checkpoint(
                checkpoint_progress_path,
                {"label": label, "completed": completed, "total": total, "done": completed >= total},
            )
    if checkpoint_results_path and staged_results:
        _append_checkpoint_results(checkpoint_results_path, staged_results)
    if checkpoint_progress_path:
        _write_json_checkpoint(
            checkpoint_progress_path,
            {"label": label, "completed": completed, "total": total, "done": completed >= total},
        )
    return [results_by_qid[r["qid"]] for r in records if r.get("qid") in results_by_qid]


def _enforce_missing_answer_guard(
    *,
    arm_name: str,
    missing_answers: int,
    total: int,
    max_missing_answers: int,
    dataset: str,
    split: str | None,
    answer_source: str,
) -> None:
    if missing_answers:
        print(
            f"Warning: {arm_name} has {missing_answers}/{total} records missing gold answers "
            f"(dataset={dataset}, split={split or 'all'})"
        )
    if missing_answers > max_missing_answers:
        raise SystemExit(
            f"{arm_name}: missing answers {missing_answers} exceeds --max-missing-answers={max_missing_answers}. "
            f"Check dataset/split and answer source ({answer_source})."
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare two rollout files via frozen-reader EM/F1 on matched qids.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--dataset", default="hotpot", help=f"Dataset for answer join (aliases supported): {', '.join(list_dataset_adapters())}")
    ap.add_argument("--split", default="validation", help="Dataset split for answer join (e.g. validation/train/test)")
    ap.add_argument("--rollout-a", required=True)
    ap.add_argument("--rollout-b", required=True)
    ap.add_argument("--name-a", default="A")
    ap.add_argument("--name-b", default="B")
    ap.add_argument("--sample-size", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top-k-per-hop", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=25)
    ap.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help="Number of paired bootstrap resamples used for EM/F1 delta confidence intervals.",
    )
    ap.add_argument("--reader-model", default=None, help="Override reader model from config")
    ap.add_argument(
        "--qid-order-file",
        default=None,
        help="Optional JSON file to persist/reuse a deterministic qid order. Sample is always a prefix of this order.",
    )
    ap.add_argument(
        "--include-no-context-baseline",
        action="store_true",
        help="Also run a no-context reader baseline on the same sampled qids.",
    )
    ap.add_argument("--name-baseline", default="no_context")
    ap.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory for resumable per-arm checkpoints. Defaults to <output>.checkpoints/",
    )
    ap.add_argument(
        "--checkpoint-every-batches",
        type=int,
        default=1,
        help="Flush checkpoint files every N reader batches.",
    )
    ap.add_argument(
        "--no-resume-checkpoints",
        action="store_true",
        help="Ignore existing checkpoint files and recompute all arms.",
    )
    ap.add_argument(
        "--cleanup-checkpoints-on-success",
        action="store_true",
        help="Delete checkpoint files after writing final output successfully.",
    )
    ap.add_argument(
        "--max-missing-answers",
        type=int,
        default=0,
        help="Maximum allowed records with missing gold answers per arm.",
    )
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    cache_dir = cfg.get("cache_dir", "data/cached_judges")
    reader_model = args.reader_model or cfg.get("reader", {}).get("model", "google/gemini-2.5-flash")
    raw_dir = Path(cfg.get("raw_dir", "data/raw"))
    dataset_name = get_dataset_adapter(args.dataset).name
    split_name = _canonical_split_name(args.split)

    api_key = _load_api_key()
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not found in env or .env")

    a_map = _load_jsonl_map(Path(args.rollout_a))
    b_map = _load_jsonl_map(Path(args.rollout_b))
    common = sorted(set(a_map) & set(b_map))
    if not common:
        raise SystemExit("No overlapping qids between rollout files")

    qid_order_path = Path(args.qid_order_file) if args.qid_order_file else None
    if qid_order_path and qid_order_path.exists():
        qid_order_payload = json.loads(qid_order_path.read_text())
        if isinstance(qid_order_payload, dict):
            ordered_qids = qid_order_payload.get("qid_order", [])
        else:
            ordered_qids = qid_order_payload
        common_set = set(common)
        common = [qid for qid in ordered_qids if qid in common_set]
        if len(common) < min(args.sample_size, len(common_set)):
            missing = min(args.sample_size, len(common_set)) - len(common)
            raise SystemExit(f"qid-order file exists but does not contain enough overlapping qids (missing at least {missing})")
    else:
        rnd = random.Random(args.seed)
        rnd.shuffle(common)
        if qid_order_path:
            qid_order_path.parent.mkdir(parents=True, exist_ok=True)
            qid_order_path.write_text(
                json.dumps(
                    {
                        "seed": args.seed,
                        "rollout_a": args.rollout_a,
                        "rollout_b": args.rollout_b,
                        "qid_order": common,
                    },
                    indent=2,
                )
            )
            print(f"Saved qid order: {qid_order_path}")
    qids = common[: min(args.sample_size, len(common))]

    answer_join = _load_answers_by_qid(raw_dir, dataset_name, split_name)
    print(
        f"Loaded answers: dataset={answer_join.dataset} split={answer_join.split or 'all'} "
        f"rows={answer_join.loaded_rows} source={answer_join.source_path}"
    )

    a_records = [a_map[qid] for qid in qids]
    b_records = [b_map[qid] for qid in qids]
    a_reader_records, missing_a = _build_reader_records(a_records, answer_join.answers_by_qid, args.top_k_per_hop)
    b_reader_records, missing_b = _build_reader_records(b_records, answer_join.answers_by_qid, args.top_k_per_hop)
    baseline_reader_records: list[dict] | None = None
    missing_c = 0
    if args.include_no_context_baseline:
        baseline_reader_records, missing_c = _build_no_context_reader_records(a_records, answer_join.answers_by_qid)

    _enforce_missing_answer_guard(
        arm_name=args.name_a,
        missing_answers=missing_a,
        total=len(a_reader_records),
        max_missing_answers=args.max_missing_answers,
        dataset=answer_join.dataset,
        split=answer_join.split,
        answer_source=answer_join.source_path,
    )
    _enforce_missing_answer_guard(
        arm_name=args.name_b,
        missing_answers=missing_b,
        total=len(b_reader_records),
        max_missing_answers=args.max_missing_answers,
        dataset=answer_join.dataset,
        split=answer_join.split,
        answer_source=answer_join.source_path,
    )
    if baseline_reader_records is not None:
        _enforce_missing_answer_guard(
            arm_name=args.name_baseline,
            missing_answers=missing_c,
            total=len(baseline_reader_records),
            max_missing_answers=args.max_missing_answers,
            dataset=answer_join.dataset,
            split=answer_join.split,
            answer_source=answer_join.source_path,
        )

    out_path = Path(args.output)
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else out_path.with_suffix(out_path.suffix + ".checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_progress_path = checkpoint_dir / "progress.json"
    checkpoint_every_batches = max(1, args.checkpoint_every_batches)
    resume_from_checkpoints = not args.no_resume_checkpoints
    arm_ckpt_paths = {
        "a": checkpoint_dir / f"{args.name_a}.results.jsonl",
        "b": checkpoint_dir / f"{args.name_b}.results.jsonl",
        "c": checkpoint_dir / f"{args.name_baseline}.results.jsonl",
    }
    if args.no_resume_checkpoints:
        for p in [arm_ckpt_paths["a"], arm_ckpt_paths["b"], arm_ckpt_paths["c"], checkpoint_progress_path]:
            if p.exists():
                p.unlink()

    config_obj = OpenRouterConfig(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        concurrency=4,
    )

    a_results = _run_reader_in_batches(
        config_obj,
        cache_dir,
        reader_model,
        a_reader_records,
        args.batch_size,
        args.name_a,
        checkpoint_results_path=arm_ckpt_paths["a"],
        checkpoint_progress_path=checkpoint_progress_path,
        resume_from_checkpoints=resume_from_checkpoints,
        checkpoint_every_batches=checkpoint_every_batches,
    )
    b_results = _run_reader_in_batches(
        config_obj,
        cache_dir,
        reader_model,
        b_reader_records,
        args.batch_size,
        args.name_b,
        checkpoint_results_path=arm_ckpt_paths["b"],
        checkpoint_progress_path=checkpoint_progress_path,
        resume_from_checkpoints=resume_from_checkpoints,
        checkpoint_every_batches=checkpoint_every_batches,
    )
    baseline_results = []
    if baseline_reader_records is not None:
        baseline_results = _run_reader_in_batches(
            config_obj,
            cache_dir,
            reader_model,
            baseline_reader_records,
            args.batch_size,
            args.name_baseline,
            checkpoint_results_path=arm_ckpt_paths["c"],
            checkpoint_progress_path=checkpoint_progress_path,
            resume_from_checkpoints=resume_from_checkpoints,
            checkpoint_every_batches=checkpoint_every_batches,
        )

    a_by_qid = {r["qid"]: r for r in a_results}
    b_by_qid = {r["qid"]: r for r in b_results}
    c_by_qid = {r["qid"]: r for r in baseline_results}
    paired = []
    for qid in qids:
        ra = a_by_qid.get(qid)
        rb = b_by_qid.get(qid)
        if not ra or not rb:
            continue
        row = {
            "qid": qid,
            "em_a": ra.get("em", 0.0),
            "f1_a": ra.get("f1", 0.0),
            "em_b": rb.get("em", 0.0),
            "f1_b": rb.get("f1", 0.0),
            "delta_em_b_minus_a": rb.get("em", 0.0) - ra.get("em", 0.0),
            "delta_f1_b_minus_a": rb.get("f1", 0.0) - ra.get("f1", 0.0),
            "pred_a": ra.get("predicted_answer", ""),
            "pred_b": rb.get("predicted_answer", ""),
            "gold_answer": ra.get("gold_answer", ""),
        }
        rc = c_by_qid.get(qid)
        if rc:
            row.update(
                {
                    "em_c": rc.get("em", 0.0),
                    "f1_c": rc.get("f1", 0.0),
                    "pred_c": rc.get("predicted_answer", ""),
                    "delta_em_a_minus_c": ra.get("em", 0.0) - rc.get("em", 0.0),
                    "delta_em_b_minus_c": rb.get("em", 0.0) - rc.get("em", 0.0),
                    "delta_f1_a_minus_c": ra.get("f1", 0.0) - rc.get("f1", 0.0),
                    "delta_f1_b_minus_c": rb.get("f1", 0.0) - rc.get("f1", 0.0),
                }
            )
        paired.append(row)

    out = {
        "meta": {
            "dataset": dataset_name,
            "split": split_name,
            "rollout_a": args.rollout_a,
            "rollout_b": args.rollout_b,
            "name_a": args.name_a,
            "name_b": args.name_b,
            "sample_size_requested": args.sample_size,
            "sample_size_actual": len(qids),
            "seed": args.seed,
            "qid_order_file": str(qid_order_path) if qid_order_path else None,
            "top_k_per_hop": args.top_k_per_hop,
            "reader_model": reader_model,
            "paired_bootstrap_samples": args.bootstrap_samples,
            "include_no_context_baseline": bool(args.include_no_context_baseline),
            "checkpoint_dir": str(checkpoint_dir),
            "checkpoint_every_batches": checkpoint_every_batches,
            "resume_from_checkpoints": resume_from_checkpoints,
            "answer_join": {
                "source_path": answer_join.source_path,
                "loaded_rows": answer_join.loaded_rows,
                "missing_answers": {
                    args.name_a: missing_a,
                    args.name_b: missing_b,
                    args.name_baseline: missing_c if baseline_reader_records is not None else None,
                },
            },
            "notes": [
                "Uses post-hop2 context built as top_k_per_hop passages per hop from rollout.retrieved, sorted by passage score within hop.",
                f"Gold answers joined from {answer_join.source_path} using dataset={dataset_name} split={split_name}.",
            ],
        },
        "summary": {
            args.name_a: _summarize(a_results),
            args.name_b: _summarize(b_results),
        },
        "paired_summary": {
            "n": len(paired),
            "mean_delta_em_b_minus_a": mean(x["delta_em_b_minus_a"] for x in paired) if paired else 0.0,
            "mean_delta_f1_b_minus_a": mean(x["delta_f1_b_minus_a"] for x in paired) if paired else 0.0,
            "b_better_em_count": sum(1 for x in paired if x["delta_em_b_minus_a"] > 0),
            "a_better_em_count": sum(1 for x in paired if x["delta_em_b_minus_a"] < 0),
            "equal_em_count": sum(1 for x in paired if x["delta_em_b_minus_a"] == 0),
            "b_better_f1_count": sum(1 for x in paired if x["delta_f1_b_minus_a"] > 0),
            "a_better_f1_count": sum(1 for x in paired if x["delta_f1_b_minus_a"] < 0),
            "equal_f1_count": sum(1 for x in paired if x["delta_f1_b_minus_a"] == 0),
        },
        "paired_bootstrap": {
            f"{args.name_b}_minus_{args.name_a}": _paired_delta_summary(
                paired,
                delta_em_key="delta_em_b_minus_a",
                delta_f1_key="delta_f1_b_minus_a",
                num_samples=args.bootstrap_samples,
                seed=args.seed,
            )
        },
        "paired_examples": paired,
    }
    if baseline_results:
        out["summary"][args.name_baseline] = _summarize(baseline_results)
        out["paired_vs_baseline"] = {
            "n": len(paired),
            "mean_delta_em_a_minus_c": mean(x.get("delta_em_a_minus_c", 0.0) for x in paired) if paired else 0.0,
            "mean_delta_em_b_minus_c": mean(x.get("delta_em_b_minus_c", 0.0) for x in paired) if paired else 0.0,
            "mean_delta_f1_a_minus_c": mean(x.get("delta_f1_a_minus_c", 0.0) for x in paired) if paired else 0.0,
            "mean_delta_f1_b_minus_c": mean(x.get("delta_f1_b_minus_c", 0.0) for x in paired) if paired else 0.0,
            "a_better_em_than_c_count": sum(1 for x in paired if x.get("delta_em_a_minus_c", 0.0) > 0),
            "b_better_em_than_c_count": sum(1 for x in paired if x.get("delta_em_b_minus_c", 0.0) > 0),
            "a_better_f1_than_c_count": sum(1 for x in paired if x.get("delta_f1_a_minus_c", 0.0) > 0),
            "b_better_f1_than_c_count": sum(1 for x in paired if x.get("delta_f1_b_minus_c", 0.0) > 0),
        }
        out["paired_bootstrap"][f"{args.name_a}_minus_{args.name_baseline}"] = _paired_delta_summary(
            paired,
            delta_em_key="delta_em_a_minus_c",
            delta_f1_key="delta_f1_a_minus_c",
            num_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        out["paired_bootstrap"][f"{args.name_b}_minus_{args.name_baseline}"] = _paired_delta_summary(
            paired,
            delta_em_key="delta_em_b_minus_c",
            delta_f1_key="delta_f1_b_minus_c",
            num_samples=args.bootstrap_samples,
            seed=args.seed,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    _write_json_checkpoint(
        checkpoint_progress_path,
        {
            "label": "final",
            "done": True,
            "sample_size_actual": len(qids),
            "arms": {
                args.name_a: len(a_results),
                args.name_b: len(b_results),
                args.name_baseline: len(baseline_results) if baseline_results else 0,
            },
            "output": str(out_path),
        },
    )
    if args.cleanup_checkpoints_on_success:
        for p in [arm_ckpt_paths["a"], arm_ckpt_paths["b"], arm_ckpt_paths["c"], checkpoint_progress_path]:
            if p.exists():
                p.unlink()
        try:
            checkpoint_dir.rmdir()
        except OSError:
            pass

    print("\n=== Reader EM/F1 comparison (matched sample) ===")
    for name, stats in out["summary"].items():
        print(f"{name}: n={stats['n']} EM={stats['em']:.4f} F1={stats['f1']:.4f} UNANSWERABLE={stats['unanswerable_rate']:.4f}")
    ps = out["paired_summary"]
    print(
        f"Delta ({args.name_b} - {args.name_a}): mean EM={ps['mean_delta_em_b_minus_a']:+.4f}, "
        f"mean F1={ps['mean_delta_f1_b_minus_a']:+.4f}; "
        f"EM wins B/A/E={ps['b_better_em_count']}/{ps['a_better_em_count']}/{ps['equal_em_count']}"
    )
    pb_main = out["paired_bootstrap"][f"{args.name_b}_minus_{args.name_a}"]
    print(
        f"Paired bootstrap 95% CI ({args.name_b} - {args.name_a}): "
        f"EM [{pb_main['em_ci_95'][0]:+.4f}, {pb_main['em_ci_95'][1]:+.4f}], "
        f"F1 [{pb_main['f1_ci_95'][0]:+.4f}, {pb_main['f1_ci_95'][1]:+.4f}]"
    )
    if baseline_results:
        pb = out["paired_vs_baseline"]
        print(
            f"Additive vs {args.name_baseline}: "
            f"{args.name_a} ΔEM={pb['mean_delta_em_a_minus_c']:+.4f} ΔF1={pb['mean_delta_f1_a_minus_c']:+.4f}; "
            f"{args.name_b} ΔEM={pb['mean_delta_em_b_minus_c']:+.4f} ΔF1={pb['mean_delta_f1_b_minus_c']:+.4f}"
        )
        for key in (
            f"{args.name_a}_minus_{args.name_baseline}",
            f"{args.name_b}_minus_{args.name_baseline}",
        ):
            ci = out["paired_bootstrap"][key]
            print(
                f"Paired bootstrap 95% CI ({key}): "
                f"EM [{ci['em_ci_95'][0]:+.4f}, {ci['em_ci_95'][1]:+.4f}], "
                f"F1 [{ci['f1_ci_95'][0]:+.4f}, {ci['f1_ci_95'][1]:+.4f}]"
            )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
