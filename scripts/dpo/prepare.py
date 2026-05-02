#!/usr/bin/env python3
"""
Convert scored DPO rollouts into preference pairs for TRL DPOTrainer.

Supports two scoring modes:
  - gold: Use Gold AP computed from gold_titles match (no API cost)
  - seer: Use Seer AP from GPT-4o-mini few-shot labels (requires prior labeling)

For each question × hop:
  1. Score each candidate by AP (Gold or Seer)
  2. Deduplicate identical queries
  3. Group by score level
  4. Create preference pairs with either adjacent-tier or exhaustive pairing

Output format (TRL DPOTrainer):
  {"prompt": "...", "chosen": "...", "rejected": "..."}

Usage:
    # Gold AP scoring (uses retrieval results directly)
    python scripts/dpo/prepare.py \
        --input data/dpo_rollouts/hotpot_train_dpo_gold_sft5k_N8.jsonl \
        --output data/dpo_pairs/hotpot_dpo_gold_5k.jsonl \
        --scoring gold

    # Seer AP scoring (requires prior Seer labeling)
    python scripts/dpo/prepare.py \
        --input data/dpo_rollouts/hotpot_train_dpo_seer_fs_sft5k_N8_labeled.jsonl \
        --output data/dpo_pairs/hotpot_dpo_seer_fs_5k.jsonl \
        --scoring seer
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from seer.util_io import read_jsonl, write_jsonl, ensure_dir
from seer.dpo_utils import iter_hop_records


INSTRUCTION = (
    "Generate a search query to find information needed to answer "
    "the following question. Use the provided context if available."
)


def format_dspy_output(query: str) -> str:
    """Format a query string in DSPy output format (matching SFT training)."""
    return f"[[ ## search_query ## ]]\n{query}\n\n[[ ## completed ## ]]"


def build_prompt(context: str, question: str) -> str:
    """Build the Alpaca-style prompt string (matching SFT training format).

    Note: We store just the instruction + input text. The Alpaca template
    wrapping (### Instruction / ### Input / ### Response) is applied by
    the tokenizer / training framework.
    """
    return f"Instruction: {INSTRUCTION}\nInput: Context: {context}\n\nQuestion: {question}"


def get_candidate_score(candidate: dict, scoring: str, hop: int) -> float | None:
    """Extract the AP score for a candidate under the given scoring mode."""
    if scoring == "gold":
        if hop == 1:
            return candidate.get("gold_ap")
        else:
            # For hop >=2 use cumulative AP under selected shared context.
            return candidate.get("gold_ap_cumulative")
    elif scoring == "seer":
        return candidate.get("seer_ap")
    elif scoring == "decomp_binary":
        score = candidate.get("decomp_binary_ap")
        if score is None:
            score = candidate.get("decomposed_binary_ap")
        return score
    elif scoring == "mmr":
        return candidate.get("mmr_ap")
    elif scoring == "raw_jina_maxmean":
        # Raw-query Jina baseline (no decomposition): mean of top-k passage scores.
        # Current ablation stores the top-3 variant under this field.
        return candidate.get("raw_jina_maxmean_top3")
    elif scoring == "raw_jina_max":
        # Raw-query Jina baseline (no decomposition): best passage score only.
        return candidate.get("raw_jina_max")
    return None


def make_pairs_from_candidates(
    candidates: list[dict],
    scoring: str,
    hop: int,
    min_gap: float = 0.0,
    pair_mode: str = "adjacent",
) -> list[tuple[dict, dict]]:
    """
    Create preference pairs from scored candidates.

    1. Deduplicate by query text
    2. Group by rounded score
    3. Create pairs (adjacent-tier or exhaustive)

    pair_mode:
      - "adjacent": Pairs between consecutive score levels only (LeReT-style).
        More conservative, avoids huge gaps that can destabilize training.
      - "exhaustive": All (higher, lower) combinations with score gap > min_gap.
    """
    # Score and deduplicate
    scored = {}
    for c in candidates:
        query = c["query"].strip()
        score = get_candidate_score(c, scoring, hop)
        if score is None:
            continue
        # Keep highest-scoring version if duplicate query
        if query not in scored or score > scored[query]:
            scored[query] = score

    if len(scored) < 2:
        return []

    if pair_mode == "adjacent":
        # Group by rounded score, pair between adjacent levels
        groups = defaultdict(list)
        for query, score in scored.items():
            groups[round(score, 3)].append(query)

        score_levels = sorted(groups.keys(), reverse=True)
        if len(score_levels) < 2:
            return []

        pairs = []
        for i in range(len(score_levels) - 1):
            high_score = score_levels[i]
            low_score = score_levels[i + 1]
            if high_score - low_score < min_gap:
                continue
            for q_high in groups[high_score]:
                for q_low in groups[low_score]:
                    pairs.append((
                        {"query": q_high, "score": high_score},
                        {"query": q_low, "score": low_score},
                    ))
        return pairs

    else:  # exhaustive
        sorted_candidates = sorted(scored.items(), key=lambda x: x[1], reverse=True)
        pairs = []
        for i in range(len(sorted_candidates)):
            q_high, s_high = sorted_candidates[i]
            for j in range(i + 1, len(sorted_candidates)):
                q_low, s_low = sorted_candidates[j]
                if s_high - s_low > min_gap:
                    pairs.append((
                        {"query": q_high, "score": s_high},
                        {"query": q_low, "score": s_low},
                    ))
        return pairs


def process_rollouts(
    rollouts: list[dict],
    scoring: str,
    min_gap: float = 0.0,
    max_pairs_per_group: int = 20,
    pair_mode: str = "adjacent",
) -> list[dict]:
    """
    Convert rollouts into DPO preference pair records.

    Returns list of {"prompt": ..., "chosen": ..., "rejected": ...} dicts.
    """
    dpo_records = []
    stats = {
        "questions": 0,
        "hop_pairs": defaultdict(int),
        "skipped_no_variance": 0,
    }

    for rollout in rollouts:
        qid = rollout["qid"]
        question = rollout["question"]
        stats["questions"] += 1

        total_pairs_for_question = 0
        for hop_num, hop_record in iter_hop_records(rollout):
            context = hop_record.get("prompt_context", "N/A")
            candidates = hop_record.get("candidates", [])
            pairs = make_pairs_from_candidates(
                candidates,
                scoring,
                hop=hop_num,
                min_gap=min_gap,
                pair_mode=pair_mode,
            )

            if len(pairs) > max_pairs_per_group:
                import random

                pairs = random.sample(pairs, max_pairs_per_group)

            prompt = build_prompt(context, question)
            for chosen, rejected in pairs:
                dpo_records.append(
                    {
                        "prompt": prompt,
                        "chosen": format_dspy_output(chosen["query"]),
                        "rejected": format_dspy_output(rejected["query"]),
                        "meta": {
                            "qid": qid,
                            "hop": hop_num,
                            "chosen_score": chosen["score"],
                            "rejected_score": rejected["score"],
                            "scoring": scoring,
                        },
                    }
                )
            stats["hop_pairs"][hop_num] += len(pairs)
            total_pairs_for_question += len(pairs)

        if total_pairs_for_question == 0:
            stats["skipped_no_variance"] += 1

    return dpo_records, stats


def main():
    parser = argparse.ArgumentParser(
        description="Prepare DPO preference pairs from scored rollouts"
    )
    parser.add_argument("--input", required=True, help="Path to DPO rollouts JSONL")
    parser.add_argument("--output", required=True, help="Path to save DPO pairs JSONL")
    parser.add_argument(
        "--scoring",
        required=True,
        choices=["gold", "seer", "decomp_binary", "mmr", "raw_jina_maxmean", "raw_jina_max"],
        help=(
            "Scoring method: 'gold' (from retrieval), 'seer' (monolithic), "
            "'decomp_binary' (decomposed), 'mmr' (SEER-MR / Jina continuous), "
            "'raw_jina_maxmean' (no-decomp raw Jina top-k mean), or "
            "'raw_jina_max' (no-decomp raw Jina max)"
        ),
    )
    parser.add_argument(
        "--min-gap",
        type=float,
        default=0.0,
        help="Minimum AP gap to create a pair (default: 0.0, any difference)",
    )
    parser.add_argument(
        "--max-pairs-per-group",
        type=int,
        default=20,
        help="Max pairs per question-hop to cap dataset size",
    )
    parser.add_argument(
        "--pair-mode",
        default="adjacent",
        choices=["adjacent", "exhaustive"],
        help="Pair mode: 'adjacent' (LeReT-style, consecutive tiers) or 'exhaustive' (all combos)",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    import random
    random.seed(args.seed)

    print(f"Loading rollouts from {args.input}...")
    rollouts = read_jsonl(args.input)
    print(f"  Loaded {len(rollouts)} rollout records")

    print(f"Creating preference pairs (scoring={args.scoring}, mode={args.pair_mode}, min_gap={args.min_gap})...")
    dpo_records, stats = process_rollouts(
        rollouts,
        scoring=args.scoring,
        min_gap=args.min_gap,
        max_pairs_per_group=args.max_pairs_per_group,
        pair_mode=args.pair_mode,
    )

    # Shuffle for training
    random.shuffle(dpo_records)

    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    write_jsonl(output_path, dpo_records)

    print(f"\nDPO Pair Statistics:")
    print(f"  Questions processed: {stats['questions']}")
    for hop_num in sorted(stats["hop_pairs"]):
        print(f"  Hop {hop_num} pairs: {stats['hop_pairs'][hop_num]}")
    print(f"  Total pairs: {len(dpo_records)}")
    print(f"  Skipped (no variance): {stats['skipped_no_variance']}")
    print(f"\n  Avg pairs/question: {len(dpo_records)/max(stats['questions'],1):.1f}")

    # Score distribution analysis
    if dpo_records:
        chosen_scores = [r["meta"]["chosen_score"] for r in dpo_records]
        rejected_scores = [r["meta"]["rejected_score"] for r in dpo_records]
        gaps = [c - r for c, r in zip(chosen_scores, rejected_scores)]
        print(f"\n  Score gap: mean={sum(gaps)/len(gaps):.4f}, "
              f"min={min(gaps):.4f}, max={max(gaps):.4f}")
        print(f"  Chosen AP: mean={sum(chosen_scores)/len(chosen_scores):.4f}")
        print(f"  Rejected AP: mean={sum(rejected_scores)/len(rejected_scores):.4f}")

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
