# SFT Scripts

Canonical SFT flow:

1. Prepare SFT data
2. Train SFT model
3. Evaluate SFT model

## 1) Prepare Data

```bash
uv run python scripts/sft/prepare_data.py \
  --selector gold \
  --input data/rollouts/ROLL_OUTS.jsonl \
  --output data/sft_data/SFT_DATA.jsonl
```

Selectors:
- `gold`
- `mmr` (`--scores` optional; auto-builds MMR scores when omitted)
- `raw_jina_maxmean` (requires `--scores`)
- `raw_jina_max` (requires `--scores`)

Data hygiene (automatic, default behavior):
- Requires complete DSPy markers if markers are present (`[[ ## search_query ## ]] ... [[ ## completed ## ]]`).
- Rejects malformed marker residue and narrative/non-query outputs.
- If top-scoring variant is invalid at a hop, selects the best next valid variant automatically.
- Prints invalid-query reason counts at the end.

## 1B) Prepare MMR Data (One Command)

```bash
uv run python scripts/sft/prepare_data.py \
  --selector mmr \
  --input data/rollouts/musique_train_multihop_k3_N10000_5variants_final.jsonl \
  --output data/sft_data/musique_train_sft_mmr_k3_N10000_from_final_clean.jsonl
```

`--scores` points at a precomputed selector-score JSONL. Use it when you want
to reuse an already-scored artifact rather than regenerating scores.

`prepare_data.py --selector mmr` executes requirements generation, Jina reranker
scoring, and MMR selection in-process when `--scores` is omitted.

For a smoke run:

```bash
uv run python scripts/sft/prepare_data.py \
  --selector mmr \
  --input data/rollouts/musique_train_multihop_k3_N10000_5variants_final.jsonl \
  --output data/sft_data/musique_train_sft_mmr_smoke50.jsonl \
  --mmr-limit 50
```

## 2) Train

```bash
uv run python scripts/sft/train.py \
  --config configs/llama3_sft_gold_k3_N5000_runpod.yml
```

Use `--dry-run` to print command without executing.

Run full SFT training on a Linux GPU machine with the Axolotl training stack.
A known-good environment from the reported runs was:

- Python 3.11
- CUDA 12.4
- RunPod image `axolotlai/axolotl-cloud:main-py3.11-cu124-2.6.0`
- RTX 4090 or L40S workers

## 3) Evaluate

```bash
uv run python scripts/sft/eval.py \
    --config configs/default.yaml \
    --dataset hotpot \
    --split dev \
    --inference-url http://RUNPOD_IP:PORT \
    --retrieval-url http://RUNPOD_IP:PORT \
    --limit 7405
```

Notes:
- `--inference-url` is optional when using `--openrouter-model`.
- `--retrieval-url` points at ColBERT `/api/search` on the runpod.

## 1C) Prepare Raw Jina Ablation SFT Data

```bash
uv run python scripts/sft/prepare_data.py \
  --selector raw_jina_maxmean \
  --input data/rollouts/hotpot_dpo_rollouts_raw_jina_scored.jsonl \
  --scores data/dpo_raw_jina/rollout_scores.jsonl \
  --output data/sft_data/hotpot_train_sft_raw_jina_maxmean.jsonl
```

`raw_jina_max` is identical syntax with `--selector raw_jina_max`.
