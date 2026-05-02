# Reproduction Guide

This document is the canonical reproduction map for the SEER paper. It is
written for reviewers and later public users who need to understand:

- which code paths are canonical,
- which external assets are required,
- which regimes correspond to the paper tables,
- which stages are fully local versus API- or RunPod-backed,
- what "faithful reproduction" means for this artifact.

## Reproducibility Contract

This repository supports **faithful methodological reproduction**, not guaranteed
bitwise regeneration of every generated artifact.

Reasons:

- some stages call hosted APIs (OpenRouter, Jina, reader models),
- MuSiQue retrieval uses a separately served RunPod ColBERT stack,
- external providers may drift over time,
- raw corpora, indexes, checkpoints, and large generated artifacts are not
  redistributed by default.

What is provided:

- code
- configs
- prompts
- tests
- bundle manifest
- release policy
- exact canonical script paths

What is not provided by default:

- base weights
- LoRA adapters
- generated rollouts
- scored files
- reader outputs
- full retrieval indexes
- raw benchmark copies

Glossary note:

- historical code/config token `mmr` means the paper's **SEER-MR** scorer
- it does **not** mean Maximal Marginal Relevance

For retrieval-backed stages, the expected model is:

- we provide the pipeline and exact commands
- users obtain compatible retrieval corpora / indexes from official or upstream
  sources
- we do not redistribute third-party passage collections when the provenance of
  the exact local artifact is unclear

## Canonical Scripts

The end-to-end pipeline uses these canonical scripts:

- rollouts: `scripts/rollouts/generate.py`
- SFT prep: `scripts/sft/prepare_data.py`
- SFT train: `scripts/sft/train.py`
- SFT eval: `scripts/sft/eval.py`
- IPO rollout generation: `scripts/dpo/generate.py`
- IPO scoring: `scripts/dpo/score.py`
- IPO pair construction: `scripts/dpo/prepare.py`
- IPO pair validation: `scripts/dpo/validate.py`
- IPO training: `scripts/dpo/train.py`
- downstream compare: `scripts/eval/compare_rollout_reader_em.py`
- monitoring report: `scripts/eval/monitoring_signal_report.py`

## Required External Assets

### Common

- Python environment from `pyproject.toml` / `uv.lock`
- `.env` values as needed:
  - `OPENROUTER_API_KEY`
  - `JINA_AI_API_KEY`

### HotpotQA

- Hotpot fullwiki question files in `data/raw/hotpot_fullwiki/`
- local `wiki2017` ColBERT index in `data/raw/colbert_index/wiki2017`
- local ColBERT passage collection in
  `data/raw/colbert_index/collection/collection.tsv`

Helper:

```bash
make download-index
```

This helper downloads:

- `hotpot_train_v1.1.json`
- `hotpot_dev_fullwiki_v1.json`
- HF `sher222/ColBERTv2-wiki2017-index`

It does **not** download the Hotpot passage `collection.tsv`. For a fully local
Hotpot rollout/eval path, you need both the index and the collection. If you do
not have the collection asset available locally, use a remote retriever instead.

Recommended upstream source for the passage collection:

- official HotpotQA processed / introductory-paragraph Wikipedia release:
  https://hotpotqa.github.io/wiki-readme.html

### MuSiQue

- MuSiQue raw data at `data/raw/musique/raw.jsonl`
- normalized output at `data/norm/musique.jsonl`
- RunPod-served `wiki20m.nbits2` ColBERT retrieval service
- matching RunPod-side `corpus/collection.tsv`

Dataset helper:

```bash
make download-datasets
```

By default this helper stages both normalized-dataset inputs:

- MuSiQue from Hugging Face (`dgslibisey/MuSiQue`) to
  `data/raw/musique/raw.jsonl`
- HotpotQA distractor from Hugging Face (`hotpot_qa`, `distractor`) to
  `data/raw/hotpot/raw.jsonl`

If you only want MuSiQue, run:

```bash
uv run python scripts/download_datasets.py --config configs/default.yaml --datasets musique
```

Normalization:

```bash
make normalize
```

MuSiQue retriever provisioning:

- see `scripts/runpod_musique/README.md`
- the canonical path builds or restores a `wiki20m.nbits2` ColBERT service from
  official/public upstream assets (for example DPR `psgs_w100`) plus the
  matching `corpus/collection.tsv`, rather than relying on a bundled local
  index

Important:

- `make download-index` does **not** set up the MuSiQue retriever
- `make normalize` does **not** download datasets or indexes

## Pipeline By Stage

### 1. Rollout generation

Rollouts generate multiple query-rewrite variants per question.

Inspect CLI:

```bash
uv run python scripts/rollouts/generate.py --help
```

Typical HotpotQA command:

```bash
uv run python scripts/rollouts/generate.py \
  --config configs/default.yaml \
  --dataset hotpot \
  --split train \
  --limit 5000 \
  --top-k-per-hop 3 \
  --variants-per-question 5 \
  --output data/rollouts/hotpot_train_multihop_k3_N5000_5variants.jsonl
```

Typical MuSiQue command:

```bash
uv run python scripts/rollouts/generate.py \
  --config configs/default.yaml \
  --dataset musique \
  --split train \
  --limit 10000 \
  --top-k-per-hop 3 \
  --variants-per-question 5 \
  --remote-url http://RUNPOD_HOST:PORT \
  --output data/rollouts/musique_train_multihop_k3_N10000_5variants_final.jsonl
```

Notes:

- Hotpot uses the local `wiki2017` index and local `collection.tsv` by default.
- MuSiQue uses a remote retriever via `--remote-url`.
- `--offset` is used for fresh question ranges in later stages.
- Bare `make rollouts` uses generic script defaults, not paper-regime values.
  For paper-like runs, use the explicit commands in this document, including
  `--top-k-per-hop 3` and `--variants-per-question 5`.

For minimal structure verification without a full retriever setup, run:

```bash
make release-smoke
```

This is an offline, fixture-based smoke check. It validates the canonical
entrypoints without downloading benchmark files or retrieval assets.

For a true local Hotpot retrieval canary after you supply compatible retrieval
assets, run:

```bash
uv run python scripts/rollouts/generate.py \
  --config configs/default.yaml \
  --dataset hotpot \
  --split dev \
  --limit 1 \
  --variants-per-question 1 \
  --top-k-per-hop 3 \
  --no-rewriter \
  --output .tmp/hotpot_retrieval_canary.jsonl
```

This verifies that the local ColBERT index plus `collection.tsv` can be loaded
and used for a real rollout.

### 2. SFT data preparation

Inspect CLI:

```bash
uv run python scripts/sft/prepare_data.py --help
```

This stage chooses the best rollout variant **per `(base_qid, hop)`** by a
selector.

Main reported selectors:

- `gold`
- `mmr`
- `raw_jina_maxmean`
- `raw_jina_max`

Typical Gold SFT prep:

```bash
uv run python scripts/sft/prepare_data.py \
  --selector gold \
  --input data/rollouts/hotpot_train_multihop_k3_N5000_5variants.jsonl \
  --output data/sft_data/hotpot_train_sft_gold_k3_N5000.jsonl
```

Typical SEER-MR / MMR SFT prep:

```bash
uv run python scripts/sft/prepare_data.py \
  --selector mmr \
  --input data/rollouts/musique_train_multihop_k3_N10000_5variants_final.jsonl \
  --output data/sft_data/musique_train_sft_mmr_k3_N10000_from_final_clean.jsonl
```

Behavior:

- `--scores` is the path to a precomputed selector-score JSONL. Use it when you
  want to reuse an already-scored artifact rather than regenerating scores.
- `mmr`: if `--scores` is omitted, `prepare_data.py` can generate the
  question-level requirements map and Jina rerank scores in-process, then use
  those scores to select the best variant.
- `raw_jina_*`: requires explicit precomputed score input.

### 3. SFT training

Typical:

```bash
uv run python scripts/sft/train.py \
  --config configs/llama3_sft_gold_k3_N5000_runpod.yml
```

Run full SFT training on a Linux GPU machine, not a CPU-only inspection
environment.

Known-good environment from the reported runs:

- Python 3.11
- CUDA 12.4
- single-GPU Axolotl training stack
- one known-good RunPod image:
  `axolotlai/axolotl-cloud:main-py3.11-cu124-2.6.0`
- observed worker classes in the reported runs:
  RTX 4090 and L40S

The paper uses QLoRA SFT on Llama 3 8B Instruct. Exact training
hyperparameters are described in the paper appendix and encoded in the config
files.

### 4. IPO rollout generation

Inspect CLI:

```bash
uv run python scripts/dpo/generate.py --help
```

This stage is where the system selects a **shared context candidate** at each
hop and then generates the next-hop candidate rollouts conditioned on that
selected context.

Typical Gold rollout generation:

```bash
uv run python scripts/dpo/generate.py \
  --input data/rollouts/hotpot_train_multihop_k3_N5000_5variants.jsonl \
  --output data/dpo_rollouts/hotpot_dpo_gold_5k.jsonl \
  --context-select gold
```

Typical SEER-MR rollout generation:

```bash
uv run python scripts/dpo/generate.py \
  --input data/rollouts/hotpot_train_multihop_k3_N5000_5variants.jsonl \
  --output data/dpo_rollouts/hotpot_dpo_mmr_5k.jsonl \
  --context-select mmr \
  --auto-requirements-out data/dpo_rollouts/hotpot_mmr_requirements_5k.json
```

`--auto-requirements-out` is the path where the script writes the generated
question-to-requirements map when `--context-select mmr` is used without a
pre-existing `--requirements` file. Reusing that file avoids regenerating
requirements in later scoring or audit steps.

### 5. IPO scoring

Inspect CLI:

```bash
uv run python scripts/dpo/score.py --help
```

This stage scores the candidates inside IPO rollouts. The reported paper runs
use:

- `gold`
- `mmr` (SEER-MR)
- `raw_jina_maxmean` / `raw_jina_max`

The scripts live under `scripts/dpo/`, but the reported runs are **IPO** runs,
implemented as TRL `DPOTrainer` with `loss_type=ipo`.

### 6. IPO pair construction

Inspect CLI:

```bash
uv run python scripts/dpo/prepare.py --help
```

The paper uses different pair modes for different supervision families:

- Gold: adjacent-tier pairing on discrete AP values
- HotpotQA non-iterative SEER-MR: exhaustive pairing with `min_gap=0.10`
- HotpotQA Iter2 SEER-MR: exhaustive pairing with `min_gap=0.05`
- MuSiQue Gold: adjacent-tier pairing (`mingap000_adj`)
- MuSiQue SEER-MR: exhaustive pairing with `min_gap=0.01143`
- MuSiQue Raw-Jina: exhaustive pairing with `min_gap=0.01035`

Typical HotpotQA SEER-MR command:

```bash
uv run python scripts/dpo/prepare.py \
  --input data/dpo_rollouts/hotpot_dpo_mmr_5k_scored.jsonl \
  --output data/dpo_pairs/hotpot_mmr_5k_pairs.jsonl \
  --scoring mmr \
  --min-gap 0.10 \
  --pair-mode exhaustive
```

### 7. IPO training

Typical:

```bash
uv run python scripts/dpo/train.py \
  --config configs/dpo_mmr_5k_r64.yml
```

The paper uses IPO settings encoded in the `configs/dpo_*` files.

Run full IPO training on the same class of Linux GPU machine used for SFT.

Known-good environment from the reported runs:

- Python 3.11
- CUDA 12.4
- TRL + PEFT + bitsandbytes 4-bit training stack
- one known-good RunPod image:
  `axolotlai/axolotl-cloud:main-py3.11-cu124-2.6.0`
- observed worker classes in the reported runs:
  RTX 4090 and L40S

The smoke path validates `dpo/train.py --dry-run`; full training requires the
actual GPU training stack above.

### 8. Evaluation

There are two distinct evaluation layers in the release:

1. **retrieval rollout evaluation**
   - generate evaluated rollout files with retrieved context and reader answers
2. **downstream comparison / monitoring analysis**
   - compare rollout files with a frozen reader and summarize metric behavior

#### 8A. Retrieval rollout evaluation

Inspect CLI:

```bash
uv run python scripts/sft/eval.py --help
```

Typical Hotpot evaluation:

```bash
uv run python scripts/sft/eval.py \
  --config configs/default.yaml \
  --dataset hotpot \
  --split dev \
  --inference-url http://INFERENCE_HOST:PORT \
  --limit 7405 \
  --output data/rollouts/hotpot_eval.jsonl
```

Typical MuSiQue evaluation:

```bash
uv run python scripts/sft/eval.py \
  --config configs/default.yaml \
  --dataset musique \
  --split validation \
  --inference-url http://INFERENCE_HOST:PORT \
  --retrieval-url http://RUNPOD_HOST:PORT \
  --output data/rollouts/musique_eval.jsonl
```

`scripts/sft/eval.py` is the retrieval rollout evaluation runner. It:

- performs retrieval
- runs the frozen reader over the retrieved context
- writes rollout/result JSONL files for later comparison

#### 8B. Downstream EM/F1 comparison

Inspect CLI:

```bash
uv run python scripts/eval/compare_rollout_reader_em.py --help
```

`scripts/eval/compare_rollout_reader_em.py` is the downstream answer-quality
comparison tool. It:

- loads two evaluated rollout/result files
- joins them on matched qids
- runs paired frozen-reader EM/F1 comparison with bootstrap summaries

#### 8C. Monitoring report

Inspect CLI:

```bash
uv run python scripts/eval/monitoring_signal_report.py --help
```

`scripts/eval/monitoring_signal_report.py` is the monitoring-analysis stage. It
combines scored rollouts and reader outputs into AUROC / threshold / agreement
summaries.

## Reported Paper Regimes

The paper’s headline regimes map to these config families:

### HotpotQA single-round Gold

- SFT: `configs/llama3_sft_gold_k3_N5000_runpod.yml`
- IPO: `configs/dpo_gold_10k_r64.yml`

### HotpotQA single-round SEER-MR

- SFT: `configs/llama3_sft_jina_mmr_k3_N5000_runpod.yml`
- IPO: `configs/dpo_mmr_10k_r64.yml`

### HotpotQA 5K controlled ablation

- shared SFT base:
  - `configs/llama3_sft_jina_mmr_k3_N5000_runpod.yml`
- SEER-MR IPO:
  - `configs/dpo_mmr_5k_r64.yml`
- Raw-Jina IPO:
  - `configs/dpo_raw_jina_maxmean_5k_r64.yml`

### HotpotQA Iter2

- Gold SFT:
  - `configs/llama3_sft_gold_iteration_20k_25k_runpod.yml`
- Gold IPO:
  - `configs/dpo_gold_iteration_20k_25k_r64.yml`
- SEER-MR SFT:
  - `configs/llama3_sft_mmr_iteration_20k_25k_runpod.yml`
- SEER-MR IPO:
  - `configs/dpo_mmr_iteration_20k_25k_r64.yml`

### MuSiQue validation

- Gold SFT:
  - `configs/llama3_sft_gold_musique_k3_N10000_runpod.yml`
- Gold IPO:
  - `configs/dpo_musique_gold_10k_r64.seer_sft.yml`
- SEER-MR SFT:
  - `configs/llama3_sft_mmr_musique_k3_N10000_fullhop_runpod.yml`
- SEER-MR IPO:
  - `configs/dpo_musique_mmr_10k_r64.seer_sft.yml`
- Raw-Jina IPO:
  - `configs/dpo_musique_raw_jina_maxmean_10k_r64.seer_sft.yml`

## Iterative Training

Iterative training reuses the same canonical scripts with different inputs and
configs.

Pattern:

1. merge the previous policy into a base checkpoint
2. generate fresh on-policy rollouts on a new question slice (`--offset`)
3. prepare iteration SFT data
4. train iteration SFT adapter
5. generate / score / pair IPO rollouts
6. train iteration IPO adapter

HotpotQA Iter2 merge commands:

```bash
uv run python scripts/merge_model.py \
  --base-model meta-llama/Meta-Llama-3-8B-Instruct \
  --sft-adapter /workspace/data/seer_sft/qlora-out-gold-5k-k3 \
  --ipo-adapter /workspace/data/dpo_out/dpo-gold-5k-ipo-r64 \
  --output /workspace/data/merged_models/gold-iter1-merged

uv run python scripts/merge_model.py \
  --base-model meta-llama/Meta-Llama-3-8B-Instruct \
  --sft-adapter /workspace/data/seer_sft/qlora-out-jina-mmr-5k-k3 \
  --ipo-adapter /workspace/data/dpo_out/dpo-mmr-5k-ipo-r64 \
  --output /workspace/data/merged_models/mmr-iter1-merged
```

So yes: it is the same script surface, plus a merge step and iteration-specific
configs / input files.

## What To Verify After Documentation-Only Changes

Documentation and release-surface cleanup should not change experimental
behavior. After doc updates, verify:

```bash
make env-check
make release-smoke
uv run pytest tests/test_sft_prepare_data.py tests/test_dpo_utils.py tests/test_dpo_validate.py tests/test_normalize_datasets_script.py
uv run python scripts/release/export_review_bundle.py --dry-run
```

If the bundle contents change intentionally, also rebuild it once:

```bash
make review_bundle
```

## What Is Deliberately Out Of Scope For The Default Release

- publishing base weights
- publishing LoRA adapters by default
- redistributing Wikipedia dumps or full indexes
- redistributing generated rollouts or reader outputs by default
- guaranteeing exact provider-stable outputs from hosted APIs

Those constraints are part of the release policy in `RELEASE.md`.
