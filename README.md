# SEER: Scalable Evidence-based Evaluation and Reward for LLM-Based Retrieval

This repository contains the code and experiment configuration for SEER, a
retrieval-supervision method for multi-hop query rewriting.

The canonical pipeline covers:

- rollout generation
- SFT data preparation and training
- IPO pair generation, scoring, and training
- batched evaluation
- monitoring / reader-compare analysis
- RunPod-based MuSiQue ColBERT serving

This is a **method/evaluation artifact**, not a new-dataset release. Raw
datasets, full indexes, checkpoints, generated rollouts, scored artifacts, and
private logs are excluded from the default review bundle unless explicitly
allowed by upstream terms.

## Canonical Docs

Use these documents in this order:

1. `README.md` - repo structure, setup, and canonical entrypoints
2. `REPRO.md` - exact reproduction map for the paper regimes
3. `RELEASE.md` - release policy, licensing, excluded assets, review bundle
4. `scripts/sft/README.md` - SFT-stage details
5. `scripts/runpod_musique/README.md` - MuSiQue retriever setup details

## Repository Layout

- `configs/` - canonical experiment configs
- `seer/` - package code
- `scripts/rollouts/` - rollout generation
- `scripts/sft/` - SFT prep / train / eval
- `scripts/dpo/` - preference optimization prep / train
- `scripts/eval/` - downstream comparison / monitoring
- `scripts/runpod_musique/` - MuSiQue RunPod ColBERT setup
- `tests/` - smoke and unit tests

## Setup

```bash
uv sync
cp .env.example .env  # optional convenience copy
make env-check
```

`make env-check` is a lightweight config/logging sanity check. It verifies that
the canonical config can be loaded and that the configured log path is writable;
it is not a full CUDA / API / retrieval-asset validator.

Required credentials depend on which pipeline stages you run:

- `OPENROUTER_API_KEY` - query-generation, requirements generation, some scoring
- `JINA_AI_API_KEY` - Jina reranker-backed `mmr` / raw-Jina scoring

Glossary note:

- historical code/config token `mmr` means the paper's **SEER-MR** scorer
- it does **not** mean Maximal Marginal Relevance

Optional credentials:

- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`, `S3_URI`
  - only for index or artifact restore flows that use S3
- `WANDB_API_KEY`, `WANDB_PROJECT`
  - only for experiment logging

See `.env.example` for the canonical environment-variable template.

## External Assets And What The Helpers Actually Do

The two datasets do **not** share one retrieval setup.

For retrieval-backed stages, this repository supports a **bring-your-own
retrieval-assets** model:

- we provide the pipeline code, configs, and smoke checks around retrieval
- users obtain or build the large retrieval assets from official/upstream
  sources
- we do **not** redistribute third-party corpora or passage collections unless
  the provenance and license are clear

### HotpotQA

The Hotpot full-wiki pipeline uses:

- question files under `data/raw/hotpot_fullwiki/`
- the local ColBERT `wiki2017` index under `data/raw/colbert_index/wiki2017`
- a local ColBERT passage collection at
  `data/raw/colbert_index/collection/collection.tsv`

Helper:

```bash
make download-index
```

This downloads:

- Hotpot fullwiki train/dev question files
- the prebuilt `wiki2017` ColBERT index from Hugging Face

It does **not** download the local Hotpot `collection.tsv` passage corpus,
it does **not** normalize `data/raw/hotpot/raw.jsonl`, and it does **not** set
up the MuSiQue retriever stack. A fresh local Hotpot rollout/eval setup needs
both the index and the collection file; otherwise use a remote retriever.

Recommended upstream source for the Hotpot passage collection:

- the official HotpotQA processed / introductory-paragraph Wikipedia release:
  https://hotpotqa.github.io/wiki-readme.html

Optional local retrieval canary once you have a compatible index and
`collection.tsv`:

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

This exercises the real local Hotpot retrieval path. It still requires query
generation access for later hops (`OPENROUTER_API_KEY` or a local inference
server).

### MuSiQue

The MuSiQue pipeline uses:

- normalized dataset records from `data/norm/musique.jsonl`
- a separate RunPod-served `wiki20m.nbits2` ColBERT retriever
- the corresponding RunPod-side passage collection (`corpus/collection.tsv`)

Dataset helper:

```bash
make download-datasets
```

By default this helper stages both normalized-dataset inputs:

- MuSiQue from Hugging Face (`dgslibisey/MuSiQue`) into
  `data/raw/musique/raw.jsonl`
- HotpotQA distractor from Hugging Face (`hotpot_qa`, `distractor`) into
  `data/raw/hotpot/raw.jsonl`

If you only want MuSiQue, run:

```bash
uv run python scripts/download_datasets.py --config configs/default.yaml --datasets musique
```

Retriever prerequisite:

- provision the MuSiQue retriever using `scripts/runpod_musique/README.md`
- the canonical MuSiQue retriever path is to build or restore a `wiki20m`
  ColBERT service plus matching `corpus/collection.tsv` from official/public
  upstream assets (DPR `psgs_w100` plus ColBERT tooling), as described in
  `scripts/runpod_musique/README.md`

Normalization helper:

```bash
make normalize
```

Current behavior of `make normalize`:

- normalizes whichever supported datasets are already staged under `data/raw/...`
- skips missing staged datasets with a log message
- does **not** download raw datasets
- does **not** build or download any retrieval index

## Canonical Entry Points

Top-level `Makefile` targets are the stable operator surface:

- `make env-check`
- `make download-datasets`
- `make download-index`
- `make normalize`
- `make rollouts`
- `make sft_prepare`
- `make sft_train`
- `make sft_eval`
- `make dpo_generate`
- `make dpo_score`
- `make dpo_prepare`
- `make dpo_train`
- `make em_compare`
- `make review_bundle`

Important:

- `make rollouts` is a generic operator entrypoint, not a paper-regime preset
- paper reproduction uses the explicit commands in `REPRO.md`, including
  overrides such as `--top-k-per-hop 3` and `--variants-per-question 5`

## Pipeline Summary

The paper pipeline is:

1. generate rollout candidates
2. select or score candidates by supervision signal
3. build SFT training data
4. train SFT adapter
5. generate IPO rollouts / score candidates / build preference pairs
6. train IPO adapter (`loss_type=ipo`)
7. run batched evaluation and downstream comparisons

Important terminology:

- script paths stay under `scripts/dpo/` for historical reasons
- the **reported preference-optimization runs are IPO runs**
- the canonical training script uses TRL `DPOTrainer` with `loss_type=ipo`

## Release Surface

The release-facing surface is:

- `README.md`
- `REPRO.md`
- `RELEASE.md`
- `configs/`
- `seer/`
- canonical scripts under `scripts/rollouts/`, `scripts/sft/`, `scripts/dpo/`,
  `scripts/eval/`, and `scripts/runpod_musique/`
- `tests/`

## Smoke Checks

Minimal local verification:

```bash
make release-smoke
uv run pytest tests/test_sft_prepare_data.py tests/test_dpo_utils.py tests/test_dpo_validate.py tests/test_normalize_datasets_script.py
```

`make release-smoke` is an offline, fixture-based check of the documented
command surface. It verifies helper CLIs, normalization, SFT prep, DPO pair
construction, training dry-runs, and review-bundle generation. It does **not**
rerun the full paper or provision the full local Hotpot retriever assets.

Optional online extension:

```bash
make release-smoke ARGS=--with-jina
```

That adds one Jina-backed raw-query scoring leg to the same fixture smoke and
requires `JINA_AI_API_KEY`.

## Training Prerequisites

The base `uv sync` environment is intended to support:

- `make release-smoke`
- the canonical unit-test slice
- offline inspection of configs, prompts, and scripts

Full training uses additional GPU-side tooling that is not required for the
smoke/test flow.

- SFT training shells out through `accelerate` into Axolotl
- IPO training uses TRL/PEFT plus the 4-bit quantization stack

Known-good training environment from the reported runs:

- Linux
- Python 3.11
- CUDA 12.4
- one known-good RunPod image:
  `axolotlai/axolotl-cloud:main-py3.11-cu124-2.6.0`
- observed worker classes in the reported runs:
  RTX 4090 and L40S

Use the paper appendix hyperparameters plus the referenced configs as the
source of truth for those runs.

## Reproducing The Paper

Use `REPRO.md` for the actual paper-result map:

- which scripts are used at each stage
- which configs correspond to the reported regimes
- what is fully local vs what requires APIs / RunPod services
- what is intentionally not redistributed

## Review Bundle

Create the anonymized code-first review bundle with:

```bash
make review_bundle
```

This writes `dist/seer_review_bundle.zip` by default and fails if it finds likely
secrets, local absolute paths, or other anonymity hazards.
