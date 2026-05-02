# Rollout Generation

Canonical entrypoint:

```bash
uv run python scripts/rollouts/generate.py --help
```

This stage generates multi-hop query-rewrite rollout variants over ColBERT
retrieval.

## What It Consumes

- `configs/default.yaml`
- dataset inputs:
  - Hotpot fullwiki questions under `data/raw/hotpot_fullwiki/`
  - MuSiQue normalized data under `data/norm/musique.jsonl`
- retrieval backend:
  - Hotpot: local `wiki2017` index plus local `collection.tsv`
  - MuSiQue: remote RunPod retriever via `--remote-url`

## Hotpot Example

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

For fresh question slices, use `--offset`:

```bash
uv run python scripts/rollouts/generate.py \
  --config configs/default.yaml \
  --dataset hotpot \
  --split train \
  --offset 5000 \
  --limit 10000 \
  --top-k-per-hop 3 \
  --variants-per-question 5 \
  --output data/rollouts/hotpot_train_multihop_k3_N10000_5variants.jsonl
```

## MuSiQue Example

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

MuSiQue uses variable hop count. The retriever setup itself is documented in
`scripts/runpod_musique/README.md`.

## Important Flags

- `--dataset {hotpot,musique,both}`
- `--split {train,dev,test}`
- `--limit`
- `--offset`
- `--top-k-per-hop`
- `--variants-per-question`
- `--model`
- `--inference-url`
- `--remote-url`

## Notes

- Hotpot fullwiki rollouts use `data/raw/hotpot_fullwiki`, not
  `data/raw/hotpot/raw.jsonl`.
- Hotpot local rollout generation also needs
  `data/raw/colbert_index/collection/collection.tsv`; the HF index download
  alone is not enough for local retrieval.
- For the Hotpot passage collection, prefer the official processed /
  introductory-paragraph Wikipedia release from HotpotQA:
  https://hotpotqa.github.io/wiki-readme.html
- MuSiQue rollouts require `data/norm/musique.jsonl`; `make normalize` must run
  after you stage the official raw MuSiQue file.
- MuSiQue retrieval assets are intentionally external to the default bundle; use
  `scripts/runpod_musique/README.md` to build or restore a compatible service.
- `--offset` is the canonical way to create non-overlapping question ranges for
  later stages such as 10K or iteration runs.
- Output file names are not fixed by the script. Use explicit `--output` paths
  and keep naming consistent with the experiment regime.
