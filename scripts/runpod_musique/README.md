# RunPod Musique ColBERT Serving

Scripts in this folder support building, backing up, and serving a ColBERT wiki20m index for Musique/FrugalRAG retrieval on RunPod.

## Key Scripts

- `build_colbert_wiki20m_runpod.sh`
  - Original helper for full build flow (setup/download/convert/index/serve/smoke)
- `s3_colbert_assets.sh`
  - Backup/restore the serving assets (`index/`, `corpus/collection.tsv`, optional `serve_colbert_min.py`) to/from AWS S3
- `serve_colbert_min.py`
  - Minimal Flask HTTP server exposing:
    - `GET /health`
    - `GET /api/search?query=...&k=...`
- `start_colbert_service.sh`
  - One-command bootstrap + launch for new pods (reuses `/workspace` venv, restores from S3 if missing, installs pinned deps, starts server)
- `requirements-colbert-serve.txt`
  - Pinned serving dependencies (includes `transformers==4.41.2` to avoid ColBERT compatibility issues)
- `runpod_job.sh`
  - Remote one-shot launcher for rollout/DPO jobs on a pod (`smoke`, `rollout`, `dpo-gold`, `dpo-mmr`, `dpo-raw-jina-score`, `dpo-raw-jina-pairs`)
- `deploy_and_start_job.sh`
  - Local one-shot deploy+launch wrapper (sync latest code to pod over SSH, then run `runpod_job.sh`)

## One-Shot Job Launch (Recommended)

From your local machine:

```bash
cd seer-sufficiency
scripts/runpod_musique/deploy_and_start_job.sh \
  --ssh-host RUNPOD_HOST \
  --ssh-port RUNPOD_SSH_PORT \
  --ssh-user root \
  --ssh-key ~/.ssh/id_ed25519 \
  --job smoke \
  --openrouter-api-key "$OPENROUTER_API_KEY"
```

Full rollout:

```bash
scripts/runpod_musique/deploy_and_start_job.sh \
  --ssh-host RUNPOD_HOST \
  --ssh-port RUNPOD_SSH_PORT \
  --ssh-user root \
  --ssh-key ~/.ssh/id_ed25519 \
  --job rollout \
  --limit 10000 \
  --openrouter-api-key "$OPENROUTER_API_KEY"
```

Exact historical multi-variant replay for the paper used an external DSPy
prompt-state artifact that is **not bundled** in the review package. If you
have that asset locally, pass it explicitly with `--dspy-state PATH`.

DPO MMR:

```bash
scripts/runpod_musique/deploy_and_start_job.sh \
  --ssh-host RUNPOD_HOST \
  --ssh-port RUNPOD_SSH_PORT \
  --ssh-user root \
  --ssh-key ~/.ssh/id_ed25519 \
  --job dpo-mmr \
  --input musique_train_multihop_k3_N10000_5variants_final.jsonl \
  --openrouter-api-key "$OPENROUTER_API_KEY" \
  --jina-api-key "$JINA_AI_API_KEY"
```

If you have the historical multi-variant prompt JSON locally, pass it
explicitly with `--prompt-variants PATH`. Otherwise the job falls back to the
built-in default prompt template in `scripts/dpo/generate.py`.

Raw-Jina ablation scoring (from MMR rollout file):

```bash
scripts/runpod_musique/deploy_and_start_job.sh \
  --ssh-host RUNPOD_HOST \
  --ssh-port RUNPOD_SSH_PORT \
  --ssh-user root \
  --ssh-key ~/.ssh/id_ed25519 \
  --job dpo-raw-jina-score \
  --input data/rollouts/musique_dpo_rollouts_mmr.jsonl \
  --dpo-output data/rollouts/musique_dpo_rollouts_raw_jina_scored.jsonl \
  --batch-size 200 \
  --jina-concurrency 8 \
  --jina-api-key "$JINA_AI_API_KEY"
```

Raw-Jina pair build:

```bash
scripts/runpod_musique/deploy_and_start_job.sh \
  --ssh-host RUNPOD_HOST \
  --ssh-port RUNPOD_SSH_PORT \
  --ssh-user root \
  --ssh-key ~/.ssh/id_ed25519 \
  --job dpo-raw-jina-pairs \
  --input data/rollouts/musique_dpo_rollouts_raw_jina_scored.jsonl \
  --scoring raw_jina_maxmean \
  --pair-mode exhaustive \
  --min-gap 0.01035 \
  --pairs-output data/seer_sft/musique_dpo_raw_jina_maxmean_mingap01035_exhaustive.jsonl
```

## Prompt Assets

The RunPod helper scripts support two modes:

- default helper mode: use the built-in prompt template bundled in the repo
- exact historical prompt replay: pass external DSPy prompt-state / prompt-variant
  JSON files with `--dspy-state` or `--prompt-variants`

Those external JSON prompt artifacts are **not bundled** in the review package.
They are only needed if you want to mirror the historical prompt-source setup
more exactly than the built-in fallback.

The remote runner always:
- reuses/creates `.venv`
- installs `pip install -e .` + `python-dotenv`
- checks `http://127.0.0.1:8000/health`
- launches with nohup and prints the exact `tail -f` log path

## Recommended Startup (Fresh Pod)

1. Upload these files to `/workspace/musique_colbert_build/incoming/`:
   - `s3_colbert_assets.sh`
   - `serve_colbert_min.py`
   - `start_colbert_service.sh`
   - `requirements-colbert-serve.txt`
2. Set AWS credentials and `S3_URI` (if restoring from S3).
3. Run:

```bash
bash /workspace/musique_colbert_build/incoming/start_colbert_service.sh
```

This script will:
- install AWS CLI v2 if `S3_URI` is set and `aws` is missing
- create/reuse `/workspace/musique_colbert_build/venv`
- restore `index/` + `corpus/collection.tsv` from S3 if missing
- install pinned serving dependencies + editable ColBERT repo
- start the ColBERT HTTP server on `0.0.0.0:8000`

## Validation

Local pod shell:

```bash
curl -s http://127.0.0.1:8000/health; echo
curl -s "http://127.0.0.1:8000/api/search?query=South%20Park%20The%20Hobbit%20Trey%20Parker&k=3" | head -c 500; echo
```

Remote machine (RunPod public TCP port mapped to container `8000`):

```bash
curl -s http://RUNPOD_IP:PUBLIC_PORT/health
```

## Operational Notes

- RunPod’s external public port is usually **not** the same as container port `8000`.
- A fresh venv does **not** inherit PyTorch from the base image; `torch` must be installed in the venv.
- ColBERT `0.2.22` can break with `transformers>=5`; use the pinned requirements file.
- The `wiki20m.nbits2` index load is RAM-heavy (observed ~113 GiB RSS during serving).
- You can run index + query-generation model service on one pod if resource budget allows.
- Keep separate ports, e.g. ColBERT on `127.0.0.1:8000` and inference service on `127.0.0.1:8001`.
- Then run evaluation with `--retrieval-url http://127.0.0.1:8000` and `--inference-url http://127.0.0.1:8001`.
- In practice, this usually requires a large-memory/compute pod, typically 80GB+ memory class for the wiki20m index plus model process.
