#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run one rollout/DPO job on RunPod with automatic venv bootstrap.

Usage:
  runpod_job.sh --job <smoke|rollout|dpo-gold|dpo-mmr|dpo-raw-jina-score|dpo-raw-jina-pairs> [options]

Core options:
  --job NAME                     Job type (required)
  --root PATH                    Repo root on pod (default: /workspace/seer-sufficiency)
  --remote-url URL               ColBERT URL (default: http://127.0.0.1:8000)
  --foreground                   Run in foreground (default: nohup background)
  --openrouter-api-key KEY       Set OPENROUTER_API_KEY for this run
  --jina-api-key KEY             Set JINA_AI_API_KEY for this run

Rollout options (smoke/rollout):
  --dataset NAME                 Dataset (default: musique)
  --split NAME                   Split (default: train)
  --limit N                      Number of base questions
  --offset N                     Offset (default: 0)
  --model NAME                   LLM model (default: meta-llama/llama-3-8b-instruct)
  --top-k-per-hop N              Retrieved docs/hop (default: 3)
  --variants-per-question N      Prompt variants/question (default: 5)
  --prompt-variant-mode MODE     (default: all)
  --num-threads N                Worker threads (default: 4)
  --dspy-state PATH              Prompt state JSON path
  --rollout-output PATH          Output JSONL path

DPO options (dpo-gold/dpo-mmr):
  --input PATH                   Rollout variants input JSONL
  --dpo-output PATH              Output JSONL path
  --prompt-variants PATH         Prompt variants JSON
  --context-policy MODE          weighted|best (default: weighted)
  --batch-size N                 Questions/batch (default: 100)
  --concurrency N                OpenRouter concurrency (default: 20)
  --seed N                       RNG seed (default: 42)

MMR-only options:
  --requirements-out PATH        Requirements JSON output
  --requirements-model NAME      (default: openai/gpt-4o-mini)
  --requirements-concurrency N   (default: 20)
  --jina-model NAME              (default: jina-reranker-v3)
  --jina-concurrency N           (default: 16)

Raw-Jina scoring options (dpo-raw-jina-score):
  --cache-dir PATH               Jina cache dir
  --topk-mean N                  Top-k for mean score (default: 3)

Raw-Jina pair options (dpo-raw-jina-pairs):
  --scoring MODE                 raw_jina_maxmean|raw_jina_max (default: raw_jina_maxmean)
  --min-gap FLOAT                Minimum pair gap (default: 0.01035)
  --pair-mode MODE               adjacent|exhaustive (default: exhaustive)
  --max-pairs-per-group N        Max pairs/question-hop (default: 20)
  --pairs-output PATH            Pair output JSONL
EOF
}

log() { echo "[$(date '+%F %T')] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

abs_path() {
  local root="$1"
  local p="$2"
  if [[ "$p" = /* ]]; then
    printf '%s\n' "$p"
  else
    printf '%s/%s\n' "$root" "$p"
  fi
}

resolve_existing_path() {
  local default_path="$1"
  shift
  if [[ -n "$default_path" && -f "$default_path" ]]; then
    printf '%s\n' "$default_path"
    return 0
  fi
  local candidate
  for candidate in "$@"; do
    if [[ -n "$candidate" && -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

ensure_venv() {
  local root="$1"
  local venv="$root/.venv"
  if [[ ! -x "$venv/bin/python" ]]; then
    log "Creating venv at $venv"
    python3 -m venv "$venv"
  fi
  "$venv/bin/pip" install -U pip
  "$venv/bin/pip" install -e "$root"
  "$venv/bin/pip" install python-dotenv
}

wait_for_index() {
  local remote_url="$1"
  local tries=30
  local i
  for i in $(seq 1 "$tries"); do
    if curl -fsS --max-time 5 "$remote_url/health" >/dev/null; then
      log "Index health check ok: $remote_url/health"
      return 0
    fi
    sleep 2
  done
  return 1
}

launch_job() {
  local root="$1"
  local log_file="$2"
  local foreground="$3"
  shift 3
  local cmd=("$@")

  mkdir -p "$(dirname "$log_file")"
  log "Command: ${cmd[*]}"

  if [[ "$foreground" == "1" ]]; then
    exec "${cmd[@]}"
  fi

  nohup "${cmd[@]}" >"$log_file" 2>&1 &
  local pid=$!
  sleep 2
  if ! kill -0 "$pid" 2>/dev/null; then
    log "Job exited immediately. Last log lines:"
    tail -n 80 "$log_file" || true
    return 1
  fi
  log "Started PID=$pid"
  log "Log: $log_file"
  log "Tail: tail -f $log_file"
}

JOB=""
ROOT="/workspace/seer-sufficiency"
REMOTE_URL="http://127.0.0.1:8000"
FOREGROUND="0"

DATASET="musique"
SPLIT="train"
LIMIT=""
OFFSET="0"
MODEL="meta-llama/llama-3-8b-instruct"
TOP_K="3"
VARIANTS_PER_QUESTION="5"
PROMPT_VARIANT_MODE="all"
NUM_THREADS="4"
DSPY_STATE=""
ROLLOUT_OUTPUT=""

INPUT_PATH=""
DPO_OUTPUT=""
PROMPT_VARIANTS=""
CONTEXT_POLICY="weighted"
BATCH_SIZE="100"
CONCURRENCY="20"
SEED="42"

REQUIREMENTS_OUT=""
REQUIREMENTS_MODEL="openai/gpt-4o-mini"
REQUIREMENTS_CONCURRENCY="20"
JINA_MODEL="jina-reranker-v3"
JINA_CONCURRENCY="16"

CACHE_DIR=""
TOPK_MEAN="3"
PAIR_SCORING="raw_jina_maxmean"
MIN_GAP="0.01035"
PAIR_MODE="exhaustive"
MAX_PAIRS_PER_GROUP="20"
PAIRS_OUTPUT=""

while (($#)); do
  case "$1" in
    --job) JOB="${2:-}"; shift 2 ;;
    --root) ROOT="${2:-}"; shift 2 ;;
    --remote-url) REMOTE_URL="${2:-}"; shift 2 ;;
    --foreground) FOREGROUND="1"; shift ;;
    --openrouter-api-key) export OPENROUTER_API_KEY="${2:-}"; shift 2 ;;
    --jina-api-key) export JINA_AI_API_KEY="${2:-}"; shift 2 ;;

    --dataset) DATASET="${2:-}"; shift 2 ;;
    --split) SPLIT="${2:-}"; shift 2 ;;
    --limit) LIMIT="${2:-}"; shift 2 ;;
    --offset) OFFSET="${2:-}"; shift 2 ;;
    --model) MODEL="${2:-}"; shift 2 ;;
    --top-k-per-hop) TOP_K="${2:-}"; shift 2 ;;
    --variants-per-question) VARIANTS_PER_QUESTION="${2:-}"; shift 2 ;;
    --prompt-variant-mode) PROMPT_VARIANT_MODE="${2:-}"; shift 2 ;;
    --num-threads) NUM_THREADS="${2:-}"; shift 2 ;;
    --dspy-state) DSPY_STATE="${2:-}"; shift 2 ;;
    --rollout-output) ROLLOUT_OUTPUT="${2:-}"; shift 2 ;;

    --input) INPUT_PATH="${2:-}"; shift 2 ;;
    --dpo-output) DPO_OUTPUT="${2:-}"; shift 2 ;;
    --prompt-variants) PROMPT_VARIANTS="${2:-}"; shift 2 ;;
    --context-policy) CONTEXT_POLICY="${2:-}"; shift 2 ;;
    --batch-size) BATCH_SIZE="${2:-}"; shift 2 ;;
    --concurrency) CONCURRENCY="${2:-}"; shift 2 ;;
    --seed) SEED="${2:-}"; shift 2 ;;

    --requirements-out) REQUIREMENTS_OUT="${2:-}"; shift 2 ;;
    --requirements-model) REQUIREMENTS_MODEL="${2:-}"; shift 2 ;;
    --requirements-concurrency) REQUIREMENTS_CONCURRENCY="${2:-}"; shift 2 ;;
    --jina-model) JINA_MODEL="${2:-}"; shift 2 ;;
    --jina-concurrency) JINA_CONCURRENCY="${2:-}"; shift 2 ;;
    --cache-dir) CACHE_DIR="${2:-}"; shift 2 ;;
    --topk-mean) TOPK_MEAN="${2:-}"; shift 2 ;;
    --scoring) PAIR_SCORING="${2:-}"; shift 2 ;;
    --min-gap) MIN_GAP="${2:-}"; shift 2 ;;
    --pair-mode) PAIR_MODE="${2:-}"; shift 2 ;;
    --max-pairs-per-group) MAX_PAIRS_PER_GROUP="${2:-}"; shift 2 ;;
    --pairs-output) PAIRS_OUTPUT="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown arg: $1 (use --help)" ;;
  esac
done

[[ -n "$JOB" ]] || die "--job is required"
[[ -d "$ROOT" ]] || die "Repo root not found: $ROOT"

ensure_venv "$ROOT"
if [[ "$JOB" == "smoke" || "$JOB" == "rollout" || "$JOB" == "dpo-gold" || "$JOB" == "dpo-mmr" ]]; then
  wait_for_index "$REMOTE_URL" || die "Retriever health check failed: $REMOTE_URL/health"
fi

PY="$ROOT/.venv/bin/python"
LOG_DIR="$ROOT/.runlogs"
mkdir -p "$LOG_DIR"

if [[ "$JOB" == "smoke" || "$JOB" == "rollout" || "$JOB" == "dpo-gold" || "$JOB" == "dpo-mmr" ]]; then
  if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    die "OPENROUTER_API_KEY is required for job=$JOB"
  fi
fi

if [[ "$JOB" == "smoke" || "$JOB" == "rollout" ]]; then
  if [[ -z "$LIMIT" ]]; then
    if [[ "$JOB" == "smoke" ]]; then
      LIMIT="10"
    else
      LIMIT="10000"
    fi
  fi

  if [[ -z "$DSPY_STATE" ]]; then
    DSPY_STATE="$(resolve_existing_path "" \
      "$ROOT/data/prompts/hotpot/query_prompt_state_candidates_leret_llama3_8b_merged.json" \
      "$ROOT/query_prompt_state_candidates_leret_llama3_8b_merged.json")" \
      || true
  else
    DSPY_STATE="$(abs_path "$ROOT" "$DSPY_STATE")"
  fi

  if [[ -z "$DSPY_STATE" && "$VARIANTS_PER_QUESTION" != "1" ]]; then
    echo "No DSPy state supplied; falling back to the built-in default rollout prompt." >&2
    echo "For exact historical multi-variant replay, pass --dspy-state <path>." >&2
    VARIANTS_PER_QUESTION="1"
  fi

  if [[ -z "$ROLLOUT_OUTPUT" ]]; then
    if [[ "$JOB" == "smoke" ]]; then
      ROLLOUT_OUTPUT="$ROOT/data/rollouts/smoke_${DATASET}_10x5.jsonl"
    else
      ROLLOUT_OUTPUT="$ROOT/data/rollouts/${DATASET}_${SPLIT}_multihop_k${TOP_K}_N${LIMIT}_${VARIANTS_PER_QUESTION}variants.jsonl"
    fi
  else
    ROLLOUT_OUTPUT="$(abs_path "$ROOT" "$ROLLOUT_OUTPUT")"
  fi

  mkdir -p "$(dirname "$ROLLOUT_OUTPUT")"
  LOG_FILE="$LOG_DIR/${JOB}_${DATASET}_${SPLIT}.log"

  CMD=(
    "$PY" -u "$ROOT/scripts/rollouts/generate.py"
    --config "$ROOT/configs/default.yaml"
    --dataset "$DATASET"
    --split "$SPLIT"
    --offset "$OFFSET"
    --limit "$LIMIT"
    --model "$MODEL"
    --prompt-variant-mode "$PROMPT_VARIANT_MODE"
    --variants-per-question "$VARIANTS_PER_QUESTION"
    --top-k-per-hop "$TOP_K"
    --remote-url "$REMOTE_URL"
    --num-threads "$NUM_THREADS"
    --output "$ROLLOUT_OUTPUT"
  )

  if [[ -n "$DSPY_STATE" ]]; then
    CMD+=(--dspy-state "$DSPY_STATE")
  fi

  launch_job "$ROOT" "$LOG_FILE" "$FOREGROUND" "${CMD[@]}"
  exit 0
fi

if [[ "$JOB" == "dpo-gold" || "$JOB" == "dpo-mmr" ]]; then
  if [[ -z "$INPUT_PATH" ]]; then
    INPUT_PATH="$(resolve_existing_path "" \
      "$ROOT/musique_train_multihop_k3_N10000_5variants_final.jsonl" \
      "$ROOT/data/rollouts/musique_train_multihop_k3_N10000_5variants_final.jsonl")" \
      || die "Missing DPO input rollout file. Pass --input <path>."
  else
    INPUT_PATH="$(abs_path "$ROOT" "$INPUT_PATH")"
  fi

  if [[ -z "$PROMPT_VARIANTS" ]]; then
    PROMPT_VARIANTS="$(resolve_existing_path "" \
      "$ROOT/query_prompt_variants_dspy.json" \
      "$ROOT/data/prompts/hotpot/query_prompt_variants_dspy.json")" \
      || true
  else
    PROMPT_VARIANTS="$(abs_path "$ROOT" "$PROMPT_VARIANTS")"
  fi

  if [[ -z "$DPO_OUTPUT" ]]; then
    if [[ "$JOB" == "dpo-gold" ]]; then
      DPO_OUTPUT="$ROOT/data/dpo/musique_dpo_rollouts_gold.jsonl"
    else
      DPO_OUTPUT="$ROOT/data/dpo/musique_dpo_rollouts_mmr.jsonl"
    fi
  else
    DPO_OUTPUT="$(abs_path "$ROOT" "$DPO_OUTPUT")"
  fi

  mkdir -p "$(dirname "$DPO_OUTPUT")"
  local_mode="gold"
  if [[ "$JOB" == "dpo-mmr" ]]; then
    local_mode="mmr"
    [[ -n "${JINA_AI_API_KEY:-}" ]] || die "JINA_AI_API_KEY is required for dpo-mmr"
    if [[ -z "$REQUIREMENTS_OUT" ]]; then
      REQUIREMENTS_OUT="$ROOT/data/dpo/musique_requirements_for_mmr.json"
    else
      REQUIREMENTS_OUT="$(abs_path "$ROOT" "$REQUIREMENTS_OUT")"
    fi
  fi

  LOG_FILE="$LOG_DIR/${JOB}_musique.log"
  CMD=(
    "$PY" -u "$ROOT/scripts/dpo/generate.py"
    --input "$INPUT_PATH"
    --output "$DPO_OUTPUT"
    --config "$ROOT/configs/default.yaml"
    --model "$MODEL"
    --temperature 0.0
    --top-k-per-hop "$TOP_K"
    --context-select "$local_mode"
    --context-policy "$CONTEXT_POLICY"
    --batch-size "$BATCH_SIZE"
    --concurrency "$CONCURRENCY"
    --seed "$SEED"
    --remote-url "$REMOTE_URL"
  )

  if [[ -n "$PROMPT_VARIANTS" ]]; then
    CMD+=(--prompt-variants "$PROMPT_VARIANTS")
  else
    echo "No prompt-variants JSON supplied; using built-in default prompt template in scripts/dpo/generate.py." >&2
    echo "For exact historical prompt replay, pass --prompt-variants <path>." >&2
  fi

  if [[ "$JOB" == "dpo-mmr" ]]; then
    CMD+=(
      --requirements-model "$REQUIREMENTS_MODEL"
      --requirements-concurrency "$REQUIREMENTS_CONCURRENCY"
      --auto-requirements-out "$REQUIREMENTS_OUT"
      --jina-model "$JINA_MODEL"
      --jina-concurrency "$JINA_CONCURRENCY"
    )
  fi

  launch_job "$ROOT" "$LOG_FILE" "$FOREGROUND" "${CMD[@]}"
  exit 0
fi

if [[ "$JOB" == "dpo-raw-jina-score" ]]; then
  [[ -n "${JINA_AI_API_KEY:-}" ]] || die "JINA_AI_API_KEY is required for dpo-raw-jina-score"

  if [[ -z "$INPUT_PATH" ]]; then
    INPUT_PATH="$(resolve_existing_path "" \
      "$ROOT/data/rollouts/musique_dpo_rollouts_mmr.jsonl" \
      "$ROOT/musique_dpo_rollouts_mmr.jsonl")" \
      || die "Missing scored-rollout input. Pass --input <path>."
  else
    INPUT_PATH="$(abs_path "$ROOT" "$INPUT_PATH")"
  fi

  if [[ -z "$DPO_OUTPUT" ]]; then
    DPO_OUTPUT="$ROOT/data/rollouts/musique_dpo_rollouts_raw_jina_scored.jsonl"
  else
    DPO_OUTPUT="$(abs_path "$ROOT" "$DPO_OUTPUT")"
  fi

  if [[ -z "$CACHE_DIR" ]]; then
    CACHE_DIR="$ROOT/data/dpo_jina_cache_raw_query_musique"
  else
    CACHE_DIR="$(abs_path "$ROOT" "$CACHE_DIR")"
  fi

  mkdir -p "$(dirname "$DPO_OUTPUT")"
  mkdir -p "$CACHE_DIR"
  LOG_FILE="$LOG_DIR/${JOB}_musique.log"
  CMD=(
    "$PY" -u "$ROOT/scripts/dpo/score.py"
    --input "$INPUT_PATH"
    --output "$DPO_OUTPUT"
    --mode raw_jina
    --jina-cache-dir "$CACHE_DIR"
    --jina-model "$JINA_MODEL"
    --concurrency "$JINA_CONCURRENCY"
    --batch-size "$BATCH_SIZE"
    --topk-mean "$TOPK_MEAN"
  )
  if [[ -n "$LIMIT" ]]; then
    CMD+=(--limit "$LIMIT")
  fi

  launch_job "$ROOT" "$LOG_FILE" "$FOREGROUND" "${CMD[@]}"
  exit 0
fi

if [[ "$JOB" == "dpo-raw-jina-pairs" ]]; then
  case "$PAIR_SCORING" in
    raw_jina_maxmean|raw_jina_max) ;;
    *) die "--scoring must be raw_jina_maxmean or raw_jina_max" ;;
  esac

  if [[ -z "$INPUT_PATH" ]]; then
    INPUT_PATH="$(resolve_existing_path "" \
      "$ROOT/data/rollouts/musique_dpo_rollouts_raw_jina_scored.jsonl" \
      "$ROOT/musique_dpo_rollouts_raw_jina_scored.jsonl")" \
      || die "Missing raw-jina-scored rollout input. Pass --input <path>."
  else
    INPUT_PATH="$(abs_path "$ROOT" "$INPUT_PATH")"
  fi

  if [[ -z "$PAIRS_OUTPUT" ]]; then
    gap_clean="${MIN_GAP/./}"
    PAIRS_OUTPUT="$ROOT/data/seer_sft/musique_dpo_${PAIR_SCORING}_mingap${gap_clean}_${PAIR_MODE}.jsonl"
  else
    PAIRS_OUTPUT="$(abs_path "$ROOT" "$PAIRS_OUTPUT")"
  fi

  mkdir -p "$(dirname "$PAIRS_OUTPUT")"
  LOG_FILE="$LOG_DIR/${JOB}_musique.log"
  CMD=(
    "$PY" -u "$ROOT/scripts/dpo/prepare.py"
    --input "$INPUT_PATH"
    --output "$PAIRS_OUTPUT"
    --scoring "$PAIR_SCORING"
    --pair-mode "$PAIR_MODE"
    --min-gap "$MIN_GAP"
    --max-pairs-per-group "$MAX_PAIRS_PER_GROUP"
    --seed "$SEED"
  )

  launch_job "$ROOT" "$LOG_FILE" "$FOREGROUND" "${CMD[@]}"
  exit 0
fi

die "Unsupported --job: $JOB"
