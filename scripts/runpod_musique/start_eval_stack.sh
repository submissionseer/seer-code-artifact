#!/usr/bin/env bash
set -euo pipefail

log() { echo "[ $(date '+%F %T') ] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

ROOT="${1:-/workspace/seer-sufficiency}"
BASE_MODEL="${BASE_MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
ADAPTER="${ADAPTER:-/workspace/data/seer_sft/qlora-out-gold-musique-10k-k3}"
PROMPT_FORMAT="${PROMPT_FORMAT:-alpaca}"
INFERENCE_PORT="${INFERENCE_PORT:-8002}"
COLBERT_PORT="${COLBERT_PORT:-8000}"
COLBERT_LOG="${COLBERT_LOG:-$ROOT/.runlogs/colbert_service.log}"
INFER_LOG="${INFER_LOG:-$ROOT/.runlogs/infer_musique.log}"
START_COLBERT="${START_COLBERT:-1}"
WAIT_SECONDS="${WAIT_SECONDS:-360}"
WAIT_SLEEP_SECONDS="${WAIT_SLEEP_SECONDS:-2}"
COLBERT_SERVICE_SCRIPT="${COLBERT_SERVICE_SCRIPT:-$ROOT/scripts/runpod_musique/start_colbert_service.sh}"
COLBERT_STATUS_URL="${COLBERT_STATUS_URL:-http://127.0.0.1:$COLBERT_PORT/health}"
REQUIRE_FASTAPI_DEPS="${REQUIRE_FASTAPI_DEPS:-1}"
VENV_DIR="${VENV_DIR:-$ROOT/.venv}"
HF_HOME_DEFAULT="${HF_HOME_DEFAULT:-/workspace/musique_colbert_build/hf_cache}"

wait_for_http() {
  local url=$1
  local label=$2
  local elapsed=0
  while (( elapsed < WAIT_SECONDS )); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      log "$label ready"
      return 0
    fi
    if (( elapsed % 30 == 0 )); then
      log "Still waiting for $label (${elapsed}s/${WAIT_SECONDS})"
    fi
    sleep "$WAIT_SLEEP_SECONDS"
    elapsed=$((elapsed + WAIT_SLEEP_SECONDS))
  done
  log "$label not ready after ${WAIT_SECONDS}s"
  return 1
}

choose_open_port() {
  local requested=$1
  local max_port=$((requested + 100))
  for p in $(seq "$requested" "$max_port"); do
    if ! ss -ltn | awk '{print $4}' | sed -E 's/.*:([0-9]+)$/\1/' | grep -qx "$p"; then
      if (( p != requested )); then
        log "Requested port $requested is busy; using $p instead"
      fi
      echo "$p"
      return 0
    fi
  done
  die "No open port found in range $requested-$max_port"
}

if [[ ! -d "$ROOT" ]]; then
  die "Repo root not found: $ROOT"
fi

mkdir -p "$ROOT/.runlogs" "$ROOT/data/seer_sft"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  log "Creating venv at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

log "Installing base project deps into $VENV_DIR"
"$VENV_DIR/bin/pip" install -U pip >/dev/null
"$VENV_DIR/bin/pip" install -e "$ROOT" >/dev/null
"$VENV_DIR/bin/pip" install python-dotenv >/dev/null
"$VENV_DIR/bin/pip" install -q peft >/dev/null

if [[ "$REQUIRE_FASTAPI_DEPS" == "1" ]]; then
  log "Installing FastAPI + Uvicorn"
  "$VENV_DIR/bin/pip" install -q fastapi "uvicorn[standard]" >/dev/null
fi

if [[ "$START_COLBERT" == "1" ]]; then
  if [[ ! -x "$COLBERT_SERVICE_SCRIPT" ]]; then
    die "ColBERT service script not found: $COLBERT_SERVICE_SCRIPT"
  fi
  log "Launching ColBERT service script: $COLBERT_SERVICE_SCRIPT"
  (cd "$ROOT" && nohup bash "$COLBERT_SERVICE_SCRIPT" > "$COLBERT_LOG" 2>&1 < /dev/null &)
  log "Waiting up to ${WAIT_SECONDS}s for ColBERT /health on port $COLBERT_PORT"
  if ! wait_for_http "$COLBERT_STATUS_URL" "ColBERT"; then
    tail -n 80 "$COLBERT_LOG" || true
    die "ColBERT health check failed"
  fi
else
  log "Skipping ColBERT start (START_COLBERT=0)"
fi

INFERENCE_PORT="$(choose_open_port "$INFERENCE_PORT")"

if [[ ! -e "$ADAPTER" ]]; then
  die "Adapter path not found: $ADAPTER"
fi
if [[ ! -d "$ADAPTER" && ! -f "$ADAPTER" ]]; then
  die "Adapter path not found: $ADAPTER"
fi

export HF_HOME="${HF_HOME:-$HF_HOME_DEFAULT}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"

log "Launching inference server (HF_HOME=$HF_HOME)"
nohup env \
  HF_HOME="$HF_HOME" \
  HUGGINGFACE_HUB_CACHE="$HUGGINGFACE_HUB_CACHE" \
  HF_TOKEN="${HF_TOKEN:-}" \
  "$VENV_DIR/bin/python" -u "$ROOT/seer/inference_server.py" \
  --base-model "$BASE_MODEL" \
  --adapter "$ADAPTER" \
  --prompt-format "$PROMPT_FORMAT" \
  --port "$INFERENCE_PORT" \
  > "$INFER_LOG" 2>&1 < /dev/null &
INFER_PID=$!
echo "$INFER_PID" > "$ROOT/.runlogs/infer_musique.pid"

if ! kill -0 "$INFER_PID" 2>/dev/null; then
  log "Inference exited. Last lines:"
  tail -n 120 "$INFER_LOG" || true
  die "Inference server failed to start"
fi

if ! wait_for_http "http://127.0.0.1:$INFERENCE_PORT/health" "Inference"; then
  tail -n 120 "$INFER_LOG" || true
  die "Inference health check failed"
fi

log "Launched colbert log: $COLBERT_LOG"
log "Launched inference log: $INFER_LOG"
log "Inference PID: $INFER_PID"
log "Quick checks:"
log "curl -s http://127.0.0.1:$COLBERT_PORT/health"
log "curl -s http://127.0.0.1:$INFERENCE_PORT/health"
