#!/usr/bin/env bash
set -euo pipefail

# RunPod helper for building/serving a FrugalRAG-compatible ColBERT wiki20M index.
#
# Usage examples:
#   bash build_colbert_wiki20m_runpod.sh setup
#   bash build_colbert_wiki20m_runpod.sh download
#   bash build_colbert_wiki20m_runpod.sh convert
#   bash build_colbert_wiki20m_runpod.sh index-help
#   bash build_colbert_wiki20m_runpod.sh index
#   bash build_colbert_wiki20m_runpod.sh serve
#   bash build_colbert_wiki20m_runpod.sh smoke
#
# Override defaults via env vars if needed:
#   BUILD_ROOT=/workspace/musique_colbert_build
#   INDEX_NAME=wiki20m.nbits2
#   PORT=8000

BUILD_ROOT="${BUILD_ROOT:-/workspace/musique_colbert_build}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-$BUILD_ROOT/downloads}"
CORPUS_DIR="${CORPUS_DIR:-$BUILD_ROOT/corpus}"
INDEX_ROOT="${INDEX_ROOT:-$BUILD_ROOT/index}"
LOG_DIR="${LOG_DIR:-$BUILD_ROOT/logs}"
CODE_DIR="${CODE_DIR:-$BUILD_ROOT/code}"
MODELS_DIR="${MODELS_DIR:-$BUILD_ROOT/models}"
VENV_DIR="${VENV_DIR:-$BUILD_ROOT/venv}"

FRUGALRAG_REPO="$CODE_DIR/FrugalRAG"
COLBERT_REPO="$CODE_DIR/ColBERT"

PSGS_URL="${PSGS_URL:-https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz}"
PSGS_GZ="$DOWNLOAD_DIR/psgs_w100.tsv.gz"
COLLECTION_TSV="$CORPUS_DIR/collection.tsv"

INDEX_NAME="${INDEX_NAME:-wiki20m.nbits2}"
COLBERT_CHECKPOINT="${COLBERT_CHECKPOINT:-colbert-ir/colbertv2.0}"
PORT="${PORT:-8000}"

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONVERTER_PY="$SELF_DIR/convert_psgs_to_colbert_collection.py"

ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*"; }

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    exit 1
  }
}

ensure_dirs() {
  mkdir -p "$DOWNLOAD_DIR" "$CORPUS_DIR" "$INDEX_ROOT" "$LOG_DIR" "$CODE_DIR" "$MODELS_DIR"
}

ensure_venv() {
  ensure_dirs
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    log "Creating virtualenv at $VENV_DIR"
    python -m venv "$VENV_DIR"
  fi
}

py() {
  "$VENV_DIR/bin/python" "$@"
}

pipv() {
  "$VENV_DIR/bin/pip" "$@"
}

setup_step() {
  ensure_dirs
  log "Installing system dependencies..."
  apt-get update
  apt-get install -y git wget pigz pv tmux jq

  ensure_venv
  log "Installing Python dependencies..."
  pipv install --upgrade pip setuptools wheel
  pipv install flask python-dotenv ujson requests datasets jsonlines regex

  log "Cloning FrugalRAG and ColBERT..."
  if [[ ! -d "$FRUGALRAG_REPO/.git" ]]; then
    git clone https://github.com/microsoft/FrugalRAG.git "$FRUGALRAG_REPO"
  else
    log "FrugalRAG already present, skipping clone."
  fi
  if [[ ! -d "$COLBERT_REPO/.git" ]]; then
    git clone https://github.com/stanford-futuredata/ColBERT.git "$COLBERT_REPO"
  else
    log "ColBERT already present, skipping clone."
  fi

  log "Installing ColBERT (editable)..."
  pipv install -e "$COLBERT_REPO"

  log "Sanity check (torch + ColBERT import)..."
  py - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
from colbert import Searcher
print("ColBERT import OK")
PY

  log "SETUP COMPLETE"
}

download_step() {
  ensure_dirs
  require_cmd wget
  log "Downloading psgs_w100.tsv.gz to $PSGS_GZ"
  /usr/bin/time -v wget -c --show-progress "$PSGS_URL" -O "$PSGS_GZ" \
    2>&1 | tee "$LOG_DIR/wget_psgs_w100.log"
  log "DOWNLOAD COMPLETE: $(ls -lh "$PSGS_GZ" | awk '{print $5, $9}')"
}

convert_step() {
  ensure_dirs
  [[ -f "$PSGS_GZ" ]] || { echo "Missing $PSGS_GZ"; exit 1; }
  [[ -f "$CONVERTER_PY" ]] || { echo "Missing converter script at $CONVERTER_PY"; exit 1; }
  log "Converting $PSGS_GZ -> $COLLECTION_TSV (FrugalRAG-compatible: title | text)"
  /usr/bin/time -v python "$CONVERTER_PY" "$PSGS_GZ" "$COLLECTION_TSV" \
    2>&1 | tee "$LOG_DIR/convert_collection.log"
  log "CONVERT COMPLETE: $(ls -lh "$COLLECTION_TSV" | awk '{print $5, $9}')"
}

index_help_step() {
  ensure_venv
  log "ColBERT index CLI help (used to confirm syntax on this image)"
  py -m colbert.index --help 2>&1 | tee "$LOG_DIR/colbert_index_help.log" || true
}

index_step() {
  ensure_dirs
  ensure_venv
  [[ -f "$COLLECTION_TSV" ]] || { echo "Missing $COLLECTION_TSV"; exit 1; }
  log "Starting ColBERT indexing (index=$INDEX_NAME, checkpoint=$COLBERT_CHECKPOINT)"
  log "Logs: $LOG_DIR/index_${INDEX_NAME}.log"
  /usr/bin/time -v "$VENV_DIR/bin/python" -m colbert.index \
    --collection "$COLLECTION_TSV" \
    --checkpoint "$COLBERT_CHECKPOINT" \
    --index_name "$INDEX_NAME" \
    --index_root "$INDEX_ROOT" \
    --doc_maxlen 180 \
    --nbits 2 \
    --amp \
    2>&1 | tee "$LOG_DIR/index_${INDEX_NAME}.log"
  log "INDEX BUILD COMPLETE"
  du -sh "$INDEX_ROOT"/* 2>/dev/null || true
}

serve_step() {
  ensure_venv
  [[ -d "$FRUGALRAG_REPO" ]] || { echo "Missing FrugalRAG repo at $FRUGALRAG_REPO"; exit 1; }
  [[ -f "$COLLECTION_TSV" ]] || { echo "Missing $COLLECTION_TSV"; exit 1; }
  log "Starting FrugalRAG ColBERT server on port $PORT"
  log "Index root: $INDEX_ROOT"
  log "Index name: $INDEX_NAME"
  cd "$FRUGALRAG_REPO"
  export PORT
  PYTHONPATH="$FRUGALRAG_REPO" "$VENV_DIR/bin/python" -m src.search.serve_colbert.py \
    --index_root "$INDEX_ROOT" \
    --index "$INDEX_NAME" \
    --colbert_path "$COLBERT_CHECKPOINT" \
    --collection_path "$COLLECTION_TSV"
}

smoke_step() {
  require_cmd curl
  require_cmd jq
  log "Smoke testing server on port $PORT"
  curl -s "http://127.0.0.1:${PORT}/api/search?query=South%20Park%20The%20Hobbit%20Trey%20Parker&k=3" \
    | jq '.topk[0:3]'
}

status_step() {
  ensure_dirs
  log "Disk usage"
  df -h /workspace || true
  du -sh "$BUILD_ROOT"/* 2>/dev/null | sort -h || true
  log "GPU"
  nvidia-smi || true
}

all_until_index_step() {
  setup_step
  download_step
  convert_step
  index_help_step
  log "About to run index build..."
  index_step
}

cmd="${1:-}"
case "$cmd" in
  setup) setup_step ;;
  download) download_step ;;
  convert) convert_step ;;
  index-help) index_help_step ;;
  index) index_step ;;
  serve) serve_step ;;
  smoke) smoke_step ;;
  status) status_step ;;
  all-until-index) all_until_index_step ;;
  *)
    cat <<EOF
Usage: $0 <command>

Commands:
  setup            Install deps, clone repos, install ColBERT
  download         Download psgs_w100.tsv.gz (wiki20M passages)
  convert          Convert DPR TSV.gz to FrugalRAG-compatible collection.tsv
  index-help       Print ColBERT indexing CLI help (version/syntax check)
  index            Build ColBERT index (wiki20M)
  serve            Start FrugalRAG ColBERT HTTP server
  smoke            Smoke-test the running server
  status           Show disk and GPU status
  all-until-index  Run setup+download+convert+index-help+index

Environment variables (optional):
  BUILD_ROOT, DOWNLOAD_DIR, CORPUS_DIR, INDEX_ROOT, LOG_DIR, CODE_DIR, MODELS_DIR
  VENV_DIR
  PSGS_URL, INDEX_NAME, COLBERT_CHECKPOINT, PORT
EOF
    exit 1
    ;;
esac
