#!/usr/bin/env bash
set -euo pipefail

# One-command bootstrap + launch for ColBERT search serving on RunPod.
# - Reuses a venv on /workspace (persistent if using a network volume)
# - Optionally restores index/corpus from S3 if missing
# - Installs pinned, known-good serving deps
# - Starts the minimal Flask search server
#
# Usage:
#   bash start_colbert_service.sh
#   S3_URI=YOUR_S3_URI bash start_colbert_service.sh

BUILD_ROOT="${BUILD_ROOT:-/workspace/musique_colbert_build}"
VENV_DIR="${VENV_DIR:-$BUILD_ROOT/venv}"
HF_HOME="${HF_HOME:-$BUILD_ROOT/hf_cache}"
CODE_DIR="${CODE_DIR:-$BUILD_ROOT/code}"
INCOMING_DIR="${INCOMING_DIR:-$BUILD_ROOT/incoming}"

INDEX_NAME="${INDEX_NAME:-wiki20m.nbits2}"
INDEX_ROOT="${INDEX_ROOT:-$BUILD_ROOT/index/musique_colbert_build/indexes}"
INDEX_DIR="${INDEX_DIR:-$INDEX_ROOT/$INDEX_NAME}"
COLLECTION_TSV="${COLLECTION_TSV:-$BUILD_ROOT/corpus/collection.tsv}"
CHECKPOINT="${CHECKPOINT:-colbert-ir/colbertv2.0}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
THREADED="${THREADED:-1}"

SERVE_SCRIPT="${SERVE_SCRIPT:-$INCOMING_DIR/serve_colbert_min.py}"
S3_HELPER="${S3_HELPER:-$INCOMING_DIR/s3_colbert_assets.sh}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/requirements-colbert-serve.txt}"
COLBERT_REPO="${COLBERT_REPO:-$CODE_DIR/ColBERT}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PATH="$VENV_DIR/bin:$PATH"

ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

ensure_dirs() {
  mkdir -p "$BUILD_ROOT" "$INCOMING_DIR" "$CODE_DIR" "$(dirname "$INDEX_ROOT")" "$(dirname "$COLLECTION_TSV")" "$HF_HOME"
}

ensure_venv() {
  ensure_dirs
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    log "Creating venv at $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi
}

py() { "$VENV_DIR/bin/python" "$@"; }
pipv() { "$VENV_DIR/bin/pip" "$@"; }

ensure_system_tools() {
  local missing=()
  for c in git curl; do
    command -v "$c" >/dev/null 2>&1 || missing+=("$c")
  done
  if [[ ${#missing[@]} -gt 0 ]]; then
    log "Installing missing system tools: ${missing[*]}"
    apt-get update
    apt-get install -y "${missing[@]}"
  fi
}

ensure_aws_cli_if_needed() {
  [[ -n "${S3_URI:-}" ]] || return 0
  if command -v aws >/dev/null 2>&1; then
    return 0
  fi
  log "S3_URI set and aws not found. Installing AWS CLI v2..."
  apt-get update
  apt-get install -y curl unzip
  local arch url
  arch="$(uname -m)"
  case "$arch" in
    x86_64) url="https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" ;;
    aarch64) url="https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" ;;
    *) die "Unsupported architecture for awscli v2 auto-install: $arch" ;;
  esac
  tmpd="$(mktemp -d)"
  curl -sSLo "$tmpd/awscliv2.zip" "$url"
  unzip -q -o "$tmpd/awscliv2.zip" -d "$tmpd"
  "$tmpd/aws/install" --update
  export PATH=/usr/local/bin:$PATH
  export AWS_PAGER="${AWS_PAGER:-}"
}

restore_from_s3_if_missing() {
  if [[ -d "$INDEX_DIR" && -f "$COLLECTION_TSV" ]]; then
    log "Index and collection already present. Skipping S3 restore."
    return 0
  fi

  [[ -n "${S3_URI:-}" ]] || die "Index/corpus missing and S3_URI not set"
  [[ -x "$S3_HELPER" ]] || die "S3 helper not found/executable: $S3_HELPER"

  log "Index/corpus missing. Restoring from S3 via $S3_HELPER"
  export BUILD_ROOT INDEX_NAME INDEX_ROOT COLLECTION_TSV SERVE_SCRIPT
  export AWS_BIN="${AWS_BIN:-$(command -v aws || echo aws)}"
  export AWS_PAGER="${AWS_PAGER:-}"
  "$S3_HELPER" auth-check
  "$S3_HELPER" download
}

ensure_colbert_repo() {
  if [[ -d "$COLBERT_REPO/.git" ]]; then
    return 0
  fi
  log "Cloning ColBERT repo to $COLBERT_REPO"
  git clone https://github.com/stanford-futuredata/ColBERT.git "$COLBERT_REPO"
}

install_python_deps() {
  [[ -f "$REQUIREMENTS_FILE" ]] || die "Missing requirements file: $REQUIREMENTS_FILE"
  log "Installing pinned ColBERT serving requirements"
  pipv install --upgrade pip setuptools wheel
  pipv install -r "$REQUIREMENTS_FILE"
  ensure_colbert_repo
  pipv install -e "$COLBERT_REPO"
}

sanity_check_imports() {
  log "Sanity-checking imports"
  py - <<'PY'
import transformers, torch
print("transformers", transformers.__version__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
from colbert import Searcher
print("colbert import ok")
PY
}

print_status() {
  log "Disk"
  df -h /workspace || true
  du -sh "$BUILD_ROOT"/{index,corpus,venv,hf_cache,incoming} 2>/dev/null || true
  log "Memory"
  free -h || true
  log "GPU"
  nvidia-smi || true
}

start_server() {
  [[ -f "$SERVE_SCRIPT" ]] || die "Missing serve script: $SERVE_SCRIPT"
  [[ -d "$INDEX_DIR" ]] || die "Missing index dir: $INDEX_DIR"
  [[ -f "$COLLECTION_TSV" ]] || die "Missing collection file: $COLLECTION_TSV"

  export HF_HOME
  local threaded_flag=()
  if [[ "$THREADED" == "1" ]]; then
    threaded_flag=(--threaded)
  fi

  log "Starting ColBERT server"
  log "Endpoint will bind to http://$HOST:$PORT"
  exec "$VENV_DIR/bin/python" "$SERVE_SCRIPT" \
    --index_root "$INDEX_ROOT" \
    --index "$INDEX_NAME" \
    --collection "$COLLECTION_TSV" \
    --checkpoint "$CHECKPOINT" \
    --host "$HOST" \
    --port "$PORT" \
    "${threaded_flag[@]}"
}

main() {
  ensure_dirs
  ensure_system_tools
  ensure_aws_cli_if_needed
  ensure_venv
  restore_from_s3_if_missing
  install_python_deps
  sanity_check_imports
  print_status
  start_server
}

main "$@"
