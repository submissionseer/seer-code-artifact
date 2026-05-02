#!/usr/bin/env bash
set -euo pipefail

# Backup/restore helper for the ColBERT wiki20m serving assets used on RunPod.
#
# Stores only the serving-critical assets by default:
#   - index directory (e.g., wiki20m.nbits2)
#   - corpus/collection.tsv
#   - incoming/serve_colbert_min.py (optional, enabled by default)
#
# Example:
#   export S3_URI=YOUR_S3_URI
#   bash s3_colbert_assets.sh auth-check
#   bash s3_colbert_assets.sh upload
#   bash s3_colbert_assets.sh download

BUILD_ROOT="${BUILD_ROOT:-/workspace/musique_colbert_build}"
INDEX_NAME="${INDEX_NAME:-wiki20m.nbits2}"
INDEX_ROOT="${INDEX_ROOT:-$BUILD_ROOT/index/musique_colbert_build/indexes}"
COLLECTION_TSV="${COLLECTION_TSV:-$BUILD_ROOT/corpus/collection.tsv}"
SERVE_SCRIPT="${SERVE_SCRIPT:-$BUILD_ROOT/incoming/serve_colbert_min.py}"
INCLUDE_SERVE_SCRIPT="${INCLUDE_SERVE_SCRIPT:-1}"
GENERATE_SHA256="${GENERATE_SHA256:-0}"
DELETE_REMOTE_EXTRA="${DELETE_REMOTE_EXTRA:-0}"
AWS_S3_EXTRA_ARGS="${AWS_S3_EXTRA_ARGS:-}"
S3_URI="${S3_URI:-}"
AWS_BIN="${AWS_BIN:-aws}"

ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

aws_cli() {
  AWS_PAGER="" "$AWS_BIN" "$@"
}

usage() {
  cat <<EOF
Usage: $0 <command>

Commands:
  auth-check   Verify AWS credentials and S3 bucket access
  upload       Upload index/corpus (and serving script) to S3
  download     Restore index/corpus (and serving script) from S3
  print-paths  Print resolved local paths and S3 locations

Required env (for auth-check/upload/download):
  S3_URI                 Example: YOUR_S3_URI

Optional env:
  BUILD_ROOT             Default: /workspace/musique_colbert_build
  INDEX_NAME             Default: wiki20m.nbits2
  INDEX_ROOT             Default: \$BUILD_ROOT/index/musique_colbert_build/indexes
  COLLECTION_TSV         Default: \$BUILD_ROOT/corpus/collection.tsv
  SERVE_SCRIPT           Default: \$BUILD_ROOT/incoming/serve_colbert_min.py
  INCLUDE_SERVE_SCRIPT   1/0 (default: 1)
  GENERATE_SHA256        1/0 (default: 0)  # Can take a long time on 100GB+ indexes
  DELETE_REMOTE_EXTRA    1/0 (default: 0)  # Adds --delete to aws s3 sync uploads
  AWS_S3_EXTRA_ARGS      Extra args passed to aws s3 cp/sync (e.g., "--sse AES256")
  AWS_BIN                aws binary to use (default: aws). Set to /usr/local/bin/aws on RunPod if needed.

Examples:
  S3_URI=YOUR_S3_URI $0 upload
  S3_URI=YOUR_S3_URI $0 download
EOF
}

parse_s3_uri() {
  [[ -n "$S3_URI" ]] || die "S3_URI is required (for example, YOUR_S3_URI)"
  [[ "$S3_URI" == s3://* ]] || die "S3_URI must start with s3://"
  S3_URI="${S3_URI%/}"

  local no_scheme="${S3_URI#s3://}"
  S3_BUCKET="${no_scheme%%/*}"
  if [[ "$no_scheme" == "$S3_BUCKET" ]]; then
    S3_PREFIX=""
  else
    S3_PREFIX="${no_scheme#*/}"
  fi
  [[ -n "$S3_BUCKET" ]] || die "Could not parse bucket from S3_URI=$S3_URI"

  REMOTE_INDEX_DIR="${S3_URI}/indexes/${INDEX_NAME}"
  REMOTE_COLLECTION="${S3_URI}/corpus/collection.tsv"
  REMOTE_SERVE_SCRIPT="${S3_URI}/incoming/serve_colbert_min.py"
  REMOTE_METADATA="${S3_URI}/metadata.json"
  REMOTE_SHA256="${S3_URI}/sha256sums.txt"
}

resolve_paths() {
  LOCAL_INDEX_DIR="${INDEX_ROOT%/}/${INDEX_NAME}"

  # Backward-compat fallback if the caller passes INDEX_ROOT=/workspace/.../index
  if [[ ! -d "$LOCAL_INDEX_DIR" && -d "${BUILD_ROOT}/index/${BUILD_ROOT##*/}/indexes/${INDEX_NAME}" ]]; then
    LOCAL_INDEX_DIR="${BUILD_ROOT}/index/${BUILD_ROOT##*/}/indexes/${INDEX_NAME}"
    INDEX_ROOT="$(dirname "$LOCAL_INDEX_DIR")"
  fi

  # Exact path seen in current RunPod setup.
  if [[ ! -d "$LOCAL_INDEX_DIR" && -d "${BUILD_ROOT}/index/musique_colbert_build/indexes/${INDEX_NAME}" ]]; then
    LOCAL_INDEX_DIR="${BUILD_ROOT}/index/musique_colbert_build/indexes/${INDEX_NAME}"
    INDEX_ROOT="$(dirname "$LOCAL_INDEX_DIR")"
  fi
}

print_paths() {
  parse_s3_uri
  resolve_paths
  cat <<EOF
BUILD_ROOT=$BUILD_ROOT
INDEX_NAME=$INDEX_NAME
INDEX_ROOT=$INDEX_ROOT
LOCAL_INDEX_DIR=$LOCAL_INDEX_DIR
COLLECTION_TSV=$COLLECTION_TSV
SERVE_SCRIPT=$SERVE_SCRIPT
S3_URI=$S3_URI
REMOTE_INDEX_DIR=$REMOTE_INDEX_DIR
REMOTE_COLLECTION=$REMOTE_COLLECTION
REMOTE_SERVE_SCRIPT=$REMOTE_SERVE_SCRIPT
REMOTE_METADATA=$REMOTE_METADATA
REMOTE_SHA256=$REMOTE_SHA256
EOF
}

aws_s3_cp() {
  if [[ -n "$AWS_S3_EXTRA_ARGS" ]]; then
    # shellcheck disable=SC2086
    aws_cli s3 cp "$1" "$2" --only-show-errors $AWS_S3_EXTRA_ARGS
  else
    aws_cli s3 cp "$1" "$2" --only-show-errors
  fi
}

aws_s3_sync() {
  local src="$1"
  local dst="$2"
  local delete_flag=()
  if [[ "$DELETE_REMOTE_EXTRA" == "1" ]]; then
    delete_flag=(--delete)
  fi
  if [[ -n "$AWS_S3_EXTRA_ARGS" ]]; then
    # shellcheck disable=SC2086
    aws_cli s3 sync "$src" "$dst" --only-show-errors "${delete_flag[@]}" $AWS_S3_EXTRA_ARGS
  else
    aws_cli s3 sync "$src" "$dst" --only-show-errors "${delete_flag[@]}"
  fi
}

write_metadata_file() {
  local tmp_meta="$1"
  local index_bytes collection_bytes index_files
  index_bytes="$(du -sb "$LOCAL_INDEX_DIR" | awk '{print $1}')"
  collection_bytes="$(du -sb "$COLLECTION_TSV" | awk '{print $1}')"
  index_files="$(find "$LOCAL_INDEX_DIR" -type f | wc -l | awk '{print $1}')"
  cat >"$tmp_meta" <<EOF
{
  "created_at_utc": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "build_root": "${BUILD_ROOT}",
  "index_name": "${INDEX_NAME}",
  "index_root": "${INDEX_ROOT}",
  "local_index_dir": "${LOCAL_INDEX_DIR}",
  "collection_tsv": "${COLLECTION_TSV}",
  "include_serve_script": ${INCLUDE_SERVE_SCRIPT},
  "index_file_count": ${index_files},
  "index_size_bytes": ${index_bytes},
  "collection_size_bytes": ${collection_bytes}
}
EOF
}

generate_sha256_file() {
  local tmp_sha="$1"
  require_cmd sha256sum
  log "Generating sha256 manifest (this can take a long time for a 100GB+ index)..."
  (
    cd "$BUILD_ROOT"
    find "${LOCAL_INDEX_DIR#$BUILD_ROOT/}" -type f -print0 | sort -z | xargs -0 sha256sum
    sha256sum "${COLLECTION_TSV#$BUILD_ROOT/}"
    if [[ "$INCLUDE_SERVE_SCRIPT" == "1" && -f "$SERVE_SCRIPT" ]]; then
      sha256sum "${SERVE_SCRIPT#$BUILD_ROOT/}"
    fi
  ) >"$tmp_sha"
}

auth_check() {
  parse_s3_uri
  require_cmd "$AWS_BIN"

  log "Checking AWS identity..."
  aws_cli sts get-caller-identity

  log "Checking S3 bucket access (s3://$S3_BUCKET)..."
  aws_cli s3 ls "s3://$S3_BUCKET" >/dev/null

  log "AWS auth + S3 access OK"
}

upload() {
  parse_s3_uri
  resolve_paths
  require_cmd "$AWS_BIN"

  [[ -d "$LOCAL_INDEX_DIR" ]] || die "Missing index dir: $LOCAL_INDEX_DIR"
  [[ -f "$COLLECTION_TSV" ]] || die "Missing collection.tsv: $COLLECTION_TSV"
  if [[ "$INCLUDE_SERVE_SCRIPT" == "1" ]]; then
    [[ -f "$SERVE_SCRIPT" ]] || die "Missing serve script: $SERVE_SCRIPT"
  fi

  log "Uploading index dir: $LOCAL_INDEX_DIR -> $REMOTE_INDEX_DIR/"
  aws_s3_sync "$LOCAL_INDEX_DIR/" "$REMOTE_INDEX_DIR/"

  log "Uploading collection: $COLLECTION_TSV -> $REMOTE_COLLECTION"
  mkdir -p "${BUILD_ROOT}/.tmp_s3_upload"
  aws_s3_cp "$COLLECTION_TSV" "$REMOTE_COLLECTION"

  if [[ "$INCLUDE_SERVE_SCRIPT" == "1" ]]; then
    log "Uploading serve script: $SERVE_SCRIPT -> $REMOTE_SERVE_SCRIPT"
    aws_s3_cp "$SERVE_SCRIPT" "$REMOTE_SERVE_SCRIPT"
  fi

  local tmp_meta="${BUILD_ROOT}/.tmp_s3_upload/metadata.json"
  write_metadata_file "$tmp_meta"
  log "Uploading metadata -> $REMOTE_METADATA"
  aws_s3_cp "$tmp_meta" "$REMOTE_METADATA"

  if [[ "$GENERATE_SHA256" == "1" ]]; then
    local tmp_sha="${BUILD_ROOT}/.tmp_s3_upload/sha256sums.txt"
    generate_sha256_file "$tmp_sha"
    log "Uploading sha256 manifest -> $REMOTE_SHA256"
    aws_s3_cp "$tmp_sha" "$REMOTE_SHA256"
  fi

  log "Upload complete"
  log "Remote prefix: $S3_URI"
}

download() {
  parse_s3_uri
  resolve_paths
  require_cmd "$AWS_BIN"

  mkdir -p "$INDEX_ROOT" "$(dirname "$COLLECTION_TSV")"
  if [[ "$INCLUDE_SERVE_SCRIPT" == "1" ]]; then
    mkdir -p "$(dirname "$SERVE_SCRIPT")"
  fi

  log "Downloading index dir: $REMOTE_INDEX_DIR/ -> $LOCAL_INDEX_DIR/"
  mkdir -p "$LOCAL_INDEX_DIR"
  aws_cli s3 sync "$REMOTE_INDEX_DIR/" "$LOCAL_INDEX_DIR/" --only-show-errors

  log "Downloading collection: $REMOTE_COLLECTION -> $COLLECTION_TSV"
  aws_cli s3 cp "$REMOTE_COLLECTION" "$COLLECTION_TSV" --only-show-errors

  if [[ "$INCLUDE_SERVE_SCRIPT" == "1" ]]; then
    log "Downloading serve script: $REMOTE_SERVE_SCRIPT -> $SERVE_SCRIPT"
    aws_cli s3 cp "$REMOTE_SERVE_SCRIPT" "$SERVE_SCRIPT" --only-show-errors || true
  fi

  log "Attempting to download metadata (optional)"
  aws_cli s3 cp "$REMOTE_METADATA" "${BUILD_ROOT}/s3_restore_metadata.json" --only-show-errors || true

  if aws_cli s3 ls "$REMOTE_SHA256" >/dev/null 2>&1; then
    log "Downloading sha256 manifest (optional)"
    aws_cli s3 cp "$REMOTE_SHA256" "${BUILD_ROOT}/sha256sums.txt" --only-show-errors
    if command -v sha256sum >/dev/null 2>&1; then
      log "sha256 manifest downloaded to ${BUILD_ROOT}/sha256sums.txt (verify manually when ready)"
    fi
  fi

  log "Download complete"
  du -sh "$LOCAL_INDEX_DIR" "$COLLECTION_TSV" 2>/dev/null || true
}

cmd="${1:-}"
case "$cmd" in
  auth-check) auth_check ;;
  upload) upload ;;
  download) download ;;
  print-paths) print_paths ;;
  -h|--help|help|"") usage ;;
  *) die "Unknown command: $cmd (use --help)" ;;
esac
