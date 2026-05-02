#!/usr/bin/env bash
set -euo pipefail

log() {
  printf "[%s] %s\n" "$(date '+%F %T')" "$*"
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

run_aws() {
  if ((${#AWS_CMD[@]} == 0)); then
    if command -v aws >/dev/null 2>&1; then
      AWS_CMD=(aws)
    elif command -v python3 >/dev/null 2>&1; then
      log "aws not found; trying python module awscli"
      if python3 -m pip show awscli >/dev/null 2>&1; then
        AWS_CMD=(python3 -m awscli)
      elif [[ "${AUTO_INSTALL_AWSCLI:-1}" == "1" ]]; then
        log "installing awscli into user site-packages"
        python3 -m pip install --user --disable-pip-version-check awscli >/tmp/awscli_install.log 2>&1 || \
          die "failed to install awscli: check /tmp/awscli_install.log"
        AWS_CMD=(python3 -m awscli)
      else
        die "aws CLI not found and AUTO_INSTALL_AWSCLI=0; install awscli package"
      fi
    else
      die "python3 not found; cannot fall back to aws module"
    fi
  fi
  "${AWS_CMD[@]}" "$@"
}

: "${S3_BUCKET:?S3_BUCKET is required}"
: "${S3_PREFIX:?S3_PREFIX is required}"
declare -a AWS_CMD=()
export AWS_PAGER="${AWS_PAGER:-""}"

S3_URI="s3://${S3_BUCKET}/${S3_PREFIX}"
DST_ROOT="/workspace/data/seer_sft"

GOLD_SFT_DST="${GOLD_SFT_DST:-$DST_ROOT/qlora-out-gold-musique-10k-k3}"
MMR_SFT_DST="${MMR_SFT_DST:-$DST_ROOT/qlora-out-mmr-musique-10k-k3-fullhop}"
GOLD_IPO_DST="${GOLD_IPO_DST:-$DST_ROOT/dpo-gold-musique-10k-ipo-r64-mingap000}"
MMR_IPO_DST="${MMR_IPO_DST:-$DST_ROOT/dpo-mmr-musique-10k-ipo-r64-mingap01143-exhaustive}"

mkdir -p "$GOLD_SFT_DST" "$MMR_SFT_DST" "$GOLD_IPO_DST" "$MMR_IPO_DST"
run_aws --version >/dev/null

SYNC_PROGRESS_OPT=""
if run_aws s3 help 2>/dev/null | grep -q -- " --progress"; then
  SYNC_PROGRESS_OPT="--progress"
fi

run_sync() {
  local src="$1"
  local dst="$2"

  if [[ -n "$SYNC_PROGRESS_OPT" ]]; then
    if ! run_aws s3 sync "$src" "$dst" "$SYNC_PROGRESS_OPT"; then
      log "retrying sync without --progress due unsupported option"
      run_aws s3 sync "$src" "$dst"
    fi
  else
    run_aws s3 sync "$src" "$dst"
  fi
}

download_dir() {
  local src="$1"
  local dst="$2"
  local label="$3"
  log "START ${label}"
  log "  from: ${src}"
  log "  to:   ${dst}"
  run_sync "$src" "$dst"
  log "DONE ${label}"
}

log "Downloading musique 10k model artifacts from ${S3_URI}"
log "AWS_REGION=${AWS_REGION:-unset} AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-unset}"

download_dir "$S3_URI/gold/qlora-out-gold-musique-10k-k3/" "$GOLD_SFT_DST/" "gold_sft"
download_dir "$S3_URI/mmr/qlora-out-mmr-musique-10k-k3-fullhop/" "$MMR_SFT_DST/" "mmr_sft"
download_dir "$S3_URI/gold/dpo-gold-musique-10k-ipo-r64-mingap000/" "$GOLD_IPO_DST/" "gold_ipo"
download_dir "$S3_URI/mmr/dpo-mmr-musique-10k-ipo-r64-mingap01143-exhaustive/" "$MMR_IPO_DST/" "mmr_ipo"

for path in "$GOLD_SFT_DST" "$MMR_SFT_DST" "$GOLD_IPO_DST" "$MMR_IPO_DST"; do
  log "Downloaded: ${path} ($(du -sh "$path" | awk '{print $1}'))"
done

log "DOWNLOAD COMPLETE to ${DST_ROOT}"
