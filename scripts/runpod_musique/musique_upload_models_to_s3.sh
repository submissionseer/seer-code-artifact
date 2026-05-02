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
TRAIN_ROOT="/workspace/data/seer_sft"
TRAIN_ROOT_ALT="/workspace/data/dpo_out"

GOLD_SFT_SRC="${GOLD_SFT_SRC:-$TRAIN_ROOT/qlora-out-gold-musique-10k-k3}"
MMR_SFT_SRC="${MMR_SFT_SRC:-$TRAIN_ROOT/qlora-out-mmr-musique-10k-k3-fullhop}"
GOLD_IPO_SRC="${GOLD_IPO_SRC:-$TRAIN_ROOT/dpo-gold-musique-10k-ipo-r64-mingap000}"
MMR_IPO_SRC="${MMR_IPO_SRC:-$TRAIN_ROOT/dpo-mmr-musique-10k-ipo-r64-mingap01143-exhaustive}"

if [[ ! -d "$GOLD_IPO_SRC" ]] && [[ -d "$TRAIN_ROOT_ALT/dpo-gold-musique-10k-ipo-r64-mingap000" ]]; then
  GOLD_IPO_SRC="$TRAIN_ROOT_ALT/dpo-gold-musique-10k-ipo-r64-mingap000"
fi
if [[ ! -d "$MMR_IPO_SRC" ]] && [[ -d "$TRAIN_ROOT_ALT/dpo-mmr-musique-10k-ipo-r64-mingap01143-exhaustive" ]]; then
  MMR_IPO_SRC="$TRAIN_ROOT_ALT/dpo-mmr-musique-10k-ipo-r64-mingap01143-exhaustive"
fi

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

for path in "$GOLD_SFT_SRC" "$MMR_SFT_SRC" "$GOLD_IPO_SRC" "$MMR_IPO_SRC"; do
  [[ -d "$path" ]] || die "Missing required directory: $path"
done

sync_to_s3() {
  local src="$1"
  local dst="$2"
  local label="$3"
  log "START ${label}"
  log "  from: ${src}"
  log "  to:   ${dst}"
  run_sync "$src" "$dst"
  log "DONE ${label}"
}

log "Uploading musique 10k model artifacts to ${S3_URI}"
log "AWS_REGION=${AWS_REGION:-unset} AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-unset}"

sync_to_s3 "$GOLD_SFT_SRC" "$S3_URI/gold/qlora-out-gold-musique-10k-k3/" "gold_sft"
sync_to_s3 "$MMR_SFT_SRC" "$S3_URI/mmr/qlora-out-mmr-musique-10k-k3-fullhop/" "mmr_sft"
sync_to_s3 "$GOLD_IPO_SRC" "$S3_URI/gold/dpo-gold-musique-10k-ipo-r64-mingap000/" "gold_ipo"
sync_to_s3 "$MMR_IPO_SRC" "$S3_URI/mmr/dpo-mmr-musique-10k-ipo-r64-mingap01143-exhaustive/" "mmr_ipo"

run_aws s3 ls "$S3_URI/" --recursive
log "UPLOAD COMPLETE: ${S3_URI}"
