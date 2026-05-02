#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Deploy latest code to RunPod and launch one job in one command.

Usage:
  deploy_and_start_job.sh \
    --ssh-host HOST --ssh-port PORT --ssh-user USER --ssh-key PATH \
    --job <smoke|rollout|dpo-gold|dpo-mmr|dpo-raw-jina-score|dpo-raw-jina-pairs> [job args...]

Examples:
  deploy_and_start_job.sh \
    --ssh-host 103.196.86.50 --ssh-port 41247 --ssh-user root --ssh-key ~/.ssh/id_ed25519 \
    --job smoke --openrouter-api-key "$OPENROUTER_API_KEY"

  deploy_and_start_job.sh \
    --ssh-host 103.196.86.50 --ssh-port 41247 --ssh-user root --ssh-key ~/.ssh/id_ed25519 \
    --job rollout --limit 10000 \
    --openrouter-api-key "$OPENROUTER_API_KEY"

  deploy_and_start_job.sh \
    --ssh-host 103.196.86.50 --ssh-port 41247 --ssh-user root --ssh-key ~/.ssh/id_ed25519 \
    --job dpo-mmr --input musique_train_multihop_k3_N10000_5variants_final.jsonl \
    --openrouter-api-key "$OPENROUTER_API_KEY" --jina-api-key "$JINA_AI_API_KEY"

  deploy_and_start_job.sh \
    --ssh-host 103.196.86.50 --ssh-port 41247 --ssh-user root --ssh-key ~/.ssh/id_ed25519 \
    --job dpo-raw-jina-score --input data/rollouts/musique_dpo_rollouts_mmr.jsonl \
    --jina-api-key "$JINA_AI_API_KEY"
EOF
}

die() { echo "ERROR: $*" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

SSH_HOST=""
SSH_PORT=""
SSH_USER="root"
SSH_KEY="${HOME}/.ssh/id_ed25519"
REMOTE_ROOT="/workspace/seer-sufficiency"
SKIP_SYNC="0"

REMOTE_ARGS=()
while (($#)); do
  case "$1" in
    --ssh-host) SSH_HOST="${2:-}"; shift 2 ;;
    --ssh-port) SSH_PORT="${2:-}"; shift 2 ;;
    --ssh-user) SSH_USER="${2:-}"; shift 2 ;;
    --ssh-key) SSH_KEY="${2:-}"; shift 2 ;;
    --remote-root) REMOTE_ROOT="${2:-}"; shift 2 ;;
    --skip-sync) SKIP_SYNC="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) REMOTE_ARGS+=("$1"); shift ;;
  esac
done

[[ -n "$SSH_HOST" ]] || die "--ssh-host is required"
[[ -f "$SSH_KEY" ]] || die "SSH key not found: $SSH_KEY"

# Support --ssh-host HOST:PORT for convenience.
if [[ "$SSH_HOST" == *:* ]]; then
  host_part="${SSH_HOST%%:*}"
  port_part="${SSH_HOST##*:}"
  SSH_HOST="$host_part"
  if [[ -z "$SSH_PORT" ]]; then
    SSH_PORT="$port_part"
  fi
fi
if [[ -z "$SSH_PORT" ]]; then
  SSH_PORT="22"
fi

if [[ "$SSH_HOST" == "ssh.runpod.io" ]]; then
  die "ssh.runpod.io does not support this deploy flow (scp + non-interactive command exec). Use Connect -> SSH over exposed TCP (root@HOST -p PORT)."
fi

has_job_arg="0"
for arg in "${REMOTE_ARGS[@]}"; do
  if [[ "$arg" == "--job" ]]; then
    has_job_arg="1"
    break
  fi
done
[[ "$has_job_arg" == "1" ]] || die "Missing --job in job args"

SSH_OPTS=(
  -o StrictHostKeyChecking=accept-new
  -o ServerAliveInterval=30
  -o BatchMode=yes
  -o PreferredAuthentications=publickey
  -o PasswordAuthentication=no
  -o IdentitiesOnly=yes
  -p "$SSH_PORT"
  -i "$SSH_KEY"
)
SCP_OPTS=(
  -O
  -o StrictHostKeyChecking=accept-new
  -o ServerAliveInterval=30
  -o BatchMode=yes
  -o PreferredAuthentications=publickey
  -o PasswordAuthentication=no
  -o IdentitiesOnly=yes
  -P "$SSH_PORT"
  -i "$SSH_KEY"
)
SSH_DEST="${SSH_USER}@${SSH_HOST}"

if [[ "$SKIP_SYNC" != "1" ]]; then
  BUNDLE="$(mktemp -t seer_sync_code.XXXXXX.tgz)"
  trap 'rm -f "$BUNDLE"' EXIT

  # Avoid macOS metadata (._ files / xattrs) that can break extraction on Linux pods.
  COPYFILE_DISABLE=1 tar \
    --exclude='._*' \
    --exclude='__MACOSX' \
    -czf "$BUNDLE" -C "$REPO_ROOT" \
    configs \
    scripts \
    seer \
    pyproject.toml \
    README.md

  scp "${SCP_OPTS[@]}" "$BUNDLE" "${SSH_DEST}:/tmp/seer_sync_code.tgz"
  ssh "${SSH_OPTS[@]}" "$SSH_DEST" "mkdir -p '$REMOTE_ROOT' && tar --no-same-owner --no-same-permissions -xzf /tmp/seer_sync_code.tgz -C '$REMOTE_ROOT' && chmod +x '$REMOTE_ROOT/scripts/runpod_musique/runpod_job.sh'"
fi

REMOTE_CMD="cd $(printf '%q' "$REMOTE_ROOT") && ./scripts/runpod_musique/runpod_job.sh"
for arg in "${REMOTE_ARGS[@]}"; do
  REMOTE_CMD+=" $(printf '%q' "$arg")"
done

ssh "${SSH_OPTS[@]}" "$SSH_DEST" "$REMOTE_CMD"
