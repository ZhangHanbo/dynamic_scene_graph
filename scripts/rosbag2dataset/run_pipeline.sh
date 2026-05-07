#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# One-shot launcher for the SceneRep detection pipeline.
#
#   1. Opens an SSH port-forward to the crane5 detection host with the
#      most-robust settings short of autossh (server-alive probes, fast
#      fail on tunnel bring-up, reconnect loop on drop).
#   2. Waits until both forwarded ports answer TCP.
#   3. Runs probe_servers.py against 127.0.0.1 to verify OWL + SAM.
#   4. (Optional) Runs the full per-dataset pipeline:
#         owl_client → sam_client → track_object_ids
#      for every dataset passed on the CLI.
#   5. Always tears the tunnel down on exit (or Ctrl-C).
#
# Usage:
#   scripts/rosbag2dataset/run_pipeline.sh                     # probe-only
#   scripts/rosbag2dataset/run_pipeline.sh apple_drop          # probe + process 1 dataset
#   scripts/rosbag2dataset/run_pipeline.sh apple_drop apple_in_the_tray ...
#
# Environment variables (all optional):
#   SSH_USER           ssh user on the detection host  (default: hanbo)
#   SSH_HOST           detection host                  (default: crane5.ddns.comp.nus.edu.sg)
#   OWL_PORT SAM2_PORT remote+local ports              (defaults: 4051 4057)
#                      The SAM2 server at SAM2_PORT also serves the legacy
#                      /sam_* endpoints, so no separate SAM_PORT is needed.
#                      Override SAM_PORT to keep tunneling the old SAM
#                      service in addition.
#   DATASET_PATH       dataset root dir                (default: $HOME/datasets)
#   PYTHON             interpreter                     (default: python)
#   USE_SAM2=1         use SAM2 video tracker          (replaces SAM + tracker)
#   SKIP_PROBE=1       skip the live server probe
#   SKIP_OWL=1         skip OWL stage   (assumes detection JSONs already exist)
#   SKIP_SAM=1         skip SAM stage   (ignored when USE_SAM2=1)
#   SKIP_TRACK=1       skip tracker     (ignored when USE_SAM2=1)
# -----------------------------------------------------------------------------

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SSH_USER="${SSH_USER:-hanbo}"
SSH_HOST="${SSH_HOST:-crane5.ddns.comp.nus.edu.sg}"
OWL_PORT="${OWL_PORT:-4051}"
# SAM2 hosts both /sam2_* and /sam_* endpoints. Set SAM_PORT to a non-empty
# value (e.g. SAM_PORT=4057) only if you also want to tunnel the legacy
# SAM v1 service running on a different host.
SAM2_PORT="${SAM2_PORT:-4057}"
SAM_PORT="${SAM_PORT:-}"
DATASET_PATH="${DATASET_PATH:-$REPO_ROOT/datasets}"
PYTHON="${PYTHON:-python}"
USE_SAM2="${USE_SAM2:-0}"

# State files for the tunnel watchdog
STATE_DIR="$REPO_ROOT/rosbag2dataset/.tunnel"
mkdir -p "$STATE_DIR"
PIDFILE_WATCHDOG="$STATE_DIR/watchdog.pid"
PIDFILE_SSH="$STATE_DIR/ssh.pid"
LOGFILE="$STATE_DIR/ssh.log"

log()   { printf '\033[1;34m[pipe]\033[0m %s\n' "$*"; }
warn()  { printf '\033[1;33m[pipe]\033[0m %s\n' "$*" >&2; }
die()   { printf '\033[1;31m[pipe]\033[0m %s\n' "$*" >&2; exit 1; }

# -----------------------------------------------------------------------------
# Tunnel management
# -----------------------------------------------------------------------------

tunnel_ports_ready() {
  # TCP-reachable on all forwarded local ports.
  nc -z -G 2 127.0.0.1 "$OWL_PORT" 2>/dev/null \
    && nc -z -G 2 127.0.0.1 "$SAM2_PORT" 2>/dev/null \
    && { [[ -z "$SAM_PORT" ]] || nc -z -G 2 127.0.0.1 "$SAM_PORT" 2>/dev/null; }
}

# Inner loop: keep an ssh -N tunnel alive, restart on crash. Exits
# immediately when it receives TERM from the watchdog cleanup.
ssh_supervisor() {
  trap 'kill -TERM "${SSH_PID:-0}" 2>/dev/null || true; exit 0' TERM INT
  while true; do
    # - ExitOnForwardFailure: if bind fails (port busy), ssh exits fast
    # - ServerAliveInterval/CountMax: detect half-open links in ~90s
    # - BatchMode + StrictHostKeyChecking=accept-new: no interactive prompts
    # - ConnectTimeout: cap the initial handshake
    # - TCPKeepAlive: keep the NAT entry warm
    ssh -N \
        -o ExitOnForwardFailure=yes \
        -o ServerAliveInterval=30 \
        -o ServerAliveCountMax=3 \
        -o TCPKeepAlive=yes \
        -o ConnectTimeout=15 \
        -o StrictHostKeyChecking=accept-new \
        -o BatchMode=yes \
        -o ControlMaster=no \
        -L "$OWL_PORT:127.0.0.1:$OWL_PORT" \
        -L "$SAM2_PORT:127.0.0.1:$SAM2_PORT" \
        ${SAM_PORT:+-L "$SAM_PORT:127.0.0.1:$SAM_PORT"} \
        "${SSH_USER}@${SSH_HOST}" \
        >>"$LOGFILE" 2>&1 &
    SSH_PID=$!
    echo "$SSH_PID" > "$PIDFILE_SSH"
    wait "$SSH_PID" || true
    # Brief back-off before reconnecting (avoid a tight loop on auth failure)
    sleep 3
  done
}

start_tunnel() {
  if [[ -s "$PIDFILE_WATCHDOG" ]] && kill -0 "$(cat "$PIDFILE_WATCHDOG")" 2>/dev/null; then
    log "tunnel watchdog already running (pid $(cat "$PIDFILE_WATCHDOG")); reusing"
  else
    : > "$LOGFILE"
    log "opening SSH tunnel  ${SSH_USER}@${SSH_HOST}  "\
"-L $OWL_PORT -L $SAM2_PORT${SAM_PORT:+ -L $SAM_PORT}  log=$LOGFILE"
    ssh_supervisor &
    echo $! > "$PIDFILE_WATCHDOG"
    disown "$(cat "$PIDFILE_WATCHDOG")"
  fi

  # Wait for both forwarded ports to answer (≤30 s).
  for i in $(seq 1 30); do
    if tunnel_ports_ready; then
      if [[ -n "$SAM_PORT" ]]; then
      log "tunnel ready on 127.0.0.1:$OWL_PORT :$SAM2_PORT (+ legacy SAM :$SAM_PORT)"
    else
      log "tunnel ready on 127.0.0.1:$OWL_PORT :$SAM2_PORT  (SAM2 also serves /sam_*)"
    fi
      return 0
    fi
    sleep 1
  done

  warn "tunnel didn't come up within 30 s — last 20 lines of $LOGFILE:"
  tail -n 20 "$LOGFILE" >&2 || true
  die "giving up; check credentials and network"
}

stop_tunnel() {
  if [[ -s "$PIDFILE_WATCHDOG" ]]; then
    local wpid ssh_pid
    wpid="$(cat "$PIDFILE_WATCHDOG")"
    log "shutting down tunnel (watchdog pid $wpid)"
    kill -TERM "$wpid" 2>/dev/null || true
  fi
  if [[ -s "$PIDFILE_SSH" ]]; then
    ssh_pid="$(cat "$PIDFILE_SSH")"
    kill -TERM "$ssh_pid" 2>/dev/null || true
  fi
  rm -f "$PIDFILE_WATCHDOG" "$PIDFILE_SSH"
}

trap stop_tunnel EXIT INT TERM

# -----------------------------------------------------------------------------
# Pipeline stages
# -----------------------------------------------------------------------------

probe_servers() {
  [[ "${SKIP_PROBE:-0}" == "1" ]] && { log "SKIP_PROBE=1, skipping probe"; return 0; }
  log "probing OWL + SAM via tunnel"
  SCENEREP_SERVER_HOST=127.0.0.1 "$PYTHON" rosbag2dataset/probe_servers.py
}

process_dataset() {
  local ds="$1"
  local ds_dir

  # 1. Absolute path / already-extracted dir.
  if [[ -d "$ds" && -d "$ds/rgb" ]]; then
    ds_dir="$ds"
  elif [[ -d "$DATASET_PATH/$ds" && -d "$DATASET_PATH/$ds/rgb" ]]; then
    ds_dir="$DATASET_PATH/$ds"
  # 2. Extracted dir without rgb/ yet — treat as target, extract into it.
  elif [[ -d "$DATASET_PATH/$ds" ]]; then
    ds_dir="$DATASET_PATH/$ds"
  # 3. Raw bag — auto-extract.
  elif [[ -f "$DATASET_PATH/$ds.bag" ]]; then
    ds_dir="$DATASET_PATH/$ds"
    log "  [extract] $ds.bag → $ds_dir/  (cross-platform, via rosbags lib)"
    mkdir -p "$ds_dir"
    "$PYTHON" rosbag2dataset/extract_bag_local.py \
        "$DATASET_PATH/$ds.bag" "$ds_dir"
  elif [[ -f "$ds" && "$ds" == *.bag ]]; then
    ds_dir="${ds%.bag}"
    log "  [extract] $ds → $ds_dir/"
    mkdir -p "$ds_dir"
    "$PYTHON" rosbag2dataset/extract_bag_local.py "$ds" "$ds_dir"
  else
    warn "dataset not found: $ds  (checked '$ds', '$DATASET_PATH/$ds', "
    warn "                           and '$DATASET_PATH/$ds.bag')"
    return 1
  fi
  log "processing $ds_dir"

  export SCENEREP_SERVER_HOST=127.0.0.1
  export DATASET_PATH

  # Perception outputs go UNDER tests/visualization_pipeline/<dataset>/perception/
  # instead of inside the dataset. Override with PERCEPTION_OUT=<abs path>
  # or PERCEPTION_OUT="" to fall back to the legacy dataset-subdir write.
  local ds_name
  ds_name="$(basename "$ds_dir")"
  local perception_default="$REPO_ROOT/tests/visualization_pipeline/$ds_name/perception"
  local perception_out="${PERCEPTION_OUT-$perception_default}"
  local owl_out sam_out
  if [[ -n "$perception_out" ]]; then
    owl_out="$perception_out/detection_boxes"
    sam_out="$perception_out/detection_h"
    mkdir -p "$owl_out" "$sam_out"
    log "  perception outputs → $perception_out/"
  else
    owl_out="detection_boxes"
    sam_out="detection_h"
  fi

  if [[ "${SKIP_OWL:-0}" != "1" ]]; then
    log "  [owl]  → $owl_out/"
    "$PYTHON" rosbag2dataset/owl/owl_client.py "$ds_dir" --out-dir "$owl_out"
  fi

  if [[ "$USE_SAM2" == "1" ]]; then
    log "  [sam2-track] OWL boxes → SAM2 video tracker → $sam_out/"
    "$PYTHON" rosbag2dataset/sam2/sam2_client.py "$ds_dir" \
        --det-dir "$owl_out" --out-dir "$sam_out"
  else
    if [[ "${SKIP_SAM:-0}" != "1" ]]; then
      log "  [sam]  → adding masks"
      "$PYTHON" rosbag2dataset/sam/sam_client.py "$ds_dir"
    fi
    if [[ "${SKIP_TRACK:-0}" != "1" ]]; then
      log "  [track] → detection_h/"
      "$PYTHON" rosbag2dataset/track_object_ids.py "$ds_dir"
    fi
  fi
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

start_tunnel
probe_servers

if [[ $# -gt 0 ]]; then
  for ds in "$@"; do
    process_dataset "$ds"
  done
else
  log "no dataset arguments — stopping after probe"
fi

log "done"
