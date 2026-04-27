#!/usr/bin/env bash
# Wrapper to run libero10_kyungtae_eval.py inside the bigenlight/libero-pro
# container. Mirrors run.sh's --vla-eval mode but mounts /workspace/kyungtae_data
# and calls the kyungtae driver instead.
#
# Usage (from Libero-pro_benchmark/):
#   ./scripts/run_kyungtae_eval.sh                      # full LIBERO-10 × 20 trial
#   ./scripts/run_kyungtae_eval.sh --num-trials 2 --task-ids 0
#   VLA_URL=http://localhost:8700 ./scripts/run_kyungtae_eval.sh --num-trials 5
#
# Anything after the first positional flag is forwarded verbatim to
# libero10_kyungtae_eval.py.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

IMAGE="${LIBERO_IMAGE:-bigenlight/libero-pro:latest}"
OOD_DATA_DIR="$REPO_ROOT/ood_data"
LIBERO_PRO_SRC="$REPO_ROOT/LIBERO-PRO"
LIBERO_SRC="$REPO_ROOT/LIBERO"
KYUNGTAE_DATA_DIR="${KYUNGTAE_DATA_DIR:-/home/theo/workspace/kyungtae_data}"
VLA_URL="${VLA_URL:-http://localhost:8700}"

# ── Pre-flight ───────────────────────────────────────────────────────────
if ! command -v docker &>/dev/null; then
    echo "ERROR: docker not found"; exit 1
fi
if ! docker image inspect "$IMAGE" &>/dev/null; then
    echo "  Pulling $IMAGE ..."
    docker pull "$IMAGE"
fi

if [[ ! -d "$LIBERO_PRO_SRC" ]]; then
    echo "ERROR: $LIBERO_PRO_SRC not found. Clone LIBERO-PRO first."; exit 1
fi
if [[ ! -d "$LIBERO_SRC" ]]; then
    echo "WARNING: $LIBERO_SRC not found (optional)."
fi
if [[ ! -d "$OOD_DATA_DIR" ]]; then
    echo "WARNING: $OOD_DATA_DIR not found (OK for libero_10 standard eval)."
fi

mkdir -p "$KYUNGTAE_DATA_DIR"

# ── Mounts ───────────────────────────────────────────────────────────────
SRC_MOUNT="-v $LIBERO_PRO_SRC:/workspace/LIBERO-PRO"
if [[ -d "$LIBERO_SRC" ]]; then
    SRC_MOUNT="$SRC_MOUNT -v $LIBERO_SRC:/workspace/LIBERO:ro"
fi
OOD_MOUNT=""
if [[ -d "$OOD_DATA_DIR" ]]; then
    OOD_MOUNT="-v $OOD_DATA_DIR:/tmp/ood_data:ro"
fi

ENTRYPOINT_CMD='
if [ -d /tmp/ood_data ]; then
    if [ -d /tmp/ood_data/bddl_files ] && [ -n "$(ls -A /tmp/ood_data/bddl_files 2>/dev/null)" ]; then
        cp -rn /tmp/ood_data/bddl_files/* /workspace/LIBERO-PRO/libero/libero/bddl_files/ || true
    fi
    if [ -d /tmp/ood_data/init_files ] && [ -n "$(ls -A /tmp/ood_data/init_files 2>/dev/null)" ]; then
        cp -rn /tmp/ood_data/init_files/* /workspace/LIBERO-PRO/libero/libero/init_files/ || true
    fi
fi
'

# Quote-safe forwarding: encode $@ as a single shell-escaped string, then let
# the inner bash -c re-parse it. printf '%q' handles spaces/special chars.
FORWARD_ARGS=""
if [[ $# -gt 0 ]]; then
    FORWARD_ARGS=$(printf ' %q' "$@")
fi

echo "=== kyungtae eval ==="
echo "  IMAGE       : $IMAGE"
echo "  VLA_URL     : $VLA_URL"
echo "  output dir  : $KYUNGTAE_DATA_DIR  -> /workspace/kyungtae_data"
echo "  forward args:$FORWARD_ARGS"
echo ""

# ── Run ──────────────────────────────────────────────────────────────────
docker run --rm --gpus all --shm-size=8g --network host \
    -e MUJOCO_GL=egl \
    -e VLA_SERVER_URL="$VLA_URL" \
    $SRC_MOUNT \
    $OOD_MOUNT \
    -v "$REPO_ROOT/scripts:/workspace/LIBERO-PRO/scripts:ro" \
    -v "$KYUNGTAE_DATA_DIR:/workspace/kyungtae_data" \
    "$IMAGE" \
    bash -c "$ENTRYPOINT_CMD
pip install --quiet requests imageio imageio-ffmpeg >/dev/null 2>&1 || true
python scripts/libero10_kyungtae_eval.py --vla-url $VLA_URL$FORWARD_ARGS"
