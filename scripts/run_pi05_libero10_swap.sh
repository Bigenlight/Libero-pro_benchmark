#!/usr/bin/env bash
set -euo pipefail

# ── Pi0.5 LIBERO-10 Swap 3-task evaluation launcher ──
# Runs libero10_swap_3task_eval.py inside the libero-pro container,
# hitting the pi05 HTTP server running on the host.
#
# Usage:
#   ./scripts/run_pi05_libero10_swap.sh
#   ./scripts/run_pi05_libero10_swap.sh --vla-url http://localhost:8400 --gpu 1
#   ./scripts/run_pi05_libero10_swap.sh --tasks "task1,task2" --num-trials 5

IMAGE="${LIBERO_IMAGE:-bigenlight/libero-pro:latest}"
OUTPUT_DIR="$(pwd)/test_outputs"
OOD_DATA_DIR="$(pwd)/ood_data"
SCRIPTS_DIR="$(pwd)/scripts"
VLA_URL="${VLA_SERVER_URL:-http://localhost:8400}"
CUDA_DEV="${CUDA_DEV:-1}"
TASKS=""
NUM_TRIALS=3
INIT_INDICES="0,1,2"
MAX_STEPS=520
REPLAN_STEPS=5
RESOLUTION=256

# ── CLI arg parsing ──────────────────────────────────────

usage() {
    cat <<USAGE
Usage: ./scripts/run_pi05_libero10_swap.sh [OPTIONS]

Options:
  --vla-url <url>         Pi0.5 server URL (default: \$VLA_SERVER_URL or http://localhost:8400)
  --gpu <id>              CUDA device index (default: \$CUDA_DEV or 1). GPU 0 is reserved.
  --tasks <csv>           Comma-separated task names to run (default: all tasks)
  --num-trials <N>        Number of trials per task (default: 3)
  --init-indices <csv>    Init state indices to use (default: 0,1,2)
  --max-steps <N>         Max steps per episode (default: 520)
  --replan-steps <N>      Steps between VLA replanning calls (default: 5)
  --resolution <N>        Camera resolution in pixels (default: 256)
  -h|--help               Show this help message
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --vla-url)
            VLA_URL="$2"
            shift 2 ;;
        --gpu)
            CUDA_DEV="$2"
            shift 2 ;;
        --tasks)
            TASKS="$2"
            shift 2 ;;
        --num-trials)
            NUM_TRIALS="$2"
            shift 2 ;;
        --init-indices)
            INIT_INDICES="$2"
            shift 2 ;;
        --max-steps)
            MAX_STEPS="$2"
            shift 2 ;;
        --replan-steps)
            REPLAN_STEPS="$2"
            shift 2 ;;
        --resolution)
            RESOLUTION="$2"
            shift 2 ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1 ;;
    esac
done

# ── GPU 0 guard ──────────────────────────────────────────

if [[ "$CUDA_DEV" == "0" ]]; then
    echo "ERROR: GPU 0 is reserved (soeun training). Pick GPU 1, 2, or 3."
    exit 1
fi

# ── Print resolved config ─────────────────────────────────

echo "=== run_pi05_libero10_swap.sh ==="
echo "  IMAGE        : $IMAGE"
echo "  VLA_URL      : $VLA_URL"
echo "  CUDA_DEV     : $CUDA_DEV"
echo "  TASKS        : ${TASKS:-(all)}"
echo "  NUM_TRIALS   : $NUM_TRIALS"
echo "  INIT_INDICES : $INIT_INDICES"
echo "  MAX_STEPS    : $MAX_STEPS"
echo "  REPLAN_STEPS : $REPLAN_STEPS"
echo "  RESOLUTION   : $RESOLUTION"
echo "  OUTPUT_DIR   : $OUTPUT_DIR"
echo ""

# ── Pre-flight checks ─────────────────────────────────────

echo "=== Pre-flight checks ==="

# 1. Docker binary
if ! command -v docker &>/dev/null; then
    echo "ERROR: docker not found. Install Docker first."
    exit 1
fi
echo "  Docker: $(docker --version | head -1)"

# 2. Image must already be present — do NOT auto-pull
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "ERROR: Image '$IMAGE' not found locally."
    echo "       Pull it manually with: docker pull $IMAGE"
    exit 1
fi
echo "  Image : $IMAGE (local)"

# 3. OOD data directories for libero_10_swap
if [[ ! -d "$OOD_DATA_DIR/bddl_files/libero_10_swap" ]]; then
    echo "ERROR: OOD bddl_files not found: $OOD_DATA_DIR/bddl_files/libero_10_swap"
    echo "       Ensure the ood_data/ directory is present and contains libero_10_swap bddl files."
    exit 1
fi
if [[ ! -d "$OOD_DATA_DIR/init_files/libero_10_swap" ]]; then
    echo "ERROR: OOD init_files not found: $OOD_DATA_DIR/init_files/libero_10_swap"
    echo "       Ensure the ood_data/ directory is present and contains libero_10_swap init files."
    exit 1
fi
echo "  OOD bddl : $OOD_DATA_DIR/bddl_files/libero_10_swap"
echo "  OOD init : $OOD_DATA_DIR/init_files/libero_10_swap"

# 4. Pi0.5 server health check
echo "  Checking pi05 server at $VLA_URL ..."
if ! curl -fsS --max-time 5 "${VLA_URL}/health" >/dev/null 2>&1; then
    echo ""
    echo "ERROR: pi05 server not reachable at $VLA_URL."
    echo "Start it with (on host):"
    echo "    docker run -d --rm --name pi05-server --gpus '\"device=2\"' --network host \\"
    echo "      -v ~/.cache/openpi:/openpi_assets \\"
    echo "      -v /home/theo/workspace/openpi:/app \\"
    echo "      bigenlight/openpi-pi05-http:latest"
    echo "Wait until 'docker logs pi05-server' shows 'Uvicorn running on http://0.0.0.0:8400', then re-run this script."
    exit 2
fi
echo "  Pi05 server: reachable"

# 5. Ensure output directory exists
mkdir -p "$OUTPUT_DIR"
echo "  Output dir: $OUTPUT_DIR"

echo ""
echo "=== Launching container ==="

# ── Build optional --tasks flag for the python invocation ─

TASKS_FLAG=""
if [ -n "${TASKS}" ]; then
    TASKS_FLAG="--tasks \"${TASKS}\""
fi

# ── Container run ─────────────────────────────────────────

docker run --rm \
    --gpus "\"device=${CUDA_DEV}\"" \
    --shm-size=8g \
    --network host \
    -e MUJOCO_GL=egl \
    -e VLA_SERVER_URL="${VLA_URL}" \
    -v "${OOD_DATA_DIR}:/tmp/ood_data:ro" \
    -v "${SCRIPTS_DIR}:/workspace/LIBERO-PRO/scripts:ro" \
    -v "${OUTPUT_DIR}:/workspace/LIBERO-PRO/test_outputs" \
    "${IMAGE}" \
    bash -c "
# ── OOD data merge (host ood_data/ -> container LIBERO-PRO) ──
cp -rn /tmp/ood_data/bddl_files/* /workspace/LIBERO-PRO/libero/libero/bddl_files/ 2>/dev/null || true
cp -rn /tmp/ood_data/init_files/* /workspace/LIBERO-PRO/libero/libero/init_files/ 2>/dev/null || true

# ── Ensure lightweight Python deps ──
pip install --quiet requests imageio imageio-ffmpeg >/dev/null 2>&1 || true

# ── Run eval ──
python scripts/libero10_swap_3task_eval.py \
    --vla-url \"${VLA_URL}\" \
    ${TASKS_FLAG} \
    --num-trials \"${NUM_TRIALS}\" \
    --init-state-indices \"${INIT_INDICES}\" \
    --max-steps \"${MAX_STEPS}\" \
    --replan-steps \"${REPLAN_STEPS}\" \
    --resolution \"${RESOLUTION}\" \
    --output-dir /workspace/LIBERO-PRO/test_outputs/pi05_libero10_swap
"

echo ""
echo "Outputs: $OUTPUT_DIR/pi05_libero10_swap/"
