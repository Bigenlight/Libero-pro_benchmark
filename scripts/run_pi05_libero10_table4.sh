#!/usr/bin/env bash
set -euo pipefail

# ── Pi0.5 LIBERO-10 Table 4 Reproduction — Multi-GPU Orchestrator ──
#
# Starts 3 pi05 HTTP servers on GPU 1/2/3 (ports 8401/8402/8403), spawns 3
# libero-pro eval containers each running libero10_table4_eval.py on a
# round-robin sharded subset of the 10 table-4 tasks, waits for all evals,
# runs the aggregate_table4.py aggregator, then cleans up.
#
# Environment variables (all overridable):
#   PI05_IMAGE       pi05 Docker image (default: bigenlight/openpi-pi05-http:latest)
#   LIBERO_IMAGE     libero-pro Docker image (default: bigenlight/libero-pro:latest)
#   OPENPI_ROOT      path to openpi repo on host (default: /home/theo/workspace/openpi)
#   OPENPI_CACHE     path to openpi checkpoint cache (default: ~/.cache/openpi)
#   PI05_GPUS        space-separated GPU ids (default: "1 2 3"). GPU 0 is FORBIDDEN.
#   SERVER_WAIT_SEC  max seconds to wait for each /health (default: 600)
#
# Usage examples:
#   cd /home/theo/workspace/Libero-pro_benchmark
#   ./scripts/run_pi05_libero10_table4.sh
#   ./scripts/run_pi05_libero10_table4.sh --dry-run
#   PI05_GPUS="1 2" SERVER_WAIT_SEC=300 ./scripts/run_pi05_libero10_table4.sh --variants-obj 3

# ── Config ───────────────────────────────────────────────────────────────────
PI05_IMAGE="${PI05_IMAGE:-bigenlight/openpi-pi05-http:latest}"
LIBERO_IMAGE="${LIBERO_IMAGE:-bigenlight/libero-pro:latest}"
OPENPI_ROOT="${OPENPI_ROOT:-/home/theo/workspace/openpi}"
OPENPI_CACHE="${OPENPI_CACHE:-$HOME/.cache/openpi}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PI05_GPUS_STR="${PI05_GPUS:-1 2 3}"
read -ra PI05_GPUS <<< "$PI05_GPUS_STR"
SERVER_WAIT_SEC="${SERVER_WAIT_SEC:-600}"

# ── Hard-coded 10 Table-4 tasks ───────────────────────────────────────────────
TASKS=(
  "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"
  "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it"
  "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it"
  "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"
  "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket"
  "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket"
  "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket"
  "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate"
  "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate"
  "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy"
)

# ── CLI flag defaults ─────────────────────────────────────────────────────────
VARIANTS_OBJ=5
VARIANTS_SEM=3
VARIANTS_TASK=5
TRIALS_ORI=10
TRIALS_POS=10
TRIALS_OTHER=5
MAX_STEPS=520
REPLAN_STEPS=5
SEED=7
PERTURBATIONS="ori,pos,obj,sem,task"
DRY_RUN_FLAG=""

usage() {
    cat <<USAGE
Usage: $(basename "$0") [OPTIONS]

Options:
  --variants-obj N       Object variants per task  (default: 5)
  --variants-sem N       Semantic variants          (default: 3)
  --variants-task N      Task variants              (default: 5)
  --trials-ori N         Trials for ori perturbation (default: 10)
  --trials-pos N         Trials for pos perturbation (default: 10)
  --trials-other N       Trials for other perturbations (default: 5)
  --max-steps N          Max steps per episode      (default: 520)
  --replan-steps N       Steps between VLA replans  (default: 5)
  --seed N               Random seed                (default: 7)
  --perturbations CSV    Perturbation types         (default: ori,pos,obj,sem,task)
  --dry-run              Pass --dry-run to eval script
  -h|--help              Show this help
USAGE
}

# ── Parse CLI flags ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --variants-obj)   VARIANTS_OBJ="$2";   shift 2 ;;
        --variants-sem)   VARIANTS_SEM="$2";   shift 2 ;;
        --variants-task)  VARIANTS_TASK="$2";  shift 2 ;;
        --trials-ori)     TRIALS_ORI="$2";     shift 2 ;;
        --trials-pos)     TRIALS_POS="$2";     shift 2 ;;
        --trials-other)   TRIALS_OTHER="$2";   shift 2 ;;
        --max-steps)      MAX_STEPS="$2";      shift 2 ;;
        --replan-steps)   REPLAN_STEPS="$2";   shift 2 ;;
        --seed)           SEED="$2";           shift 2 ;;
        --perturbations)  PERTURBATIONS="$2";  shift 2 ;;
        --dry-run)        DRY_RUN_FLAG="--dry-run"; shift ;;
        -h|--help)        usage; exit 0 ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1 ;;
    esac
done

# ── Print resolved config ─────────────────────────────────────────────────────
echo "=== run_pi05_libero10_table4.sh ==="
echo "  PI05_IMAGE      : $PI05_IMAGE"
echo "  LIBERO_IMAGE    : $LIBERO_IMAGE"
echo "  OPENPI_ROOT     : $OPENPI_ROOT"
echo "  OPENPI_CACHE    : $OPENPI_CACHE"
echo "  REPO_ROOT       : $REPO_ROOT"
echo "  PI05_GPUS       : ${PI05_GPUS[*]}"
echo "  SERVER_WAIT_SEC : $SERVER_WAIT_SEC"
echo "  VARIANTS_OBJ    : $VARIANTS_OBJ"
echo "  VARIANTS_SEM    : $VARIANTS_SEM"
echo "  VARIANTS_TASK   : $VARIANTS_TASK"
echo "  TRIALS_ORI      : $TRIALS_ORI"
echo "  TRIALS_POS      : $TRIALS_POS"
echo "  TRIALS_OTHER    : $TRIALS_OTHER"
echo "  MAX_STEPS       : $MAX_STEPS"
echo "  REPLAN_STEPS    : $REPLAN_STEPS"
echo "  SEED            : $SEED"
echo "  PERTURBATIONS   : $PERTURBATIONS"
echo "  DRY_RUN_FLAG    : ${DRY_RUN_FLAG:-(none)}"
echo ""

# ── GPU 0 hard-reject ─────────────────────────────────────────────────────────
for gpu in "${PI05_GPUS[@]}"; do
    if [[ "$gpu" == "0" ]]; then
        echo "ERROR: GPU 0 is reserved for another user's training job." >&2
        echo "       Remove GPU 0 from PI05_GPUS and use GPUs 1, 2, or 3." >&2
        exit 1
    fi
done

# ── Round-robin task sharding ────────────────────────────────────────────────
N_GPUS=${#PI05_GPUS[@]}
N_TASKS=${#TASKS[@]}

# SHARD_TASKS_CSV[shard_idx] = comma-joined task names for that shard
declare -a SHARD_TASKS_CSV
for (( s=0; s<N_GPUS; s++ )); do
    SHARD_TASKS_CSV[$s]=""
done

for (( i=0; i<N_TASKS; i++ )); do
    shard=$(( i % N_GPUS ))
    task="${TASKS[$i]}"
    if [[ -z "${SHARD_TASKS_CSV[$shard]}" ]]; then
        SHARD_TASKS_CSV[$shard]="$task"
    else
        SHARD_TASKS_CSV[$shard]="${SHARD_TASKS_CSV[$shard]},${task}"
    fi
done

echo "[sharding] ${N_TASKS} tasks across ${N_GPUS} GPU(s) (round-robin):"
for (( s=0; s<N_GPUS; s++ )); do
    echo "  shard $s -> GPU ${PI05_GPUS[$s]} port $((8400 + ${PI05_GPUS[$s]}))"
    echo "             tasks: ${SHARD_TASKS_CSV[$s]}"
done
echo ""

# ── Cleanup trap ─────────────────────────────────────────────────────────────
cleanup() {
    echo "[cleanup] stopping containers..."
    for gpu in "${PI05_GPUS[@]}"; do
        docker rm -f "pi05-table4-g${gpu}" "libero-table4-g${gpu}" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

# ── [0/8] Initial cleanup (idempotent) ───────────────────────────────────────
echo "[0/8] removing any stale containers..."
for gpu in "${PI05_GPUS[@]}"; do
    docker rm -f "pi05-table4-g${gpu}" "libero-table4-g${gpu}" 2>/dev/null || true
done

# ── [1/8] Preflight checks ───────────────────────────────────────────────────
echo "[1/8] preflight checks..."

if ! command -v docker &>/dev/null; then
    echo "ERROR: docker not found in PATH." >&2
    exit 1
fi
echo "  docker: $(docker --version | head -1)"

if ! docker image inspect "$PI05_IMAGE" >/dev/null 2>&1; then
    echo "ERROR: Pi05 image '$PI05_IMAGE' not found locally." >&2
    echo "       Pull it first: docker pull $PI05_IMAGE" >&2
    exit 1
fi
echo "  pi05 image : $PI05_IMAGE (local)"

if ! docker image inspect "$LIBERO_IMAGE" >/dev/null 2>&1; then
    echo "ERROR: LIBERO image '$LIBERO_IMAGE' not found locally." >&2
    echo "       Pull it first: docker pull $LIBERO_IMAGE" >&2
    exit 1
fi
echo "  libero image: $LIBERO_IMAGE (local)"

if [[ ! -d "$OPENPI_CACHE" ]]; then
    echo "ERROR: OPENPI_CACHE directory not found: $OPENPI_CACHE" >&2
    exit 1
fi
echo "  openpi cache: $OPENPI_CACHE"

if [[ ! -d "$REPO_ROOT/ood_data/bddl_files/libero_10_swap" ]]; then
    echo "ERROR: OOD bddl_files not found: $REPO_ROOT/ood_data/bddl_files/libero_10_swap" >&2
    exit 1
fi
echo "  ood_data    : $REPO_ROOT/ood_data/bddl_files/libero_10_swap"

if ! command -v curl &>/dev/null; then
    echo "ERROR: curl not found in PATH." >&2
    exit 1
fi
echo "  curl: $(curl --version | head -1)"
echo ""

# ── [2/8] Timestamp + output dir ─────────────────────────────────────────────
echo "[2/8] creating output directory..."
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR_HOST="$REPO_ROOT/test_outputs/pi05_libero10_table4_repro/$TIMESTAMP"
OUT_DIR_CONTAINER="/workspace/LIBERO-PRO/test_outputs/pi05_libero10_table4_repro/$TIMESTAMP"
mkdir -p "$OUT_DIR_HOST"
echo "  host      : $OUT_DIR_HOST"
echo "  container : $OUT_DIR_CONTAINER"
echo ""

# ── [3/8] Start pi05 servers ─────────────────────────────────────────────────
echo "[3/8] starting pi05 servers on GPU(s) ${PI05_GPUS[*]}..."
for (( idx=0; idx<N_GPUS; idx++ )); do
    gpu="${PI05_GPUS[$idx]}"
    PORT=$((8400 + gpu))
    docker run -d --name "pi05-table4-g${gpu}" --network host \
        --gpus "device=${gpu}" --shm-size=4g \
        -v "$OPENPI_ROOT:/app" \
        -v "$OPENPI_CACHE:/openpi_assets" \
        -e PI05_HTTP_PORT="$PORT" \
        "$PI05_IMAGE" >/dev/null
    echo "  started pi05-table4-g${gpu} on GPU ${gpu}, port ${PORT}"
done
echo ""

# ── [4/8] Wait for /health on all servers ────────────────────────────────────
echo "[4/8] waiting for pi05 servers to become ready (timeout: ${SERVER_WAIT_SEC}s each)..."
MAX_POLLS=$(( SERVER_WAIT_SEC / 3 ))
for (( idx=0; idx<N_GPUS; idx++ )); do
    gpu="${PI05_GPUS[$idx]}"
    PORT=$((8400 + gpu))
    echo "  polling http://localhost:${PORT}/health ..."
    ready=0
    for (( attempt=1; attempt<=MAX_POLLS; attempt++ )); do
        if curl -fsS --max-time 3 "http://localhost:${PORT}/health" >/dev/null 2>&1; then
            echo "  port ${PORT} ready (attempt ${attempt})"
            ready=1
            break
        fi
        sleep 3
    done
    if [[ $ready -eq 0 ]]; then
        echo "ERROR: pi05 server on port ${PORT} did not become ready within ${SERVER_WAIT_SEC}s." >&2
        echo "--- last 80 lines of pi05-table4-g${gpu} logs ---" >&2
        docker logs "pi05-table4-g${gpu}" 2>&1 | tail -80 >&2
        exit 1
    fi
done
echo ""

# ── [5/8] Spawn eval containers in parallel ───────────────────────────────────
echo "[5/8] spawning ${N_GPUS} libero eval container(s) in parallel..."
for (( idx=0; idx<N_GPUS; idx++ )); do
    gpu="${PI05_GPUS[$idx]}"
    PORT=$((8400 + gpu))
    SHARD_TASKS="${SHARD_TASKS_CSV[$idx]}"
    docker run -d --name "libero-table4-g${gpu}" \
        --gpus "device=${gpu}" --shm-size=8g --network host \
        -e MUJOCO_GL=egl \
        -e VLA_SERVER_URL="http://localhost:${PORT}" \
        -v "$REPO_ROOT/ood_data:/tmp/ood_data:ro" \
        -v "$REPO_ROOT/scripts:/workspace/LIBERO-PRO/scripts:ro" \
        -v "$REPO_ROOT/test_outputs:/workspace/LIBERO-PRO/test_outputs" \
        "$LIBERO_IMAGE" \
        bash -c "cp -rn /tmp/ood_data/bddl_files/* /workspace/LIBERO-PRO/libero/libero/bddl_files/ 2>/dev/null || true; \
                 cp -rn /tmp/ood_data/init_files/* /workspace/LIBERO-PRO/libero/libero/init_files/ 2>/dev/null || true; \
                 pip install --quiet requests imageio imageio-ffmpeg >/dev/null 2>&1 || true; \
                 python /workspace/LIBERO-PRO/scripts/libero10_table4_eval.py \
                   --vla-url \"http://localhost:${PORT}\" \
                   --tasks \"${SHARD_TASKS}\" \
                   --output-dir \"${OUT_DIR_CONTAINER}\" \
                   --shard-tag \"g${gpu}\" \
                   --variants-obj ${VARIANTS_OBJ} \
                   --variants-sem ${VARIANTS_SEM} \
                   --variants-task ${VARIANTS_TASK} \
                   --trials-ori ${TRIALS_ORI} \
                   --trials-pos ${TRIALS_POS} \
                   --trials-other ${TRIALS_OTHER} \
                   --max-steps ${MAX_STEPS} \
                   --replan-steps ${REPLAN_STEPS} \
                   --seed ${SEED} \
                   --perturbations \"${PERTURBATIONS}\" \
                   ${DRY_RUN_FLAG}" >/dev/null
    echo "  started libero-table4-g${gpu} on GPU ${gpu} (port ${PORT})"
    echo "    tasks: ${SHARD_TASKS}"
done
echo ""

# ── [6/8] Wait for all eval containers ───────────────────────────────────────
echo "[6/8] waiting for eval shards to finish..."
echo "  (follow logs: docker logs -f libero-table4-g${PI05_GPUS[0]})"
echo ""
declare -A EXIT_CODES
for gpu in "${PI05_GPUS[@]}"; do
    code=$(docker wait "libero-table4-g${gpu}")
    EXIT_CODES[$gpu]=$code
    docker logs "libero-table4-g${gpu}" > "$OUT_DIR_HOST/g${gpu}.log" 2>&1 || true
    echo "  shard g${gpu} exit=${code} (log: $OUT_DIR_HOST/g${gpu}.log)"
done
echo ""
# Don't abort on non-zero — continue to aggregator regardless.

# ── [7/8] Aggregate results ───────────────────────────────────────────────────
echo "[7/8] running aggregator..."
docker run --rm --shm-size=2g \
    -v "$REPO_ROOT:/repo" \
    -e MUJOCO_GL=egl \
    "$LIBERO_IMAGE" \
    bash -c "pip install --quiet pandas >/dev/null 2>&1 || true; \
             python /repo/scripts/aggregate_table4.py --run-dir \"/repo/test_outputs/pi05_libero10_table4_repro/$TIMESTAMP\""
echo ""

# ── [8/8] Write top-level run_meta.json ──────────────────────────────────────
echo "[8/8] writing run_meta.json..."
GPUS_PY=$(IFS=,; echo "${PI05_GPUS[*]}")
python3 -c "
import json, glob, os
d = '$OUT_DIR_HOST'
shards = []
for fp in sorted(glob.glob(os.path.join(d, 'run_meta_g*.json'))):
    with open(fp) as f:
        shards.append(json.load(f))
gpus = [${GPUS_PY}]
meta = {'timestamp': '$TIMESTAMP', 'gpus': gpus, 'shards': shards}
with open(os.path.join(d, 'run_meta.json'), 'w') as f:
    json.dump(meta, f, indent=2, default=str)
print('  wrote', os.path.join(d, 'run_meta.json'))
"

echo ""
echo "=========================================================="
echo "  DONE"
echo "  Output dir  : $OUT_DIR_HOST"
echo "  Shard logs  : $OUT_DIR_HOST/g*.log"
echo "  Run meta    : $OUT_DIR_HOST/run_meta.json"
echo "  Table 4     : $OUT_DIR_HOST/table4_reproduction.txt"
echo "=========================================================="
echo ""

# Print shard exit codes summary
echo "Shard exit codes:"
for gpu in "${PI05_GPUS[@]}"; do
    code="${EXIT_CODES[$gpu]:-N/A}"
    status="OK"
    if [[ "$code" != "0" ]]; then
        status="FAIL (rc=$code)"
    fi
    echo "  g${gpu}: $status"
done
echo ""

# Print table4 reproduction result if it exists
if [[ -f "$OUT_DIR_HOST/table4_reproduction.txt" ]]; then
    cat "$OUT_DIR_HOST/table4_reproduction.txt"
else
    echo "(table4_reproduction.txt not found — aggregator may have failed or not yet written it)"
fi
