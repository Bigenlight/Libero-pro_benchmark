#!/usr/bin/env bash
# Multi-GPU libero_spatial full evaluation for pi0.5.
#
# Assumes:
#   - 3 physical GPUs available (default: 1, 2, 3) -- GPU 0 is intentionally
#     LEFT ALONE because it is occupied by another user's training job.
#   - Docker image `bigenlight/openpi-pi05-http:latest` present locally.
#   - Docker image `bigenlight/libero-pro:latest` present locally.
#   - Host has `~/.cache/openpi` populated (checkpoint) — first run downloads.
#   - Current working directory is the Libero-pro_benchmark repo root.
#
# What it does:
#   1. Starts 3 pi0.5 HTTP servers on GPU ${PI05_GPUS:-1 2 3},
#      ports 8401/8402/8403.
#   2. Waits for all 3 /health to return status=ok.
#   3. Launches 3 libero eval shards in parallel, each pinned to the same GPU
#      as its paired pi0.5 server, each covering a slice of the 10 tasks:
#        - shard 0 -> tasks 0,1,2,3 (4 tasks) on GPU1  via :8401
#        - shard 1 -> tasks 4,5,6   (3 tasks) on GPU2  via :8402
#        - shard 2 -> tasks 7,8,9   (3 tasks) on GPU3  via :8403
#   4. Waits for all 3 eval containers to finish.
#   5. Merges the 3 per-shard summary.json files into
#      test_outputs/eval_multigpu/libero_spatial_<timestamp>/
#      libero_spatial_full_summary.json.
#   6. Copies the final summary + per-shard summaries to the host workspace root.
#   7. Tears everything down.
#
# Usage:
#   cd /home/theo/workspace/Libero-pro_benchmark
#   ./scripts/run_libero_spatial_multigpu.sh                   # default 20 trials
#   NUM_TRIALS=10 ./scripts/run_libero_spatial_multigpu.sh     # quick 10 trials

set -euo pipefail

# ----- Config -------------------------------------------------------------- #
PI05_IMAGE="${PI05_IMAGE:-bigenlight/openpi-pi05-http:latest}"
LIBERO_IMAGE="${LIBERO_IMAGE:-bigenlight/libero-pro:latest}"
OPENPI_ROOT="${OPENPI_ROOT:-/home/theo/workspace/openpi}"
OPENPI_CACHE="${OPENPI_CACHE:-$HOME/.cache/openpi}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
NUM_TRIALS="${NUM_TRIALS:-20}"
SUITE="${SUITE:-libero_spatial}"
# The three GPUs this script is allowed to use (GPU 0 is reserved for soeun).
GPUS=(${PI05_GPUS:-1 2 3})
PORTS=(8401 8402 8403)

if [[ ${#GPUS[@]} -ne 3 ]]; then
    echo "expected 3 GPUs, got: ${GPUS[*]}" >&2
    exit 1
fi

# Hard safety: GPU 0 is reserved for another user's training job.
for gpu in "${GPUS[@]}"; do
    if [[ "$gpu" -eq 0 ]]; then
        echo "ERROR: GPU 0 is reserved for another user (soeun). Remove it from PI05_GPUS." >&2
        exit 1
    fi
done

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR_HOST="$REPO_ROOT/test_outputs/eval_multigpu/${SUITE}_${TIMESTAMP}"
mkdir -p "$OUT_DIR_HOST"

SHARD0_IDS="0,1,2,3"   # 4 tasks
SHARD1_IDS="4,5,6"     # 3 tasks
SHARD2_IDS="7,8,9"     # 3 tasks
SHARD_IDS=("$SHARD0_IDS" "$SHARD1_IDS" "$SHARD2_IDS")

PI05_NAMES=(pi05-g${GPUS[0]} pi05-g${GPUS[1]} pi05-g${GPUS[2]})
EVAL_NAMES=(libero-eval-g${GPUS[0]} libero-eval-g${GPUS[1]} libero-eval-g${GPUS[2]})

cleanup() {
    echo "[cleanup] stopping containers..."
    for c in "${PI05_NAMES[@]}" "${EVAL_NAMES[@]}"; do
        docker rm -f "$c" >/dev/null 2>&1 || true
    done
}
trap cleanup EXIT INT TERM

# ----- 1. Stop any stale containers ---------------------------------------- #
cleanup

# ----- 2. Start 3 pi0.5 servers -------------------------------------------- #
echo "[1/6] starting pi0.5 servers on GPUs ${GPUS[*]} ..."
for i in 0 1 2; do
    g="${GPUS[$i]}"
    p="${PORTS[$i]}"
    name="${PI05_NAMES[$i]}"
    docker run -d --name "$name" \
        --network host \
        --gpus "device=${g}" \
        --shm-size=4g \
        -v "$OPENPI_ROOT:/app" \
        -v "$OPENPI_CACHE:/openpi_assets" \
        -e PI05_HTTP_PORT="$p" \
        "$PI05_IMAGE" >/dev/null
    echo "  started $name on GPU $g port $p"
done

# ----- 3. Wait for all /health ok ------------------------------------------ #
echo "[2/6] waiting for all 3 pi0.5 servers to become ready..."
for i in 0 1 2; do
    p="${PORTS[$i]}"
    for attempt in $(seq 1 120); do
        if curl -sf "http://localhost:$p/health" | grep -q '"status":"ok"'; then
            echo "  port $p ready"
            break
        fi
        if [[ $attempt -eq 120 ]]; then
            echo "ERROR: server on port $p did not become ready" >&2
            docker logs "${PI05_NAMES[$i]}" 2>&1 | tail -60 >&2
            exit 1
        fi
        sleep 3
    done
done

# ----- 4. Launch 3 eval shards in parallel --------------------------------- #
echo "[3/6] launching eval shards..."
OUT_DIR_CONTAINER=/workspace/LIBERO-PRO/test_outputs/eval_multigpu/${SUITE}_${TIMESTAMP}

for i in 0 1 2; do
    g="${GPUS[$i]}"
    p="${PORTS[$i]}"
    name="${EVAL_NAMES[$i]}"
    ids="${SHARD_IDS[$i]}"
    docker run -d --name "$name" \
        --network host \
        --gpus "device=${g}" \
        --shm-size=4g \
        -e MUJOCO_GL=egl \
        -e VLA_SERVER_URL="http://localhost:$p" \
        -e PYTHONUNBUFFERED=1 \
        -v "$REPO_ROOT/ood_data:/tmp/ood_data:ro" \
        -v "$REPO_ROOT/scripts:/workspace/LIBERO-PRO/scripts:ro" \
        -v "$REPO_ROOT/test_outputs:/workspace/LIBERO-PRO/test_outputs" \
        "$LIBERO_IMAGE" \
        bash -c "
            if [ -d /tmp/ood_data ]; then
                cp -rn /tmp/ood_data/bddl_files/* /workspace/LIBERO-PRO/libero/libero/bddl_files/ 2>/dev/null || true
                cp -rn /tmp/ood_data/init_files/* /workspace/LIBERO-PRO/libero/libero/init_files/ 2>/dev/null || true
            fi
            pip install --quiet requests imageio imageio-ffmpeg 2>/dev/null || true
            python /workspace/LIBERO-PRO/scripts/libero_vla_eval.py \
                --vla-url http://localhost:$p \
                --suite $SUITE \
                --task-ids $ids \
                --num-trials $NUM_TRIALS \
                --output-dir $OUT_DIR_CONTAINER \
                --shard-tag g$g \
                --no-video
        " >/dev/null
    echo "  started $name on GPU $g (tasks $ids, trials $NUM_TRIALS)"
done

# ----- 5. Wait for all eval containers to exit ----------------------------- #
echo "[4/6] waiting for eval shards to finish (follow logs with: docker logs -f ${EVAL_NAMES[0]})"
fail=0
for name in "${EVAL_NAMES[@]}"; do
    rc=$(docker wait "$name")
    status="OK"
    if [[ "$rc" != "0" ]]; then
        status="FAIL rc=$rc"
        fail=1
    fi
    echo "  $name -> $status"
    docker logs "$name" 2>&1 | tail -20 >"$OUT_DIR_HOST/$name.log"
done

if [[ $fail -ne 0 ]]; then
    echo "ERROR: one or more shards failed — see $OUT_DIR_HOST/*.log" >&2
    exit 1
fi

# ----- 6. Merge per-shard summaries ---------------------------------------- #
echo "[5/6] merging summary.json across shards..."
shard_jsons=()
for g in "${GPUS[@]}"; do
    # libero_vla_eval writes: <OUT_DIR_CONTAINER>/<suite>_<ts>_<shard_tag>/summary.json
    # On the host the same path lives under OUT_DIR_HOST.
    match=$(find "$OUT_DIR_HOST" -type d -name "${SUITE}_*_g${g}" | head -1)
    if [[ -z "$match" ]]; then
        echo "ERROR: no shard output for GPU $g under $OUT_DIR_HOST" >&2
        exit 1
    fi
    shard_jsons+=("$match/summary.json")
done

FINAL_SUMMARY="$OUT_DIR_HOST/libero_spatial_full_summary.json"
# Rewrite host absolute paths -> container paths (/repo/...).
merge_inputs=()
for sj in "${shard_jsons[@]}"; do
    merge_inputs+=("/repo/${sj#"$REPO_ROOT/"}")
done
docker run --rm -i \
    -v "$REPO_ROOT:/repo" \
    "$LIBERO_IMAGE" \
    python /repo/scripts/merge_eval_summaries.py \
        --inputs "${merge_inputs[@]}" \
        --output "/repo/${FINAL_SUMMARY#"$REPO_ROOT/"}"

# ----- 7. Publish final summary to workspace root -------------------------- #
echo "[6/6] publishing final summary to workspace root..."
WS_ROOT="${WS_ROOT:-/home/theo/workspace}"
cp "$FINAL_SUMMARY" "$WS_ROOT/libero_spatial_full_summary.json"
echo ""
echo "=========================================================="
echo "  DONE"
echo "  Shard summaries: $OUT_DIR_HOST/*/summary.json"
echo "  Final summary:   $FINAL_SUMMARY"
echo "  Workspace copy:  $WS_ROOT/libero_spatial_full_summary.json"
echo "=========================================================="
cat "$FINAL_SUMMARY" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"{d['suite']}  {d['total_successes']}/{d['total_episodes']}  ({100*d['success_rate']:.1f}%)\")" 2>/dev/null || true
