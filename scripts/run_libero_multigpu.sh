#!/usr/bin/env bash
# Multi-GPU LIBERO full evaluation for pi0.5 — N-GPU generic.
#
# Assumes:
#   - N physical GPUs available (default: 1 2 3). GPU 0 is REJECTED by a hard
#     safety check because it is reserved for another user's training job.
#   - Docker image `bigenlight/openpi-pi05-http:latest` present locally.
#   - Docker image `bigenlight/libero-pro:latest` present locally.
#   - Host has `~/.cache/openpi` populated (checkpoint) — first run downloads.
#   - Current working directory is the Libero-pro_benchmark repo root.
#
# Environment variables:
#   PI05_GPUS    space-separated GPU ids to use (default: "1 2 3"). Any count >= 1.
#                GPU 0 is forbidden. Ports are derived as 8400 + gpu_id.
#   SUITE        libero suite name (default: libero_spatial). Examples:
#                libero_spatial, libero_object, libero_goal, libero_10,
#                libero_spatial_task, libero_goal_swap, ...
#   NUM_TRIALS   trials per task (default: 20).
#   SAVE_VIDEO   0 or 1 (default: 0). When 1, rollouts are recorded to mp4
#                and consolidated under $OUT_DIR_HOST/videos/. When 0 (default),
#                `--no-video` is passed to libero_vla_eval.py.
#   PI05_IMAGE / LIBERO_IMAGE / OPENPI_ROOT / OPENPI_CACHE / WS_ROOT — overrides.
#
# What it does:
#   1. Starts N pi0.5 HTTP servers, one per GPU, on ports 8400+gpu_id.
#   2. Waits for all /health endpoints to return status=ok.
#   3. Launches N libero eval shards in parallel, each pinned to its GPU, with
#      the 10 suite tasks split evenly across shards (see "Task sharding" below).
#   4. Waits for all eval containers to finish.
#   5. Merges the per-shard summary.json files into
#      test_outputs/eval_multigpu/${SUITE}_<timestamp>/${SUITE}_full_summary.json.
#   6. Copies the final summary to $WS_ROOT/${SUITE}_full_summary.json.
#   7. If SAVE_VIDEO=1, consolidates per-shard videos/ into $OUT_DIR_HOST/videos/.
#   8. Tears everything down.
#
# Task sharding:
#   10 tasks are split across N GPUs using integer division with remainder
#   distributed to the FIRST `rem` shards (so earlier shards get the extra task).
#     N=1 -> [0-9]
#     N=2 -> [0-4], [5-9]
#     N=3 -> [0-3], [4-6], [7-9]          (matches legacy behavior)
#     N=4 -> [0-2], [3-4], [5-6], [7-9]
#
# Usage examples:
#   cd /home/theo/workspace/Libero-pro_benchmark
#   # 1-GPU run on GPU 1, default suite, 20 trials:
#   PI05_GPUS="1" ./scripts/run_libero_multigpu.sh
#   # 2-GPU run on GPUs 2,3, libero_object suite, quick 10 trials:
#   PI05_GPUS="2 3" SUITE=libero_object NUM_TRIALS=10 ./scripts/run_libero_multigpu.sh
#   # 3-GPU run on libero_spatial_task OOD suite WITH video capture:
#   PI05_GPUS="1 2 3" SUITE=libero_spatial_task SAVE_VIDEO=1 \
#       ./scripts/run_libero_multigpu.sh

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
SAVE_VIDEO="${SAVE_VIDEO:-0}"

# GPUs this script is allowed to use (GPU 0 is reserved — hard rejected below).
GPUS=(${PI05_GPUS:-1 2 3})
N_GPUS=${#GPUS[@]}

if [[ $N_GPUS -lt 1 ]]; then
    echo "ERROR: PI05_GPUS is empty. Need at least 1 GPU." >&2
    exit 1
fi

# Hard safety: GPU 0 is reserved for another user's training job.
for gpu in "${GPUS[@]}"; do
    if [[ "$gpu" -eq 0 ]]; then
        echo "ERROR: GPU 0 is reserved for another user (soeun). Remove it from PI05_GPUS." >&2
        exit 1
    fi
done

# Derive ports = 8400 + gpu_id.
PORTS=()
for gpu in "${GPUS[@]}"; do
    PORTS+=($((8400 + gpu)))
done

# Container names.
PI05_NAMES=()
EVAL_NAMES=()
for gpu in "${GPUS[@]}"; do
    PI05_NAMES+=("pi05-g${gpu}")
    EVAL_NAMES+=("libero-eval-g${gpu}")
done

# ----- Task sharding: split 10 tasks across N GPUs ------------------------- #
# Integer division with remainder going to the first `rem` shards.
NUM_TASKS=10
base=$(( NUM_TASKS / N_GPUS ))
rem=$(( NUM_TASKS % N_GPUS ))
SHARD_IDS=()
cursor=0
for (( i=0; i<N_GPUS; i++ )); do
    count=$base
    if [[ $i -lt $rem ]]; then
        count=$(( count + 1 ))
    fi
    if [[ $count -le 0 ]]; then
        echo "ERROR: shard $i got 0 tasks (N_GPUS=$N_GPUS > NUM_TASKS=$NUM_TASKS?)" >&2
        exit 1
    fi
    ids=""
    for (( j=0; j<count; j++ )); do
        if [[ -z "$ids" ]]; then
            ids="$cursor"
        else
            ids="${ids},${cursor}"
        fi
        cursor=$(( cursor + 1 ))
    done
    SHARD_IDS+=("$ids")
done

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR_HOST="$REPO_ROOT/test_outputs/eval_multigpu/${SUITE}_${TIMESTAMP}"
mkdir -p "$OUT_DIR_HOST"

echo "[config] GPUs=${GPUS[*]}  ports=${PORTS[*]}  suite=$SUITE  trials=$NUM_TRIALS  save_video=$SAVE_VIDEO"
for (( i=0; i<N_GPUS; i++ )); do
    echo "[config]   shard $i -> GPU ${GPUS[$i]} port ${PORTS[$i]} tasks=[${SHARD_IDS[$i]}]"
done

cleanup() {
    echo "[cleanup] stopping containers..."
    for c in "${PI05_NAMES[@]}" "${EVAL_NAMES[@]}"; do
        docker rm -f "$c" >/dev/null 2>&1 || true
    done
}
trap cleanup EXIT INT TERM

# ----- 1. Stop any stale containers ---------------------------------------- #
cleanup

# ----- 2. Start N pi0.5 servers -------------------------------------------- #
echo "[1/6] starting pi0.5 servers on GPUs ${GPUS[*]} ..."
for (( i=0; i<N_GPUS; i++ )); do
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
echo "[2/6] waiting for all $N_GPUS pi0.5 servers to become ready..."
for (( i=0; i<N_GPUS; i++ )); do
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

# ----- 4. Launch N eval shards in parallel --------------------------------- #
echo "[3/6] launching eval shards..."
OUT_DIR_CONTAINER=/workspace/LIBERO-PRO/test_outputs/eval_multigpu/${SUITE}_${TIMESTAMP}

# Build the optional --no-video arg once.
if [[ "$SAVE_VIDEO" == "1" ]]; then
    VIDEO_ARG=""
else
    VIDEO_ARG="--no-video"
fi

for (( i=0; i<N_GPUS; i++ )); do
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
                $VIDEO_ARG
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
shard_dirs=()
for g in "${GPUS[@]}"; do
    # libero_vla_eval writes: <OUT_DIR_CONTAINER>/<suite>_<ts>_<shard_tag>/summary.json
    # On the host the same path lives under OUT_DIR_HOST.
    match=$(find "$OUT_DIR_HOST" -type d -name "${SUITE}_*_g${g}" | head -1)
    if [[ -z "$match" ]]; then
        echo "ERROR: no shard output for GPU $g under $OUT_DIR_HOST" >&2
        exit 1
    fi
    shard_jsons+=("$match/summary.json")
    shard_dirs+=("$match")
done

FINAL_SUMMARY="$OUT_DIR_HOST/${SUITE}_full_summary.json"
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

# ----- 6b. Consolidate videos (if SAVE_VIDEO=1) ---------------------------- #
if [[ "$SAVE_VIDEO" == "1" ]]; then
    echo "[5b/6] consolidating rollout videos into $OUT_DIR_HOST/videos/ ..."
    mkdir -p "$OUT_DIR_HOST/videos"
    for d in "${shard_dirs[@]}"; do
        if [[ -d "$d/videos" ]]; then
            # Flatten: copy every mp4 into the unified dir (filenames already
            # include task_segment + suffix so collisions are unlikely).
            find "$d/videos" -type f -name '*.mp4' -exec cp -n {} "$OUT_DIR_HOST/videos/" \; || true
        fi
    done
    n_videos=$(find "$OUT_DIR_HOST/videos" -type f -name '*.mp4' 2>/dev/null | wc -l)
    echo "  consolidated $n_videos mp4 file(s)"
fi

# ----- 7. Publish final summary to workspace root -------------------------- #
echo "[6/6] publishing final summary to workspace root..."
WS_ROOT="${WS_ROOT:-/home/theo/workspace}"
cp "$FINAL_SUMMARY" "$WS_ROOT/${SUITE}_full_summary.json"
echo ""
echo "=========================================================="
echo "  DONE"
echo "  Shard summaries: $OUT_DIR_HOST/*/summary.json"
echo "  Final summary:   $FINAL_SUMMARY"
echo "  Workspace copy:  $WS_ROOT/${SUITE}_full_summary.json"
if [[ "$SAVE_VIDEO" == "1" ]]; then
    echo "  Videos:          $OUT_DIR_HOST/videos/"
fi
echo "=========================================================="
cat "$FINAL_SUMMARY" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"{d['suite']}  {d['total_successes']}/{d['total_episodes']}  ({100*d['success_rate']:.1f}%)\")" 2>/dev/null || true
