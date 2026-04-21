#!/usr/bin/env bash
set -euo pipefail

# ── LIBERO / LIBERO-PRO 로컬 Docker 검증 스크립트 ──
# 사용법:
#   ./run.sh                          # 기본 (EGL, 자동 osmesa fallback)
#   ./run.sh --api-key sk-ant-xxxx    # Claude Code API 키 방식
#   ./run.sh --skip-video             # 비디오 테스트 스킵
#   ./run.sh --shell                  # 테스트 대신 bash 셸 진입

IMAGE="${LIBERO_IMAGE:-bigenlight/libero-pro:latest}"
OUTPUT_DIR="$(pwd)/test_outputs"
OOD_DATA_DIR="$(pwd)/ood_data"
LIBERO_PRO_SRC="$(pwd)/LIBERO-PRO"
LIBERO_SRC="$(pwd)/LIBERO"
EXTRA_DOCKER_ARGS=""
EXTRA_TEST_ARGS=""
SHELL_MODE=false
VLA_EVAL_MODE=false
VLA_EVAL_SUITE="libero_spatial"
VLA_URL="${VLA_SERVER_URL:-http://localhost:8400}"
VLA_NUM_TASKS=2
VLA_NUM_TRIALS=1

# ── 인자 파싱 ────────────────────────────────────────────

usage() {
    cat <<USAGE
Usage:
  ./run.sh                          # default: run test_local.py (EGL → osmesa fallback)
  ./run.sh --shell                  # interactive bash shell inside the container
  ./run.sh --skip-video             # skip video test
  ./run.sh --skip-pro               # skip LIBERO-PRO OOD tests

VLA eval mode:
  ./run.sh --vla-eval <suite> [--vla-url http://host:port] [--vla-num-tasks N] [--vla-num-trials N]
    example: ./run.sh --vla-eval libero_spatial --vla-url http://localhost:8400
    example: ./run.sh --vla-eval libero_goal_swap --vla-num-tasks 2

In VLA eval mode the container uses --network host so it can reach the VLA
model server on localhost.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --api-key)
            EXTRA_DOCKER_ARGS="$EXTRA_DOCKER_ARGS -e ANTHROPIC_API_KEY=$2"
            shift 2 ;;
        --skip-video)
            EXTRA_TEST_ARGS="$EXTRA_TEST_ARGS --skip-video"
            shift ;;
        --skip-pro)
            EXTRA_TEST_ARGS="$EXTRA_TEST_ARGS --skip-pro"
            shift ;;
        --shell)
            SHELL_MODE=true
            shift ;;
        --vla-eval)
            VLA_EVAL_MODE=true
            VLA_EVAL_SUITE="$2"
            shift 2 ;;
        --vla-url)
            VLA_URL="$2"
            shift 2 ;;
        --vla-num-tasks)
            VLA_NUM_TASKS="$2"
            shift 2 ;;
        --vla-num-trials)
            VLA_NUM_TRIALS="$2"
            shift 2 ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1 ;;
    esac
done

# ── Pre-flight checks ───────────────────────────────────

echo "=== Pre-flight checks ==="

# Docker
if ! command -v docker &>/dev/null; then
    echo "ERROR: docker not found. Install Docker first."
    exit 1
fi
echo "  Docker: $(docker --version | head -1)"

# nvidia runtime
if ! docker info 2>/dev/null | grep -q nvidia; then
    echo "WARNING: nvidia runtime not detected in docker info."
    echo "         --gpus all may fail. Install nvidia-container-toolkit."
fi

# GPU on host
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    echo "  GPU: $GPU_NAME"
else
    echo "WARNING: nvidia-smi not found on host."
fi

# Image
if ! docker image inspect "$IMAGE" &>/dev/null; then
    echo "  Image not found locally. Pulling $IMAGE ..."
    docker pull "$IMAGE"
else
    echo "  Image: $IMAGE (local)"
fi

# ── Claude Code 인증 마운트 (구독 방식) ──────────────────

CLAUDE_MOUNTS=""
CRED_PATH="${CLAUDE_CREDENTIALS:-$HOME/.claude/.credentials.json}"
# --api-key 플래그로 이미 설정된 경우 환경변수 주입 건너뜀
if [[ "$EXTRA_DOCKER_ARGS" == *"ANTHROPIC_API_KEY"* ]]; then
    echo "  Claude Code: API key from --api-key flag"
elif [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
    EXTRA_DOCKER_ARGS="$EXTRA_DOCKER_ARGS -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY"
    echo "  Claude Code: API key from env"
elif [[ -f "$CRED_PATH" ]]; then
    CLAUDE_MOUNTS="-v $CRED_PATH:/tmp/.credentials.json:ro"
    echo "  Claude Code: subscription (credentials mounted)"
else
    echo "  Claude Code: no credentials found (skipping)"
fi

# ── Output directory ─────────────────────────────────────

mkdir -p "$OUTPUT_DIR"
echo "  Output dir: $OUTPUT_DIR"

# OOD data (repo-tracked OOD bddl/init suites)
OOD_MOUNT=""
if [[ -d "$OOD_DATA_DIR" ]]; then
    OOD_MOUNT="-v $OOD_DATA_DIR:/tmp/ood_data:ro"
    echo "  OOD data: $OOD_DATA_DIR (mounted)"
else
    echo "  OOD data: $OOD_DATA_DIR NOT FOUND — OOD tests may fail"
fi

# LIBERO-PRO / LIBERO 소스 (v1.3 부터: 이미지에 포함되지 않음 - 반드시 호스트 clone 마운트 필요)
SRC_MOUNT=""
if [[ -d "$LIBERO_PRO_SRC" ]]; then
    SRC_MOUNT="$SRC_MOUNT -v $LIBERO_PRO_SRC:/workspace/LIBERO-PRO"
    echo "  LIBERO-PRO source: $LIBERO_PRO_SRC (mounted rw)"
else
    echo "ERROR: $LIBERO_PRO_SRC not found."
    echo "       v1.3 이미지는 소스를 포함하지 않습니다. 먼저 clone 해주세요:"
    echo "         git clone https://github.com/Zxy-MLlab/LIBERO-PRO.git $LIBERO_PRO_SRC"
    exit 1
fi
if [[ -d "$LIBERO_SRC" ]]; then
    SRC_MOUNT="$SRC_MOUNT -v $LIBERO_SRC:/workspace/LIBERO:ro"
    echo "  LIBERO source: $LIBERO_SRC (mounted ro)"
else
    echo "WARNING: $LIBERO_SRC not found. 원본 LIBERO 는 optional 이지만 일관성을 위해 clone 권장:"
    echo "           git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git $LIBERO_SRC"
fi

# ── Container entrypoint script ──────────────────────────
# Claude Code 인증 설정 + test_local.py 실행

ENTRYPOINT_CMD='
# Claude Code 인증 복사 (구독 방식)
if [ -f /tmp/.credentials.json ]; then
    mkdir -p /root/.claude
    cp /tmp/.credentials.json /root/.claude/.credentials.json
    cat > /root/.claude.json <<CJSON
{
  "hasCompletedOnboarding": true,
  "theme": "dark",
  "shiftEnterKeyBindingInstalled": true,
  "projects": {
    "/workspace/LIBERO-PRO": {
      "hasTrustDialogAccepted": true,
      "allowedTools": []
    }
  }
}
CJSON
fi

# OOD 데이터 병합 (호스트 ood_data/ -> 컨테이너 LIBERO-PRO)
# ood_data/ 는 parent repo 에서 버전 관리되는 OOD suite bddl/init 파일
if [ -d /tmp/ood_data ]; then
    if [ -d /tmp/ood_data/bddl_files ] && [ -n "$(ls -A /tmp/ood_data/bddl_files 2>/dev/null)" ]; then
        cp -rn /tmp/ood_data/bddl_files/* /workspace/LIBERO-PRO/libero/libero/bddl_files/
    else
        echo "WARNING: /tmp/ood_data/bddl_files missing or empty" >&2
    fi
    if [ -d /tmp/ood_data/init_files ] && [ -n "$(ls -A /tmp/ood_data/init_files 2>/dev/null)" ]; then
        cp -rn /tmp/ood_data/init_files/* /workspace/LIBERO-PRO/libero/libero/init_files/
    else
        echo "WARNING: /tmp/ood_data/init_files missing or empty" >&2
    fi
else
    echo "WARNING: /tmp/ood_data not mounted — OOD tests will fail" >&2
fi
'

# ── Run ──────────────────────────────────────────────────

echo ""
echo "=== Starting container ==="

# ── VLA eval mode ────────────────────────────────────────
if $VLA_EVAL_MODE; then
    echo "  Mode: VLA eval (suite=$VLA_EVAL_SUITE, url=$VLA_URL)"
    echo ""

    # Bring the benchmark container onto host network so it can hit
    # localhost:8400 where the openpi-pi05-http server is listening.
    docker run --rm --gpus all --shm-size=8g --network host \
        -e MUJOCO_GL=egl \
        -e VLA_SERVER_URL="$VLA_URL" \
        $SRC_MOUNT \
        $OOD_MOUNT \
        $EXTRA_DOCKER_ARGS \
        -v "$(pwd)/scripts:/workspace/LIBERO-PRO/scripts:ro" \
        -v "$OUTPUT_DIR:/workspace/LIBERO-PRO/test_outputs" \
        "$IMAGE" \
        bash -c "$ENTRYPOINT_CMD
pip install --quiet requests imageio imageio-ffmpeg >/dev/null 2>&1 || true
python scripts/libero_vla_eval.py \
    --vla-url \"$VLA_URL\" \
    --suite \"$VLA_EVAL_SUITE\" \
    --num-tasks $VLA_NUM_TASKS \
    --num-trials $VLA_NUM_TRIALS \
    --output-dir /workspace/LIBERO-PRO/test_outputs/eval"
    rc=$?
    if [[ $rc -eq 0 ]]; then
        echo ""
        echo "=== VLA EVAL COMPLETED (suite=$VLA_EVAL_SUITE) ==="
        echo "Results: $OUTPUT_DIR/eval/"
    else
        echo ""
        echo "=== VLA EVAL FAILED (exit $rc) ==="
        exit $rc
    fi
    exit 0
fi

if $SHELL_MODE; then
    echo "  Mode: interactive shell"
    docker run -it --rm --gpus all --shm-size=8g \
        -e MUJOCO_GL=egl \
        $CLAUDE_MOUNTS \
        $SRC_MOUNT \
        $OOD_MOUNT \
        $EXTRA_DOCKER_ARGS \
        -v "$(pwd)/test_local.py:/workspace/LIBERO-PRO/test_local.py:ro" \
        -v "$OUTPUT_DIR:/workspace/LIBERO-PRO/test_outputs" \
        "$IMAGE" \
        bash -c "$ENTRYPOINT_CMD
exec bash"
    exit 0
fi

echo "  Mode: test (EGL)"
echo ""

# EGL 시도
if docker run --rm --gpus all --shm-size=8g \
    -e MUJOCO_GL=egl \
    $CLAUDE_MOUNTS \
    $SRC_MOUNT \
    $OOD_MOUNT \
    $EXTRA_DOCKER_ARGS \
    -v "$(pwd)/test_local.py:/workspace/LIBERO-PRO/test_local.py:ro" \
    -v "$OUTPUT_DIR:/workspace/LIBERO-PRO/test_outputs" \
    "$IMAGE" \
    bash -c "$ENTRYPOINT_CMD
python test_local.py --output-dir /workspace/LIBERO-PRO/test_outputs $EXTRA_TEST_ARGS"; then
    echo ""
    echo "=== ALL TESTS PASSED (EGL) ==="
else
    EGL_EXIT=$?
    echo ""
    echo "=== EGL failed (exit $EGL_EXIT), retrying with OSMesa... ==="
    echo ""

    if docker run --rm --gpus all --shm-size=8g \
        -e MUJOCO_GL=osmesa \
        $CLAUDE_MOUNTS \
        $SRC_MOUNT \
        $OOD_MOUNT \
        $EXTRA_DOCKER_ARGS \
        -v "$(pwd)/test_local.py:/workspace/LIBERO-PRO/test_local.py:ro" \
        -v "$OUTPUT_DIR:/workspace/LIBERO-PRO/test_outputs" \
        "$IMAGE" \
        bash -c "$ENTRYPOINT_CMD
python test_local.py --output-dir /workspace/LIBERO-PRO/test_outputs $EXTRA_TEST_ARGS"; then
        echo ""
        echo "=== ALL TESTS PASSED (OSMesa fallback) ==="
    else
        echo ""
        echo "=== TESTS FAILED ==="
        echo "Check outputs in: $OUTPUT_DIR"
        exit 1
    fi
fi

echo "Outputs saved to: $OUTPUT_DIR"
