#!/usr/bin/env bash
set -euo pipefail

# ── LIBERO / LIBERO-PRO 로컬 Docker 검증 스크립트 ──
# 사용법:
#   ./run.sh                          # 기본 (EGL, 자동 osmesa fallback)
#   ./run.sh --api-key sk-ant-xxxx    # Claude Code API 키 방식
#   ./run.sh --skip-video             # 비디오 테스트 스킵
#   ./run.sh --shell                  # 테스트 대신 bash 셸 진입

IMAGE="bigenlight/libero-pro:latest"
OUTPUT_DIR="$(pwd)/test_outputs"
EXTRA_DOCKER_ARGS=""
EXTRA_TEST_ARGS=""
SHELL_MODE=false

# ── 인자 파싱 ────────────────────────────────────────────

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
        *)
            echo "Unknown option: $1"
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
'

# ── Run ──────────────────────────────────────────────────

echo ""
echo "=== Starting container ==="

if $SHELL_MODE; then
    echo "  Mode: interactive shell"
    docker run -it --rm --gpus all --shm-size=8g \
        -e MUJOCO_GL=egl \
        $CLAUDE_MOUNTS \
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
