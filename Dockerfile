FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
ENV PYTHONUNBUFFERED=1
ENV LANG=C.UTF-8

# System packages: Python 3.8, EGL headless rendering, build tools, Node.js deps
RUN apt-get update && apt-get install -y \
    python3.8 python3.8-dev python3-pip \
    libegl1-mesa-dev libgl1-mesa-dev libgles2-mesa-dev \
    libosmesa6-dev libglfw3-dev \
    git cmake build-essential patchelf wget ffmpeg \
    libsm6 libxext6 libxrender-dev \
    curl ca-certificates gnupg \
    && ln -sf /usr/bin/python3.8 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && rm -rf /var/lib/apt/lists/*

# Node.js 20 LTS + Claude Code CLI
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g @anthropic-ai/claude-code

# typing-extensions 먼저 핀 (4.10+ 는 Python 3.9+ 요구, 3.8과 충돌)
RUN pip install --no-cache-dir "typing-extensions==4.7.1"

# PyTorch 1.11 + CUDA 11.3
RUN pip install --no-cache-dir \
    torch==1.11.0+cu113 \
    torchvision==0.12.0+cu113 \
    torchaudio==0.11.0 \
    --extra-index-url https://download.pytorch.org/whl/cu113

WORKDIR /workspace

# LIBERO-PRO 설치 (원본 LIBERO 별도 설치 불필요 - LIBERO-PRO가 완전 상위호환)
COPY LIBERO-PRO /workspace/LIBERO-PRO
WORKDIR /workspace/LIBERO-PRO
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -e .

# HuggingFace LIBERO-PRO OOD bddl/init 데이터 다운로드
RUN pip install --no-cache-dir huggingface_hub && \
    python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('zhouxueyang/LIBERO-Pro', repo_type='dataset', local_dir='/tmp/libero-pro-data', \
    allow_patterns=['bddl_files/**', 'init_files/**'])" && \
    cp -rn /tmp/libero-pro-data/bddl_files/* /workspace/LIBERO-PRO/libero/libero/bddl_files/ && \
    cp -rn /tmp/libero-pro-data/init_files/* /workspace/LIBERO-PRO/libero/libero/init_files/ && \
    rm -rf /tmp/libero-pro-data /root/.cache/huggingface

# ~/.libero/config.yaml 사전 생성 (없으면 import 시 interactive prompt 발생)
RUN mkdir -p /root/.libero && cat > /root/.libero/config.yaml << 'EOF'
benchmark_root: /workspace/LIBERO-PRO/libero/libero
bddl_files: /workspace/LIBERO-PRO/libero/libero/bddl_files
init_states: /workspace/LIBERO-PRO/libero/libero/init_files
datasets: /workspace/LIBERO-PRO/libero/datasets
assets: /workspace/LIBERO-PRO/libero/libero/assets
EOF

# robosuite macros_private.py 초기화 (없으면 매번 경고 출력됨)
RUN python /usr/local/lib/python3.8/dist-packages/robosuite/scripts/setup_macros.py

# 설치 검증
RUN python -c "\
import torch; print('PyTorch:', torch.__version__); \
import robosuite; print('robosuite:', robosuite.__version__); \
from libero.libero import get_libero_path; \
print('bddl_files:', get_libero_path('bddl_files')); \
print('All OK') \
"

WORKDIR /workspace/LIBERO-PRO
CMD ["/bin/bash"]
