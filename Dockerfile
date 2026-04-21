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

# ── v1.3: LIBERO/LIBERO-PRO 소스는 이미지에 굽지 않는다 ──
# 런타임에 호스트 clone을 /workspace/LIBERO-PRO 와 /workspace/LIBERO 로 bind mount 한다.
# 빌드 단계에서는 pip 의존성만 설치하기 위해 upstream repo 에서 requirements.txt 를 임시로 가져와 설치한다.
# (LIBERO-PRO == LIBERO 포크, requirements.txt 동일)
RUN git clone --depth 1 https://github.com/Zxy-MLlab/LIBERO-PRO.git /tmp/libero-pro \
    && pip install --no-cache-dir -r /tmp/libero-pro/requirements.txt \
    && rm -rf /tmp/libero-pro

# ~/.libero/config.yaml 사전 생성 (없으면 import 시 interactive prompt 발생)
# 경로는 런타임 마운트 경로(/workspace/LIBERO-PRO)와 일치해야 한다.
RUN mkdir -p /root/.libero && printf '%s\n' \
    'benchmark_root: /workspace/LIBERO-PRO/libero/libero' \
    'bddl_files: /workspace/LIBERO-PRO/libero/libero/bddl_files' \
    'init_states: /workspace/LIBERO-PRO/libero/libero/init_files' \
    'datasets: /workspace/LIBERO-PRO/libero/datasets' \
    'assets: /workspace/LIBERO-PRO/libero/libero/assets' \
    > /root/.libero/config.yaml

# robosuite macros_private.py 초기화 (없으면 매번 경고 출력됨)
RUN python /usr/local/lib/python3.8/dist-packages/robosuite/scripts/setup_macros.py

# 최소 설치 검증 (libero 는 런타임 마운트 이후에만 import 가능하므로 여기선 torch/robosuite 만 확인)
RUN python -c "\
import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda); \
import robosuite; print('robosuite:', robosuite.__version__); \
print('Image build OK. LIBERO-PRO source must be mounted at runtime to /workspace/LIBERO-PRO') \
"

# LIBERO-PRO 소스는 런타임에 여기로 bind mount 된다. 빌드 시점에는 비어있다.
WORKDIR /workspace/LIBERO-PRO
ENV PYTHONPATH=/workspace/LIBERO-PRO:/workspace/LIBERO:${PYTHONPATH}

CMD ["/bin/bash"]
