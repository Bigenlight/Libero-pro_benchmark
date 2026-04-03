# LIBERO-PRO 실험 상세 계획

> 작성일: 2026-04-03  
> 환경: 현재 머신(RTX 3060 12GB)에서 Docker 이미지 빌드 → 실험 서버(~30GB VRAM, A100 추정)에서 pull 후 실험

---

## 사전에 고쳐야 할 버그 3개 (시작 전 필수)

### Bug 1: `ood_object.yaml` line 39 — 빈 키

```bash
sed -i '39s/^    :/    wooden_cabinet:/' LIBERO-PRO/libero_ood/ood_object.yaml
```

### Bug 2: `perturbation.py` line 658 — seed 기본값이 `int` 타입 자체

```python
# 수정 전
seed=configs.get("seed", int),
# 수정 후
seed=configs.get("seed", 42),
```

### Bug 3: `generate_init_states.py` — zipfile+pickle로 저장하는데 `evaluate.py`는 `torch.load()`로 읽음

```python
# 수정 전 (zipfile 방식)
with zipfile.ZipFile(output_filepath, 'w', ...) as zipf:
    zipf.writestr("archive/data.pkl", pickle.dumps(all_initial_states))
# 수정 후
torch.save(np.array(all_initial_states), output_filepath)
```

> **참고**: HuggingFace에서 사전 생성된 init 파일을 다운로드하면 이 버그는 우회 가능.

---

## Phase 0: Docker 이미지 빌드 (현재 머신, RTX 3060)

### Dockerfile

`/home/theo_lab/Libero-pro/Dockerfile` 로 저장:

```dockerfile
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# 시스템 패키지 (EGL headless 렌더링 포함)
RUN apt-get update && apt-get install -y \
    python3.8 python3.8-dev python3-pip \
    libegl1-mesa-dev libgl1-mesa-dev libgles2-mesa-dev \
    libosmesa6-dev libglfw3-dev \
    git cmake build-essential patchelf wget ffmpeg \
    libsm6 libxext6 libxrender-dev \
    curl ca-certificates \
    && ln -sf /usr/bin/python3.8 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && rm -rf /var/lib/apt/lists/*

# Node.js 20 LTS + Claude Code CLI
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g @anthropic-ai/claude-code

WORKDIR /workspace

# PyTorch 1.11 + CUDA 11.3
RUN pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 \
    --extra-index-url https://download.pytorch.org/whl/cu113

# LIBERO-PRO 설치 (원본 LIBERO 별도 설치 불필요 — LIBERO-PRO가 완전 상위호환)
COPY LIBERO-PRO /workspace/LIBERO-PRO
WORKDIR /workspace/LIBERO-PRO
RUN pip install -r requirements.txt && pip install -e .

# ~/.libero/config.yaml 사전 생성 (없으면 import 시 interactive prompt 실행됨)
RUN mkdir -p /root/.libero && cat > /root/.libero/config.yaml << 'EOF'
benchmark_root: /workspace/LIBERO-PRO/libero/libero
bddl_files: /workspace/LIBERO-PRO/libero/libero/bddl_files
init_states: /workspace/LIBERO-PRO/libero/libero/init_files
datasets: /workspace/LIBERO-PRO/libero/datasets
assets: /workspace/LIBERO-PRO/libero/libero/assets
EOF

# Bug 수정
RUN sed -i '39s/^    :/    wooden_cabinet:/' /workspace/LIBERO-PRO/libero_ood/ood_object.yaml
RUN sed -i 's/seed=configs.get("seed", int)/seed=configs.get("seed", 42)/' \
    /workspace/LIBERO-PRO/perturbation.py

# robosuite macros 초기화
RUN python -c "import robosuite" || true

# 설치 검증
RUN python -c "
from libero.libero import get_libero_path
print('bddl_files:', get_libero_path('bddl_files'))
import torch; print('PyTorch:', torch.__version__)
import robosuite; print('robosuite:', robosuite.__version__)
print('All OK')
"

WORKDIR /workspace/LIBERO-PRO
CMD ["/bin/bash"]
```

### `.dockerignore`

```
.git
**/.git
**/__pycache__
**/*.pyc
```

### 빌드 & 전송

```bash
# 빌드 (현재 머신)
cd /home/theo_lab/Libero-pro
docker build -t libero-pro-base:v1.0 .
# 예상 빌드 시간: 15-25분, 이미지 크기: ~12-15GB

# 이미지 전송 (scp 방식 — Docker Hub보다 빠름)
docker save libero-pro-base:v1.0 | gzip > /tmp/libero-pro-v1.0.tar.gz
scp /tmp/libero-pro-v1.0.tar.gz user@experiment-server:/tmp/

# 실험 서버에서 로드
docker load < /tmp/libero-pro-v1.0.tar.gz
```

> **대안 (서버에 Docker 없을 때)**: Singularity/Apptainer로 변환
> ```bash
> singularity build libero-pro-base.sif docker-archive://libero-pro-base.tar
> singularity exec --nv libero-pro-base.sif python run_eval.py
> ```

---

## Phase 1: 데이터 준비 (실험 서버)

### 컨테이너 실행

```bash
# === 구독 계정 연결 방식 (권장) ===
# Step 1: 현재 머신에서 한 번만 로그인 (브라우저 OAuth)
#   claude  →  Claude.ai 계정으로 로그인
#   자격증명이 ~/.claude/ 에 저장됨

# Step 2: docker run 시 ~/.claude/ 를 volume mount
docker run -it --gpus all --shm-size=16g \
  -v ~/.claude:/root/.claude \          # 구독 자격증명 주입 (키 불필요)
  -v /data/libero/datasets:/workspace/LIBERO-PRO/libero/datasets \
  -v /data/libero/results:/data/results \
  -e MUJOCO_GL=egl \
  -e CUDA_VISIBLE_DEVICES=0 \
  libero-pro-base:v1.0 bash

# 컨테이너 내부에서 바로 사용 가능
# claude
```

> **API Key 방식 (별도 결제 원할 때)**
> ```bash
> docker run ... -e ANTHROPIC_API_KEY=sk-ant-xxxx ... libero-pro-base:v1.0 bash
> ```

### 데이터 다운로드 (컨테이너 내부)

```bash
pip install huggingface_hub

# 1. LIBERO-PRO bddl/init 파일
python -c "
from huggingface_hub import snapshot_download
snapshot_download('zhouxueyang/LIBERO-Pro', repo_type='dataset', local_dir='/tmp/libero-pro-data')
"
cp -r /tmp/libero-pro-data/bddl_files/* libero/libero/bddl_files/
cp -r /tmp/libero-pro-data/init_files/*  libero/libero/init_files/

# 2. 학습 데모 데이터 (bc_transformer baseline용)
python benchmark_scripts/download_libero_datasets.py --datasets all --use-huggingface

# 3. 환경 검증
python benchmark_scripts/check_task_suites.py
```

> **주의**: Environment perturbation BDDL 파일은 Issue #9로 아직 미출시.  
> `use_environment: true` 실험은 HuggingFace 파일 확인 후 진행.

---

## Phase 2: Baseline — `bc_transformer_policy`

### 학습 (4 suite × 3 seed)

```bash
for SUITE in LIBERO_SPATIAL LIBERO_OBJECT LIBERO_GOAL LIBERO_10; do
  for SEED in 10000 20000 30000; do
    python libero/lifelong/main.py \
      benchmark_name=$SUITE \
      policy=bc_transformer_policy \
      lifelong=multitask \
      seed=$SEED \
      device=cuda:0 \
      use_wandb=false
  done
done
```

### 평가

```bash
for SUITE in libero_spatial libero_object libero_goal libero_10; do
  for TASK_ID in $(seq 0 9); do
    python libero/lifelong/evaluate.py \
      --benchmark $SUITE \
      --algo multitask \
      --policy bc_transformer_policy \
      --seed 10000 \
      --ep 45 \
      --task_id $TASK_ID \
      --device_id 0
  done
done
```

> **참고**: 기존 `evaluate.py`는 perturbation을 인식하지 않음.  
> perturbation 평가 시 `~/.libero/config.yaml`의 `bddl_files`/`init_states` 경로를  
> perturbation된 디렉토리로 변경 후 실행하거나, VLA 통합 스크립트를 사용.

---

## Phase 3: VLA 모델 평가

### 모델 선택 전략 (30GB VRAM 기준)

| 우선순위 | 모델 | 추론 VRAM | 이유 |
|---------|------|-----------|------|
| 1 | **OpenVLA** | ~15GB | LIBERO fine-tuned weight 4개 바로 사용 가능 |
| 2 | **OpenVLA-OFT** | ~16GB | LIBERO SOTA 97.1%, 공식 평가 스크립트 완비 |
| 3 | **SpatialVLA** | ~8GB | 가장 낮은 VRAM, 여유 있음 |
| 4 | **SmolVLA** | ~2GB | 450M 경량 (단, LIBERO 재현 이슈 주의) |
| 5 | **pi0** | ~8GB | 추론만 가능, LIBERO-PRO 논문 최강 성능 |

### OpenVLA / OpenVLA-OFT 통합 방법

VLA 모델은 Python 3.10 + PyTorch 2.x가 필요하므로 **별도 conda 환경** 사용:

```bash
conda create -n openvla python=3.10 -y && conda activate openvla
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118

git clone https://github.com/moojink/openvla-oft.git && cd openvla-oft
pip install -e .

# LIBERO-PRO 연결
ln -s /workspace/LIBERO-PRO experiments/robot/libero/LIBERO-PRO
```

`run_libero_eval.py` 수정 필요 사항 (README 202-371행 가이드 참고):

1. `TaskSuite` Enum에 perturbation suite 추가 (`libero_goal_obj`, `libero_goal_swap` 등)
2. `eval_libero()`에서 `evaluation_config.yaml` 읽어 `perturbation.create_env()` 호출
3. **Issue #14 workaround**: `use_task=True` 시 language instruction을 파일명이 아닌 BDDL 파일에서 직접 파싱

```python
# Issue #14 workaround: BDDL에서 language instruction 직접 추출
import re
with open(bddl_file_path, 'r') as f:
    bddl_content = f.read()
match = re.search(r'\(:language\s*(.*?)\)', bddl_content, re.S)
if match:
    task_description = match.group(1).strip()
```

```bash
# 평가 실행
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --num_trials_per_task 20
```

---

## Phase 4: Perturbation 실험 설계

### 단일 Perturbation (`evaluation_config.yaml` 수정)

5가지 perturbation을 개별 적용. 해당 flag만 `true`로 설정:

```yaml
# 예: object perturbation만 활성화
bddl_files_path: "./LIBERO-PRO/libero/libero/bddl_files/"
script_path: "./LIBERO-PRO/notebooks/generate_init_states.py"
init_file_dir: "./LIBERO-PRO/libero/libero/init_files/"

use_environment: false
use_swap: false
use_object: true      # <-- 이것만 true
use_language: false
use_task: false

ood_task_configs:
  environment: "./libero_ood/ood_environment.yaml"
  swap: "./libero_ood/ood_spatial_relation.yaml"
  object: "./libero_ood/ood_object.yaml"
  language: "./libero_ood/ood_language.yaml"
  task: "./libero_ood/ood_task.yaml"
```

| Perturbation | 측정 목표 |
|---|---|
| `use_swap` | Position generalization: 객체 초기 위치 교환에 대한 강인성 |
| `use_object` | Object generalization: 외형이 다른 객체 인식 능력 |
| `use_language` | Semantic generalization: 언어 paraphrase 강인성 |
| `use_task` | Task generalization: 완전히 다른 목표 전이 능력 |
| `use_environment` | Environment generalization: 환경 변경 강인성 (파일 미출시 확인 필요) |

> **규칙**: `use_task=True`는 다른 모든 perturbation과 상호 배타적. 함께 사용 시 ValueError 발생.  
> **실행 순서**: swap → environment → object → language (코드 내 고정 순서)

### 조합 Perturbation

```yaml
# 권장 조합 (use_task는 단독만)
use_swap: true
use_object: true      # swap + object: 위치 + 외형 동시 변화

use_swap: true
use_language: true    # swap + language: 위치 + 언어 동시 변화

use_object: true
use_language: true    # object + language: 외형 + 언어 동시 변화

use_swap: true
use_object: true
use_language: true    # 3중 (최대 난이도)
```

### Position Perturbation (강도별)

```bash
# 사전 생성된 파일 사용 (x0.1 ~ x0.5, y0.1 ~ y0.5, 총 10단계)
# perturbation.py의 5가지 perturbation과는 별개 파이프라인

for INTENSITY in x0.1 x0.2 x0.3 x0.4 x0.5; do
  cp -r libero/libero/bddl_files/libero_object_temp_${INTENSITY}/* \
        libero/libero/bddl_files/libero_object_temp/
  cp -r libero/libero/init_files/libero_object_temp_${INTENSITY}/* \
        libero/libero/init_files/libero_object_temp/

  # 평가 실행 (run_libero_eval.py 또는 evaluate.py)
  python run_eval.py --task_suite libero_object_temp ... \
    2>&1 | tee logs/position_${INTENSITY}.log
done
```

---

## 평가 메트릭

### Success Rate

각 task에 대해 `n_eval=20` episode rollout → 성공 비율:

```
success_rate = num_success / 20
```

### Confusion Matrix

```python
import torch
result = torch.load("experiments/.../result.pt")
S = result["S_conf_mat"]  # (10, 10) success confusion matrix
L = result["L_conf_mat"]  # (10, 10) loss confusion matrix
```

### Robustness Drop (추가 제안)

```
Robustness_Drop = SR_original - SR_perturbed
```

값이 클수록 해당 perturbation에 취약.

### LIBERO-PRO Leaderboard 포맷

| Model | Goal-Obj | Goal-Pos | Goal-Sem | Goal-Task | Spatial-Obj | ... | Total |
|-------|----------|----------|----------|-----------|-------------|-----|-------|

---

## 알려진 이슈 및 Workaround 정리

| Issue | 현상 | Workaround |
|-------|------|-----------|
| **#8** | `ood_object.yaml` 39행 빈 키 | `sed`로 `wooden_cabinet:` 으로 수정 |
| **#9** | Environment BDDL 파일 미출시 | `use_environment` 실험은 파일 확인 후 진행 |
| **#14** | `use_task=True` 시 language 원래 값으로 복귀 | BDDL 파일에서 `(:language ...)` 직접 파싱 |
| **#15** | pi0.5 결과 재현 불가 | 논문 수치를 참조치로만 활용 |
| **#20** | OpenVLA 결과 불일치 | `unnorm_key`, `center_crop`, seed 통일 확인 |
| **robosuite** | 1.5+ 설치 시 `SingleArmEnv` crash | `requirements.txt`에서 `robosuite==1.4.0` 고정 |
| **EGL** | 헤드리스 서버 렌더링 실패 | `MUJOCO_GL=egl` + EGL 패키지 설치, fallback: `osmesa` |
| **`perturbation.py`** | `seed=int` 타입 버그 | `seed=42`로 수정 |
| **`generate_init_states.py`** | zipfile 포맷과 `torch.load()` 불일치 | `torch.save()` 방식으로 수정 또는 HuggingFace init 파일 사용 |

---

## 예상 일정 (8주)

| 주차 | 작업 |
|------|------|
| **Week 1** | Docker 빌드 + 이미지 전송 + 데이터 다운로드 + 환경 검증 |
| **Week 2** | `bc_transformer_policy` baseline 학습 (4 suite × 3 seed) |
| **Week 3** | bc_transformer perturbation 평가 + OpenVLA 환경 구축 |
| **Week 4** | OpenVLA + OpenVLA-OFT 단일 perturbation 평가 |
| **Week 5** | SpatialVLA + SmolVLA 평가 |
| **Week 6** | pi0 평가 + 조합 perturbation 실험 |
| **Week 7** | Position perturbation intensity (x0.1~x0.5) 실험 |
| **Week 8** | 결과 분석 + confusion matrix 시각화 + 정리 |

---

## 주요 주의사항 요약

1. **robosuite 1.4.x 고정** — 1.5+ 설치되면 `SingleArmEnv` import 크래시
2. **LIBERO-PRO만 `pip install -e .`** — 원본 LIBERO 별도 설치 불필요 (상위호환)
3. **`MUJOCO_GL=egl` 필수** — 헤드리스 서버에서 필수
4. **`--shm-size=16g`** — MuJoCo 병렬 환경이 shared memory 많이 사용
5. **VLA 모델은 별도 Python 3.10 + PyTorch 2.x 환경** — LIBERO-PRO와 버전 충돌
6. **`evaluation_config.yaml`의 경로 설정** — `~/.libero/config.yaml`과 일치해야 함
7. **Environment perturbation** — `perturbation.py` 410행에 `living_room_table`로 하드코딩됨

---

## 참고 링크

- LIBERO-PRO GitHub: https://github.com/Zxy-MLlab/LIBERO-PRO
- LIBERO-PRO 논문: https://arxiv.org/abs/2510.03827
- LIBERO-PRO HuggingFace 데이터셋: https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro
- OpenVLA-OFT: https://github.com/moojink/openvla-oft
- OpenVLA HuggingFace: https://huggingface.co/openvla
- pi0 (openpi): https://github.com/Physical-Intelligence/openpi
- SpatialVLA: https://github.com/SpatialVLA/SpatialVLA
- SmolVLA: https://huggingface.co/lerobot/smolvla_base
