# LIBERO-PRO Benchmark — Docker Setup & Experiment Plan

> VLA(Vision-Language-Action) 모델의 진짜 일반화 능력을 평가하는 LIBERO-PRO 벤치마크 실험 환경.  
> Docker 이미지 기반으로 누구나 동일한 환경에서 재현 가능.

---

## 개요

[LIBERO-PRO](https://arxiv.org/abs/2510.03827)는 기존 LIBERO 벤치마크를 확장하여 VLA 모델이 **암기가 아닌 진짜 일반화**를 달성했는지 평가하는 벤치마크다.

5가지 OOD(Out-of-Distribution) perturbation 차원으로 모델을 평가한다:

| Perturbation | 설명 |
|---|---|
| **Object** | 외형이 다른 동일 기능 객체로 교체 |
| **Spatial (Swap)** | 객체 초기 위치 교환 |
| **Semantic (Language)** | 동일 의미의 다른 자연어 표현 |
| **Task** | 완전히 다른 목표 태스크로 교체 |
| **Environment** | 테이블/환경 교체 (데이터 준비 중) |

OpenVLA, Pi0 같은 SOTA VLA 모델도 원본 LIBERO에서 90%+ 달성하지만, LIBERO-PRO perturbation 적용 시 대부분 0%에 가깝게 붕괴된다.

---

## Prerequisites

실험 서버에 아래가 설치되어 있어야 한다:

```bash
# 1. Docker
curl -fsSL https://get.docker.com | sh

# 2. NVIDIA Container Toolkit (--gpus all 동작에 필수)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# 설치 확인
docker run --rm --gpus all nvidia/cuda:11.3.1-base-ubuntu20.04 nvidia-smi
```

---

## Docker 이미지

**Docker Hub**: [`bigenlight/libero-pro`](https://hub.docker.com/r/bigenlight/libero-pro)

| 태그 | 내용 | 크기 |
|------|------|------|
| `v1.3` = `latest` | Base 환경 (**LIBERO/LIBERO-PRO 소스, OOD 데이터 모두 런타임 마운트**) | ~15GB |
| `v1.2` | Base 환경 + LIBERO-PRO 소스 내장 (OOD 만 마운트) | ~23GB |
| `v1.1` | Base 환경 + OOD bddl/init 데이터 내장 (HF 다운로드) | ~25GB |
| `v1.0` | Base 환경 (OOD 데이터 없음) | ~23GB |
| `train` | Base + 학습 데이터셋 포함 (예정) | ~60GB+ |

> **v1.3 변경 (2026-04-21)**: LIBERO-PRO 및 LIBERO 소스를 이미지에서 제거하고 호스트
> bind mount 로 전환했다. 이미지는 CUDA + PyTorch + pip 의존성만 포함한다. 소스 수정이
> 재빌드 없이 즉시 반영되며, 이미지 크기는 ~23GB → ~15GB 로 감소.
>
> **사용 전 필수**: 아래 "빠른 시작" 의 `git clone` 두 줄을 **반드시** 먼저 실행해야 한다.
> 소스가 없으면 `run.sh` 가 에러로 종료된다.

### 이미지 구성

- Ubuntu 20.04 + CUDA 11.3.1 + cuDNN 8
- Python 3.8.10 / PyTorch 1.11.0+cu113
- robosuite 1.4.0 / robomimic 0.2.0 / LIBERO-PRO
- Node.js 20 LTS + **Claude Code** (빌드 시점 최신 버전) 내장
- 버그 픽스 3개 적용 완료 (아래 참고)

---

## 빠른 시작

### 1. 이미지 Pull

```bash
docker pull bigenlight/libero-pro:latest
```

### 2. LIBERO / LIBERO-PRO 소스 Clone (v1.3 부터 필수)

v1.3 이미지는 소스를 포함하지 않는다. 이 리포의 루트에서 두 upstream repo 를 clone 한다:

```bash
cd Libero-pro_benchmark
git clone https://github.com/Zxy-MLlab/LIBERO-PRO.git LIBERO-PRO
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git LIBERO
```

두 디렉토리는 `.gitignore` 에 등록되어 있어 parent repo 에 포함되지 않는다. `run.sh` 가
자동으로 컨테이너에 bind mount 한다:

| 호스트 | 컨테이너 | 모드 | 역할 |
|--------|----------|------|------|
| `LIBERO-PRO/` | `/workspace/LIBERO-PRO` | rw | LIBERO-PRO 확장 코드 (perturbation, libero_ood, libero) |
| `LIBERO/` | `/workspace/LIBERO` | ro | 원본 LIBERO (비교/참고용, optional) |
| `ood_data/` | `/tmp/ood_data` | ro | OOD bddl/init 파일 (런타임에 LIBERO-PRO 로 병합) |

이미지에는 `PYTHONPATH=/workspace/LIBERO-PRO:/workspace/LIBERO` 가 설정되어 있어서,
마운트만 되면 `from libero.libero import benchmark` 와 `from perturbation import ...` 가 바로 작동한다 (editable install 불필요).

> **왜 마운트인가?** 컨테이너는 일회용 실행환경으로 취급하고 코드/데이터/결과는 호스트에 둔다.
> 소스 수정이 재빌드 없이 즉시 반영되고, 실험 결과도 `test_outputs/` 에 영속된다.

### 3. 컨테이너 실행

**권장: `run.sh` 사용** (Claude Code 인증 자동 주입)

```bash
./run.sh                          # 기본 실행 (구독 인증 자동)
./run.sh --api-key sk-ant-xxxx    # API 키 방식
./run.sh --skip-video             # 비디오 테스트 스킵
./run.sh --skip-pro               # LIBERO-PRO 테스트 스킵
./run.sh --shell                  # bash 셸 진입 (테스트 대신)
CUDA_VISIBLE_DEVICES=0,1 ./run.sh # GPU 선택
LIBERO_IMAGE=bigenlight/libero-pro:v1.0 ./run.sh  # 이미지 버전 지정
```

<details>
<summary>수동 실행 (run.sh 없이)</summary>

```bash
# 기본 실행 (eval/inference, Claude Code 없이)
docker run -it --gpus all --shm-size=8g \
  -e MUJOCO_GL=egl \
  bigenlight/libero-pro:latest bash
```

```bash
# Claude Code — API 키 방식
docker run -it --gpus all --shm-size=8g \
  -e ANTHROPIC_API_KEY=sk-ant-xxxx \
  -e MUJOCO_GL=egl \
  bigenlight/libero-pro:latest bash
```

</details>

### 4. 컨테이너 내부 확인

```bash
# 환경 확인
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# LIBERO-PRO 벤치마크 확인
python -c "
from libero.libero import benchmark
bd = benchmark.get_benchmark_dict()
print(list(bd.keys())[:6])
"

# Claude Code 실행 (run.sh로 진입한 경우 로그인 불필요)
claude
```

---

## Docker 내부에서 Claude Code 로그인 없이 사용하기

Docker 컨테이너 안에서 `claude`를 실행하면 매번 로그인을 요구한다. 이는 컨테이너가 매 실행마다 새 환경을 생성하기 때문이다. 아래 방법으로 **호스트의 인증 정보를 재사용**하여 로그인 없이 바로 사용할 수 있다.

### 왜 `~/.claude` 전체를 마운트하면 안 되나?

```bash
# ❌ 이렇게 하면 안 된다
docker run -v ~/.claude:/root/.claude ... bash
```

`~/.claude/` 디렉토리에는 자격증명 외에도 세션, 프로젝트 설정, 캐시 등 **호스트 경로에 종속된 파일**이 포함되어 있다. 이를 통째로 마운트하면:

- 호스트 경로(`/home/user/project`)와 컨테이너 경로(`/workspace/LIBERO-PRO`)가 불일치
- 세션/프로젝트 데이터 충돌로 온보딩 화면 또는 로그인 화면이 다시 뜸

### 해결 원리

필요한 것은 딱 **2가지**다:

| 파일 | 위치 (컨테이너 내부) | 역할 |
|------|----------------------|------|
| `.credentials.json` | `/root/.claude/.credentials.json` | OAuth 토큰 (인증) |
| `.claude.json` | `/root/.claude.json` | 온보딩 완료 + 프로젝트 trust 설정 |

`.credentials.json`만 있으면 `claude -p` (headless)는 동작하지만, 인터랙티브 모드(`claude`)에서는 **온보딩 화면**이 먼저 뜬다. `.claude.json`에 아래 설정이 있어야 온보딩과 trust 확인을 건너뛴다:

```json
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
```

> **주의**: `projects` 키의 경로는 **컨테이너 내부 작업 디렉토리**와 일치해야 한다.

### 단계별 방법

#### Step 1: 호스트에서 Claude Code 로그인 (1회)

```bash
# 호스트 터미널에서 실행
claude
# → 브라우저가 열리면 Claude.ai 계정으로 로그인
# → ~/.claude/.credentials.json 에 토큰 저장됨
```

> `CLAUDE_CONFIG_DIR` 환경변수가 설정된 경우, 해당 경로에 저장된다.

#### Step 2: `run.sh`로 컨테이너 실행

```bash
./run.sh
```

`run.sh`가 자동으로 수행하는 작업:

1. 호스트의 `.credentials.json`을 `/tmp/.credentials.json`으로 읽기 전용 마운트
2. 컨테이너 시작 시 `/root/.claude/`로 복사 (쓰기 가능하게)
3. `/root/.claude.json`에 온보딩 스킵 + trust 설정 생성
4. bash 셸 진입

#### Step 3: 컨테이너 내부에서 확인

```bash
# 로그인 없이 바로 사용 가능
claude
```

### API 키 방식 (대안)

구독 계정 대신 API 키를 사용하려면:

```bash
./run.sh --api-key sk-ant-xxxx
# 또는
docker run -it --gpus all --shm-size=8g \
  -e ANTHROPIC_API_KEY=sk-ant-xxxx \
  -e MUJOCO_GL=egl \
  bigenlight/libero-pro:latest bash
```

API 키 방식은 `.credentials.json`이나 `.claude.json` 없이도 동작한다.

### Troubleshooting

**여전히 로그인 화면이 뜨는 경우**

```bash
# 자격증명 파일 존재 확인
ls -la ~/.claude/.credentials.json
# CLAUDE_CONFIG_DIR 설정 시 해당 경로 확인
echo $CLAUDE_CONFIG_DIR
ls -la $CLAUDE_CONFIG_DIR/.credentials.json
```

**OAuth 토큰 만료 시**

```bash
# 호스트에서 다시 로그인
claude
# → 토큰이 갱신되면 다음 run.sh 실행 시 자동 반영
```

**커스텀 자격증명 경로**

```bash
CLAUDE_CREDENTIALS=/path/to/.credentials.json ./run.sh
```

---

## 데이터셋 준비

### OOD bddl/init 데이터 (v1.2 부터: `ood_data/` 자동 마운트)

`v1.2` = `latest` 부터는 **OOD bddl/init 파일이 이 리포의 `ood_data/` 에 커밋**되어 있다.
`run.sh` 가 컨테이너 시작 시 자동으로 마운트 + 병합하므로 별도 절차가 필요 없다.

```
ood_data/
├── bddl_files/   # 16 OOD suites × 10 tasks (swap / object / lan / task)
└── init_files/   # torch.save 형식 init states (각 task 당 50 states)
```

`run.sh` 동작:
1. 호스트 `ood_data/` → 컨테이너 `/tmp/ood_data:ro` 마운트
2. 컨테이너 진입 직후 `cp -rn /tmp/ood_data/bddl_files/*` 를
   `/workspace/LIBERO-PRO/libero/libero/bddl_files/` 로 병합 (`-n` no-clobber)
3. `init_files` 동일하게 병합

> 직접 `docker run` 으로 이미지를 쓸 경우 OOD 파일이 없으니, 테스트를 돌리려면 반드시
> `run.sh` 를 사용하거나 같은 마운트 규칙을 직접 지정해야 한다.

### 학습 데모 데이터 (bc_transformer baseline 학습용)

```bash
# 컨테이너 내부에서 실행
cd /workspace/LIBERO-PRO
python benchmark_scripts/download_libero_datasets.py --datasets all --use-huggingface
```

> **참고**: `datasets` 경로 Warning은 정상. `-v` 마운트 또는 다운로드 후 사라짐.

---

## 실험 서버 스펙 (원격 서버 `aadd`)

| 항목 | 사양 |
|------|------|
| OS | Ubuntu 22.04.5 LTS |
| CPU | AMD EPYC 7313 16-Core (32 threads, 3.73GHz) |
| RAM | 252GB |
| Storage | 1.8TB (여유 447GB) |
| GPU | RTX A6000 × 4장 (48GB × 4 = **192GB VRAM**) |
| CUDA | 12.4 / Driver 550.144.03 |

> CUDA 12.4 호스트에서 CUDA 11.3 컨테이너 실행 가능 (하위 호환).

### 192GB VRAM으로 가능한 실험

| 모델 | 추론 VRAM | 비고 |
|------|-----------|------|
| OpenVLA (7B) | ~15GB | GPU 1장으로 가능 |
| OpenVLA-OFT (7B) | ~16GB | GPU 1장으로 가능 |
| SpatialVLA (4B) | ~8GB | GPU 1장으로 가능 |
| SmolVLA (450M) | ~2GB | GPU 1장으로 가능 |
| pi0 (3B) | ~8GB | GPU 1장으로 가능 |
| **pi0.5 Full FT** | ~70GB | **GPU 2장으로 가능** (추정치) |
| 멀티 모델 동시 평가 | — | 4장 독립 실행 가능 |

---

## 현재 상태

- [x] Docker 이미지 빌드 완료 (v1.0 → v1.1 → v1.2)
- [x] Docker Hub 푸시 완료
- [x] 버그 픽스 3개 적용
- [x] 컨테이너 검증 완료 (패키지, 벤치마크, Claude Code)
- [x] OOD bddl/init 데이터 이미지 내장 (`v1.1`)
- [x] OOD 데이터 repo 편입 (`ood_data/`) + 마운트 워크플로우 (`v1.2`, 2026-04-15)
- [x] 로컬 검증 10/10 PASS (RTX 3060 12GB, EGL, v1.1→v1.2, 2026-04-08/15)
- [x] 원격 서버(aadd, RTX A6000 × 4) eval 환경 검증 — 10/10 PASS (v1.1, 2026-04-15)
- [x] 원격 서버에서 v1.2 재검증 — 10/10 PASS (2026-04-15)
- [x] LIBERO / LIBERO-PRO 소스까지 완전 마운트화 (`v1.3`, 2026-04-21)
- [ ] `train` 태그 이미지 (데이터셋 포함) 빌드
- [ ] bc_transformer_policy baseline 학습
- [ ] VLA 모델 LIBERO-PRO 평가

---

## 비디오 녹화 (시각화)

LIBERO/LIBERO-PRO 환경을 영상으로 녹화할 수 있다.

```bash
# LIBERO-PRO OOD 환경 녹화 (4종)
docker run --rm --gpus all --shm-size=8g \
  -e MUJOCO_GL=egl \
  -v $(pwd)/record_pro_video.py:/workspace/LIBERO-PRO/record_pro_video.py:ro \
  -v $(pwd)/test_outputs:/workspace/LIBERO-PRO/test_outputs \
  bigenlight/libero-pro:latest python record_pro_video.py

# 원본 vs OOD 비교 영상 (3쌍)
docker run --rm --gpus all --shm-size=8g \
  -e MUJOCO_GL=egl \
  -v $(pwd)/record_comparison.py:/workspace/LIBERO-PRO/record_comparison.py:ro \
  -v $(pwd)/test_outputs:/workspace/LIBERO-PRO/test_outputs \
  bigenlight/libero-pro:latest python record_comparison.py
```

출력물은 `test_outputs/` 디렉토리에 저장된다.

---

## 실험 계획

### Phase 1: Baseline (`bc_transformer_policy`)

```bash
# 4개 suite × 3 seed 학습
python libero/lifelong/main.py \
  benchmark_name=LIBERO_SPATIAL \
  policy=bc_transformer_policy \
  lifelong=multitask \
  seed=10000 \
  device=cuda:0
```

### Phase 2: VLA 모델 LIBERO-PRO 평가

평가 순서: **OpenVLA → OpenVLA-OFT → SpatialVLA → SmolVLA → pi0 → pi0.5**

```bash
# evaluation_config.yaml에서 perturbation 설정 후 평가
# use_object: true  (또는 swap / language / task)
python libero/lifelong/evaluate.py \
  --benchmark libero_spatial \
  --algo multitask \
  --policy bc_transformer_policy \
  --seed 10000 --ep 45 --task_id 0 --device_id 0
```

### Phase 3: Perturbation 조합 실험

| 조합 | 목표 |
|------|------|
| swap + object | 위치 + 외형 동시 변화 강인성 |
| swap + language | 위치 + 언어 표현 변화 강인성 |
| swap + object + language | 3중 perturbation (최대 난이도) |

### Phase 4: Position Perturbation Intensity

```bash
# bddl_files 내 libero_object_temp_x0.1 ~ x0.5 사전 생성 파일 사용
for INTENSITY in x0.1 x0.2 x0.3 x0.4 x0.5; do
  cp -r libero/libero/bddl_files/libero_object_temp_${INTENSITY}/* \
        libero/libero/bddl_files/libero_object_temp/
  cp -r libero/libero/init_files/libero_object_temp_${INTENSITY}/* \
        libero/libero/init_files/libero_object_temp/
  # 평가 실행
done
```

자세한 계획: [`LIBERO_PRO_PLAN.md`](./LIBERO_PRO_PLAN.md)

---

## 알려진 이슈

| Issue | 현상 | 상태 |
|-------|------|------|
| [#8](https://github.com/Zxy-MLlab/LIBERO-PRO/issues/8) | `ood_object.yaml` 빈 키 | ✅ 이미지에서 수정됨 |
| [#9](https://github.com/Zxy-MLlab/LIBERO-PRO/issues/9) | Environment BDDL 미출시 | ⏳ 업스트림 대기 |
| [#14](https://github.com/Zxy-MLlab/LIBERO-PRO/issues/14) | `use_task` 언어 지시 버그 | ⚠️ VLA 통합 시 workaround 필요 |
| [#20](https://github.com/Zxy-MLlab/LIBERO-PRO/issues/20) | OpenVLA 결과 불일치 | ⚠️ 확인 필요 |

---

## Troubleshooting

**EGL 렌더링 실패**
```bash
# 증상: EGL initialization failed / 검은 화면
# 해결: osmesa fallback
docker run ... -e MUJOCO_GL=osmesa ...
```

> `run.sh` 사용 시 EGL 실패하면 자동으로 OSMesa로 fallback된다.

**`--gpus all` 미동작**
```bash
# nvidia-container-toolkit 설치 여부 확인
nvidia-container-cli info
# 미설치 시 Prerequisites 섹션 참고
```

**robosuite import 오류 (`SingleArmEnv` not found)**
```bash
# robosuite 1.5+ 설치된 경우
pip install robosuite==1.4.0
```

**datasets 경로 Warning**
```bash
# 정상 동작 — 데이터셋을 volume mount하거나 컨테이너 내부에 다운로드하면 해결
docker run ... -v /host/path/datasets:/workspace/LIBERO-PRO/libero/datasets ...
```

---

## 앞으로 방향

1. **eval 이미지 검증** — 원격 서버에서 벤치마크 로딩 및 환경 렌더링 확인
2. **train 이미지 빌드** — 학습 데이터셋(libero_spatial/object/goal/10) 포함 레이어 추가
3. **VLA 모델 통합** — OpenVLA-OFT 기준으로 LIBERO-PRO perturbation 평가 파이프라인 구축
4. **결과 공개** — 모델별 robustness drop 분석 및 leaderboard 제출

---

<details>
<summary>적용된 버그 픽스 (클릭해서 펼치기)</summary>

### Bug 1: `ood_object.yaml` line 39 — 빈 키
```yaml
# 수정 전
    :
      - yellow_cabinet
# 수정 후
    wooden_cabinet:
      - yellow_cabinet
```

### Bug 2: `perturbation.py` line 658 — seed 타입 버그
```python
# 수정 전
seed=configs.get("seed", int),
# 수정 후
seed=configs.get("seed", 42),
```

### Bug 3: `generate_init_states.py` — 저장 포맷 불일치
```python
# 수정 전: zipfile+pickle (evaluate.py의 torch.load와 불일치)
with zipfile.ZipFile(output_filepath, 'w', ...) as zipf: ...
# 수정 후
torch.save(np.array(all_initial_states), output_filepath)
```

</details>

---

## 참고 링크

| 리소스 | 링크 |
|--------|------|
| LIBERO-PRO 논문 | https://arxiv.org/abs/2510.03827 |
| LIBERO-PRO GitHub | https://github.com/Zxy-MLlab/LIBERO-PRO |
| HuggingFace 데이터셋 | https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro |
| OpenVLA-OFT | https://github.com/moojink/openvla-oft |
| pi0 (openpi) | https://github.com/Physical-Intelligence/openpi |
| Docker Hub | https://hub.docker.com/r/bigenlight/libero-pro |
