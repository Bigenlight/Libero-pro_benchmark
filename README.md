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

## Docker 이미지

**Docker Hub**: [`bigenlight/libero-pro`](https://hub.docker.com/r/bigenlight/libero-pro)

| 태그 | 내용 | 크기 |
|------|------|------|
| `v1.0`, `latest` | Base 환경 (eval/inference용) | ~23GB |
| `train` | Base + 학습 데이터셋 포함 (예정) | ~60GB+ |

### 이미지 구성

- Ubuntu 20.04 + CUDA 11.3.1 + cuDNN 8
- Python 3.8.10 / PyTorch 1.11.0+cu113
- robosuite 1.4.0 / robomimic 0.2.0 / LIBERO-PRO
- Node.js 20 LTS + **Claude Code v2.1.91** 내장
- 버그 픽스 3개 적용 완료 (아래 참고)

---

## 빠른 시작

### 1. 이미지 Pull

```bash
docker pull bigenlight/libero-pro:latest
```

### 2. 컨테이너 실행

```bash
# 기본 실행 (eval/inference)
docker run -it --gpus all --shm-size=16g \
  -e MUJOCO_GL=egl \
  bigenlight/libero-pro:latest bash
```

```bash
# Claude Code 사용 (구독 계정 연결)
docker run -it --gpus all --shm-size=16g \
  -v ~/.claude:/root/.claude \
  -e MUJOCO_GL=egl \
  bigenlight/libero-pro:latest bash
```

```bash
# 전체 옵션 (데이터셋 마운트 + 결과 저장 + Claude)
docker run -it --gpus all --shm-size=16g \
  -v ~/.claude:/root/.claude \
  -v /path/to/datasets:/workspace/LIBERO-PRO/libero/datasets \
  -v /path/to/results:/data/results \
  -e MUJOCO_GL=egl \
  -e CUDA_VISIBLE_DEVICES=0 \
  bigenlight/libero-pro:latest bash
```

> **Claude Code**: `ANTHROPIC_API_KEY` 환경변수 또는 `~/.claude` 마운트 중 택일.  
> API 키 방식: `-e ANTHROPIC_API_KEY=sk-ant-xxxx`

### 3. 컨테이너 내부 확인

```bash
# 환경 확인
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# LIBERO-PRO 벤치마크 확인
python -c "
from libero.libero import benchmark
bd = benchmark.get_benchmark_dict()
print(list(bd.keys())[:6])
"

# Claude Code 실행
claude
```

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

> CUDA 12.4 호스트에서 CUDA 11.3 컨테이너 실행 가능 (forward compatibility).

### 192GB VRAM으로 가능한 실험

| 모델 | 추론 VRAM | 비고 |
|------|-----------|------|
| OpenVLA (7B) | ~15GB | GPU 1장으로 가능 |
| OpenVLA-OFT (7B) | ~16GB | GPU 1장으로 가능 |
| SpatialVLA (4B) | ~8GB | GPU 1장으로 가능 |
| SmolVLA (450M) | ~2GB | GPU 1장으로 가능 |
| pi0 (3B) | ~8GB | GPU 1장으로 가능 |
| **pi0.5 Full FT** | ~70GB | **GPU 2장으로 가능** |
| 멀티 모델 동시 평가 | — | 4장 독립 실행 가능 |

---

## 현재 상태 (2026-04-03)

- [x] Docker 이미지 빌드 완료 (`bigenlight/libero-pro:v1.0`)
- [x] Docker Hub 푸시 완료
- [x] 버그 픽스 3개 적용
- [x] 컨테이너 검증 완료 (패키지, 벤치마크, Claude Code)
- [ ] 원격 서버에서 eval 검증
- [ ] `train` 태그 이미지 (데이터셋 포함) 빌드
- [ ] bc_transformer_policy baseline 학습
- [ ] VLA 모델 LIBERO-PRO 평가

---

## 적용된 버그 픽스

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

### Bug 3: `generate_init_states.py` — 포맷 불일치
```python
# 수정 전: zipfile+pickle (evaluate.py의 torch.load와 불일치)
with zipfile.ZipFile(output_filepath, 'w', ...) as zipf: ...
# 수정 후
torch.save(np.array(all_initial_states), output_filepath)
```

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
for INTENSITY in x0.1 x0.2 x0.3 x0.4 x0.5; do
  cp -r libero/libero/bddl_files/libero_object_temp_${INTENSITY}/* \
        libero/libero/bddl_files/libero_object_temp/
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

## 앞으로 방향

1. **eval 이미지 검증** — 원격 서버에서 벤치마크 로딩 및 환경 렌더링 확인
2. **train 이미지 빌드** — 학습 데이터셋(libero_spatial/object/goal/10) 포함 레이어 추가
3. **VLA 모델 통합** — OpenVLA-OFT 기준으로 LIBERO-PRO perturbation 평가 파이프라인 구축
4. **결과 공개** — 모델별 robustness drop 분석 및 leaderboard 제출

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
