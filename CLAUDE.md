# CLAUDE.md — LIBERO-PRO 프로젝트 컨텍스트

> 이 파일은 Claude Code가 대화 시작 시 자동으로 읽는 프로젝트 instruction 파일이다.

---

## 프로젝트 개요

LIBERO-PRO 벤치마크로 VLA(Vision-Language-Action) 모델의 OOD 일반화 능력을 평가하는 실험 환경.
Docker 이미지(`bigenlight/libero-pro:latest`) 기반으로 재현 가능한 실험 파이프라인을 구축 중.

---

## 리포 구조

```
Libero-pro/
├── CLAUDE.md              # 이 파일 (프로젝트 컨텍스트)
├── README.md              # 전체 setup 가이드 + 실험 계획
├── LIBERO_PRO_PLAN.md     # 실험 상세 계획서
├── Dockerfile             # Docker 이미지 빌드 (CUDA 11.3 + robosuite + Claude Code)
├── run.sh                 # Docker 컨테이너 실행 스크립트
├── test_local.py          # 컨테이너 내부 10단계 검증 스크립트
├── record_pro_video.py    # LIBERO-PRO OOD 환경 영상 녹화 스크립트
├── record_comparison.py   # 원본 LIBERO vs OOD 비교 영상 녹화 스크립트
├── .gitignore             # test_outputs/ 제외
├── ood_data/              # LIBERO-PRO OOD suite bddl/init 파일 (repo-tracked, ~2MB)
│   ├── bddl_files/            # 16 OOD suites × 10 tasks
│   └── init_files/            # torch.save 형식 init states
├── LIBERO/                # 원본 LIBERO 벤치마크 (.gitignore 됨 — 호스트에서 직접 clone)
└── LIBERO-PRO/            # LIBERO-PRO 확장 코드 (.gitignore 됨 — 호스트에서 직접 clone)
    ├── perturbation.py        # OOD perturbation 엔진 (5종)
    ├── evaluation_config.yaml # perturbation 플래그 설정
    ├── libero_ood/            # OOD YAML configs (5개)
    └── libero/                # LIBERO 코드 + PRO 확장 benchmark 등록
```

---

## 원본 코드 변경 이력

> **LIBERO/, LIBERO-PRO/ 코드는 수정하지 않았다. Dockerfile은 2026-04-08에 수정.**
> 아래는 리포 루트에서 추가/수정한 파일만 기록한다.

### 2026-04-07: 로컬 Docker 검증 환경 구축

#### `run.sh` (수정 — 기존 빈 파일 0 bytes → 177줄)

원래 빈 파일이었던 `run.sh`를 Docker 컨테이너 실행 스크립트로 재작성했다.

- Docker 이미지 pull 여부 자동 체크
- nvidia runtime / GPU 존재 확인 (pre-flight)
- `--gpus all --shm-size=8g` 으로 컨테이너 실행
- EGL 렌더링 우선 시도, 실패 시 자동 OSMesa fallback
- Claude Code 인증 자동 주입 (구독 `.credentials.json` 마운트 또는 `--api-key` 플래그)
- `--api-key` 플래그와 `ANTHROPIC_API_KEY` 환경변수 이중 주입 방지 로직
- `test_local.py`를 read-only 마운트하여 컨테이너 내부에서 실행
- 출력물(PNG/MP4)은 호스트의 `./test_outputs/`에 volume mount
- `--shell` 모드로 대화형 셸 진입 가능

> **주의 (bash -c + 변수 결합)**: `ENTRYPOINT_CMD` 변수 안에 heredoc이나 multiline if-fi 블록이 있으면
> `bash -c "$VAR && cmd"` 형태로 연결 시 syntax error 발생. trailing newline 뒤에 `&&`가 오면
> bash가 파싱을 실패함. 해결: `&&` 대신 개행(`\n`)으로 명령을 분리 (`bash -c "$VAR\ncmd"`).

```bash
./run.sh                          # 기본 실행 (EGL → osmesa fallback)
./run.sh --skip-video             # 비디오 테스트 스킵
./run.sh --skip-pro               # LIBERO-PRO 테스트 스킵
./run.sh --api-key sk-ant-xxxx    # API 키 방식
./run.sh --shell                  # bash 셸 진입
```

#### `test_local.py` (신규 — 481줄)

컨테이너 내부(`/workspace/LIBERO-PRO`)에서 실행되는 10단계 검증 스크립트.

| # | 테스트 | GPU 필요 | 검증 내용 |
|---|--------|----------|-----------|
| 1 | Basic Environment | X | Python 3.8, PyTorch 1.11, CUDA, robosuite 1.4.0, bddl/init 경로 |
| 2 | LIBERO Benchmark Loading | X | 5개 suite 로딩, task 수 (10/10/10/10/90), bddl 파일 존재 |
| 3 | Env Creation + Reset | O | OffScreenRenderEnv 생성, obs dict 검증, 128x128 이미지 |
| 4 | Rendering (PNG) | O | agentview + eye-in-hand 이미지 → PNG 저장 |
| 5 | Simulation Step | O | 10 step zero action, reward/done/check_success 검증 |
| 6 | LIBERO-PRO Loading | X | OOD benchmark suite 20개 등록 확인 |
| 7 | Perturbation System | X | BDDLParser + 5개 perturbator in-memory 테스트 |
| 8 | OOD File Verification | X | 20개 OOD suite bddl/init 파일 존재 확인 |
| 9 | OOD Env Creation | O | libero_goal_swap 환경 생성, init state 로딩, 시뮬레이션 |
| 10 | Video Rendering | O | 30 frame rollout → mp4 저장 |

설계 결정:
- 각 env 테스트는 `contextmanager`로 `env.close()` 보장 (GPU 메모리 누수 방지)
- 128x128 해상도 사용 (RTX 3060 12GB에서 안전)
- `signal.SIGALRM` 기반 120초 timeout (env.reset()의 RandomizationError 무한루프 방지)
- Perturbation 테스트는 디스크 쓰기 없이 in-memory만 (process_bddl_file_mixed 호출 안 함)
- OOD suite bddl/init 파일은 이미지 빌드 시 HuggingFace에서 내장됨 (v1.1부터)

#### 코드 리뷰에서 발견/수정한 버그

| 버그 | 위치 | 수정 |
|------|------|------|
| `total_mem` AttributeError | `test_local.py` | PyTorch 속성명은 `total_memory` → 수정 |
| 빌트인 `TimeoutError` 섀도잉 | `test_local.py` | `TestTimeoutError`로 커스텀 클래스명 변경 |
| bddl/init 디렉토리 assert 누락 | `test_local.py` | `os.path.isdir()` assert 추가 |
| `obs` 참조가 env.close() 이후 | `test_local.py` | `with` 블록 안으로 이동 |
| API key 이중 주입 | `run.sh` | `--api-key` 플래그 우선, 환경변수 중복 시 스킵 |

#### `CLAUDE.md` (신규 — 이 파일)

프로젝트 컨텍스트와 변경 이력을 기록하여 다음 대화에서 자동으로 읽히도록 함.

---

### 2026-04-08: LIBERO-PRO OOD 데이터 이미지 내장 + 테스트 확장

#### `Dockerfile` (수정)
- HuggingFace `zhouxueyang/LIBERO-Pro`에서 OOD bddl/init 파일 다운로드 단계 추가
- `allow_patterns`로 bddl_files, init_files만 선별 다운로드 (demo HDF5 제외)
- 이미지 크기 최소화를 위해 HF 캐시 삭제
- `cp -rn` (no-clobber)으로 기존 LIBERO 원본 bddl 파일 덮어쓰기 방지

#### `run.sh` (수정)
- `IMAGE` 변수를 `LIBERO_IMAGE` 환경변수로 오버라이드 가능하게 변경
  - `LIBERO_IMAGE=bigenlight/libero-pro:v1.0 ./run.sh` 형태로 이미지 버전 전환 가능

#### `record_pro_video.py` (신규)
- LIBERO-PRO OOD 환경에서 256x256 해상도 영상 녹화
- 4개 OOD suite (goal_swap, spatial_object, goal_lan, 10_swap) 자동 녹화

#### `record_comparison.py` (신규)
- 원본 LIBERO vs OOD 동일 task 비교 영상 3쌍 녹화
- 같은 random seed로 로봇 동작 동일, 환경 차이만 비교 가능

#### `.gitignore` (신규)
- `test_outputs/` 디렉토리 제외 (생성된 PNG/MP4 산출물)

#### `test_local.py` (수정 — 8단계 → 10단계)
- Test 8: OOD bddl/init 파일 존재 검증 (CPU, 20개 suite)
- Test 9: OOD 환경 생성 + Simulation Step (GPU, libero_goal_swap)
- 기존 Test 8 (Video) → Test 10으로 변경

---

### 2026-04-21: LIBERO / LIBERO-PRO 소스 완전 마운트화 (v1.3)

#### 배경
v1.2 까지는 OOD 데이터만 마운트였고 LIBERO-PRO 소스는 여전히 `COPY LIBERO-PRO` 로 이미지에 구워졌다.
v1.3 에서는 **소스도 호스트 clone 을 bind mount** 하도록 전환해서 "코드/데이터/결과는 호스트에" 원칙을 완전히 적용한다. 이미지 사이즈도 줄고, 소스 수정이 재빌드 없이 즉시 반영된다.

#### `Dockerfile` (수정)
- `COPY LIBERO-PRO /workspace/LIBERO-PRO` **제거**
- `pip install -e .` 단계 제거 (런타임에 `PYTHONPATH` 로 해결)
- 이미지 빌드 시 LIBERO-PRO upstream 에서 **임시 git clone → requirements.txt 설치 → 소스 삭제** 패턴으로 전환
- `ENV PYTHONPATH=/workspace/LIBERO-PRO:/workspace/LIBERO:${PYTHONPATH}` 추가 — 마운트만 되면 `libero` / `perturbation` import 가능
- 빌드 검증은 `torch` / `robosuite` 만 확인 (libero 는 런타임 마운트 이후에만 존재)

#### `run.sh` (수정)
- 호스트 경로 변수 추가: `LIBERO_PRO_SRC=$(pwd)/LIBERO-PRO`, `LIBERO_SRC=$(pwd)/LIBERO`
- `SRC_MOUNT` 구성 로직 추가:
  - LIBERO-PRO 없으면 **에러 종료** + clone 명령 안내 (필수)
  - LIBERO 없으면 경고만 (optional, ro mount)
- 4개 `docker run` 블록 (vla-eval / shell / EGL / osmesa) 모두 `$SRC_MOUNT` 적용

#### `.dockerignore` (수정)
- `LIBERO-PRO` 추가 (COPY 안 하므로 빌드 context 에서 제외 → 빌드 속도 향상)
- `test_outputs`, `*.pdf`, `image.png` 도 추가

#### 호스트 사전 준비 (사용자가 직접 한 번만 실행)
```bash
cd Libero-pro_benchmark
git clone https://github.com/Zxy-MLlab/LIBERO-PRO.git LIBERO-PRO
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git LIBERO
```

두 디렉토리는 `.gitignore` 에 등재되어 있어서 parent repo 에 포함되지 않는다.

---

### 2026-04-15: OOD 데이터 repo 편입 + 마운트 기반 개발 워크플로우

#### 배경
LIBERO-PRO 코드를 수정하며 개발할 일이 생길 것이므로, 이미지에 모든 걸 굽기보다는
**코드/데이터는 호스트에서 마운트**하는 방향으로 전환. CLAUDE.md 의 "개발 워크플로우"
원칙 (*"컨테이너는 일회용 실행환경, 코드/데이터/결과는 호스트에"*) 과도 일치.

#### `ood_data/` (신규 — 1.9MB)
- 16개 OOD suite × 10 task × 2 (bddl + init) = 320개 파일
- `docker cp` 로 기존 v1.1 이미지에서 추출 후 커밋
- init 파일은 `torch.save()` 형식이라 zip 시그니처로 보이지만 정상
- `ood_environment` 는 상류 데이터 미완이라 repo 에 포함하지 않음 (기존과 동일)

#### `Dockerfile` (수정)
- HuggingFace OOD 다운로드 단계 **제거** — 이제 OOD 데이터는 `ood_data/` 가 단일 진실 원본
- 결과: 이미지 빌드 시간 단축, `huggingface_hub` 의존성 축소, 빌드 결정적
- 이미지 자체는 OOD 데이터가 없으므로, `run.sh` 없이 생(raw)으로 실행하면 OOD 테스트는 실패
- 원본 LIBERO bddl/init (비 OOD) 은 그대로 `COPY LIBERO-PRO` 로 이미지에 포함

#### `run.sh` (수정)
- `OOD_DATA_DIR` 변수 및 마운트 로직 추가 — `$(pwd)/ood_data` → `/tmp/ood_data:ro`
- `ENTRYPOINT_CMD` 에 컨테이너 시작 시 OOD 파일 병합 단계 추가:
  - `cp -rn /tmp/ood_data/bddl_files/* /workspace/LIBERO-PRO/libero/libero/bddl_files/`
  - `cp -rn` (no-clobber) 이므로 원본 LIBERO suites 는 안 덮어씀
- 3개 `docker run` 블록 (shell / EGL / osmesa) 모두에 `$OOD_MOUNT` 적용

---

## Docker 이미지 정보

- **이미지**: `bigenlight/libero-pro:latest` (= `v1.3`, 2026-04-21)
- **크기**: ~15GB (v1.2 ~23GB 대비 LIBERO-PRO 소스/egg-info 제거로 감소)
- **WORKDIR**: `/workspace/LIBERO-PRO` — **빌드 시점엔 비어있음**, 런타임에 호스트가 마운트
- **렌더링**: `MUJOCO_GL=egl` (기본), `osmesa` (fallback)
- **OOD 데이터**: 이미지에 없음 — `ood_data/` 가 mount + copy 되어 런타임에 병합
- **LIBERO-PRO / LIBERO 소스**: 이미지에 없음 — 호스트 clone 이 bind mount 되어야 동작
- **버그 3개**: v1.3 에서는 **수정되지 않은 upstream 을 마운트**하므로 더 이상 이미지 레이어로 보유하지 않음. 필요한 경우 사용자가 호스트 clone 에 직접 패치 적용.
- **PYTHONPATH** 환경변수가 `/workspace/LIBERO-PRO:/workspace/LIBERO` 를 포함하므로 마운트만 되면 `libero`, `perturbation` import 가능 (editable install 불필요)

---

## 현재 진행 상태

- [x] Docker 이미지 빌드/푸시/검증 완료
- [x] 로컬 검증 스크립트 작성 완료 (`run.sh` + `test_local.py`)
- [x] 로컬에서 `./run.sh` 실행 — **10/10 PASS** (EGL, RTX 3060 12GB, v1.1, 2026-04-08)
- [x] LIBERO-PRO OOD 데이터 이미지 내장 (v1.1)
- [x] OOD 환경 테스트 추가 (test 8, 9)
- [x] 원격 서버(aadd, RTX A6000 x4) eval 환경 검증 — **10/10 PASS** (v1.1, 2026-04-15)
- [x] OOD 데이터 repo 편입 + 마운트 기반 워크플로우 (v1.2, 2026-04-15)
- [x] 원격 서버(aadd, RTX A6000) v1.2 재검증 — **10/10 PASS** (2026-04-15)
- [x] LIBERO / LIBERO-PRO 소스 완전 마운트화 (v1.3, 2026-04-21)
- [ ] train 이미지 빌드 (데이터셋 포함)
- [ ] bc_transformer baseline 학습
- [ ] VLA 모델 LIBERO-PRO 평가

---

## 알려진 이슈

| Issue | 상태 | 비고 |
|-------|------|------|
| `use_task` 언어 지시 버그 (#14) | 미해결 | VLA 통합 시 workaround 필요 |
| OpenVLA 결과 불일치 (#20) | 미확인 | 평가 전 확인 필요 |
| Environment BDDL 미출시 (#9) | 업스트림 대기 | `ood_environment.yaml`은 있으나 데이터 미완 |
| OOD suite init_states | 해결 (v1.2) | `ood_data/` 가 repo 에 편입되어 `run.sh` 가 런타임에 병합 |

---

## 개발 워크플로우 (서버 작업 시)

> 2026-04-08 논의 후 결정된 워크플로우. v1.3 (2026-04-21) 에서 소스까지 마운트로 확장.

**원칙: 컨테이너는 일회용 실행환경, 코드/데이터/결과는 호스트에.**

- **docker commit 사용 금지** — 레이어 누적으로 스토리지 폭발, 재현 불가
- **bind mount** — LIBERO-PRO, LIBERO, ood_data, scripts, test_outputs 모두 호스트에서 마운트
- **환경 변경은 Dockerfile에 기록** — pip install 추가 시 Dockerfile 수정 후 재빌드
- **LIBERO-PRO 소스 수정** — 호스트 `LIBERO-PRO/` 에서 직접 수정 (feedback: 원본 수정 금지이므로 병렬 파일 작성 권장)
- **named container** — `--rm` 없이 컨테이너 유지, `docker exec`로 재진입
- **tmux** — SSH 세션 영속화, 실험 돌리는 중에 detach 가능

```bash
# 서버에서의 일상 워크플로우
ssh server
tmux attach -t libero
docker exec -it libero-pro bash    # 기존 컨테이너 재진입
# 코드 편집 → 실험 → git commit (호스트에서)
```

---

## 핵심 API 레퍼런스 (자주 쓰는 것)

```python
# 벤치마크 로딩
from libero.libero import benchmark, get_libero_path
bd = benchmark.get_benchmark_dict()
suite = bd["libero_spatial"]()
bddl = suite.get_task_bddl_file_path(task_id)

# 환경 생성 (headless)
from libero.libero.envs.env_wrapper import OffScreenRenderEnv
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
obs = env.reset()                    # dict, 'agentview_image' key
obs, reward, done, info = env.step(np.zeros(env.env.action_dim))
env.close()                          # 반드시 호출

# Perturbation (in-memory)
from perturbation import BDDLParser, SwapPerturbator  # /workspace/LIBERO-PRO/perturbation.py
parser = BDDLParser(bddl_content)
p = SwapPerturbator(parser, "libero_ood/ood_spatial_relation.yaml")
result = p.perturb("libero_goal", "task_name")
```
