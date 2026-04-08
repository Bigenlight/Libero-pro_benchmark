#!/usr/bin/env python3
"""LIBERO / LIBERO-PRO Docker 로컬 검증 스크립트.

컨테이너 내부(/workspace/LIBERO-PRO)에서 실행.
10단계 테스트를 통해 벤치마크 환경이 정상 동작하는지 확인한다.
"""

import sys
import os
import time
import argparse
import traceback
import signal
import numpy as np
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List

# ── Result tracking ──────────────────────────────────────────────

@dataclass
class TestResult:
    name: str
    passed: bool
    elapsed: float
    detail: str

results: List[TestResult] = []

def run_test(name, func, *args, **kwargs):
    """Run a test function, catch exceptions, record result."""
    print(f"\n{'='*60}")
    print(f"  TEST: {name}")
    print(f"{'='*60}")
    t0 = time.time()
    try:
        func(*args, **kwargs)
        elapsed = time.time() - t0
        results.append(TestResult(name, True, elapsed, "OK"))
        print(f"\n  ✓ PASS  ({elapsed:.1f}s)")
    except Exception as e:
        elapsed = time.time() - t0
        detail = f"{type(e).__name__}: {e}"
        results.append(TestResult(name, False, elapsed, detail))
        print(f"\n  ✗ FAIL  ({elapsed:.1f}s)")
        print(f"    {detail}")
        traceback.print_exc()

# ── Timeout helper ───────────────────────────────────────────────

class TestTimeoutError(Exception):
    pass

@contextmanager
def timeout(seconds, msg="Timeout"):
    def _handler(signum, frame):
        raise TestTimeoutError(msg)
    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)

# ── Env context manager (GPU leak 방지) ─────────────────────────

@contextmanager
def make_env(bddl_path, height=128, width=128):
    from libero.libero.envs.env_wrapper import OffScreenRenderEnv
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=height,
        camera_widths=width,
    )
    try:
        yield env
    finally:
        env.close()

# ═══════════════════════════════════════════════════════════════
# TEST 1: Basic Environment
# ═══════════════════════════════════════════════════════════════

def test_basic_env(args):
    print(f"  Python: {sys.version}")

    import torch
    print(f"  PyTorch: {torch.__version__}")
    assert torch.cuda.is_available(), "CUDA not available!"
    print(f"  CUDA available: True")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    import robosuite
    print(f"  robosuite: {robosuite.__version__}")

    import libero.libero
    print(f"  libero: imported OK")

    from libero.libero import get_libero_path
    bddl_path = get_libero_path("bddl_files")
    init_path = get_libero_path("init_states")
    assert os.path.isdir(bddl_path), f"bddl_files dir missing: {bddl_path}"
    print(f"  bddl_files: {bddl_path} ✓")
    assert os.path.isdir(init_path), f"init_states dir missing: {init_path}"
    print(f"  init_states: {init_path} ✓")

    print(f"  MUJOCO_GL: {os.environ.get('MUJOCO_GL', 'not set')}")

# ═══════════════════════════════════════════════════════════════
# TEST 2: LIBERO Benchmark Loading
# ═══════════════════════════════════════════════════════════════

CORE_SUITES = {
    "libero_spatial": 10,
    "libero_object": 10,
    "libero_goal": 10,
    "libero_10": 10,
    "libero_90": 90,
}

def test_benchmark_loading(args):
    from libero.libero import benchmark
    bd = benchmark.get_benchmark_dict()
    print(f"  Registered benchmarks: {len(bd)}")

    for suite_name, expected_count in CORE_SUITES.items():
        assert suite_name in bd, f"{suite_name} not registered!"
        suite = bd[suite_name]()
        n = suite.n_tasks
        names = suite.get_task_names()
        assert n == expected_count, f"{suite_name}: expected {expected_count} tasks, got {n}"
        assert len(names) == n
        # bddl 파일 존재 확인 (첫 번째 task만)
        bddl = suite.get_task_bddl_file_path(0)
        assert os.path.exists(bddl), f"BDDL missing: {bddl}"
        print(f"  {suite_name}: {n} tasks, bddl[0] exists ✓")

# ═══════════════════════════════════════════════════════════════
# TEST 3: Env Creation + Reset
# ═══════════════════════════════════════════════════════════════

def _get_first_bddl():
    from libero.libero import benchmark
    bd = benchmark.get_benchmark_dict()
    suite = bd["libero_spatial"]()
    return suite.get_task_bddl_file_path(0), suite

def test_env_creation(args):
    bddl_path, _ = _get_first_bddl()
    print(f"  BDDL: {bddl_path}")

    with timeout(120, "Env creation + reset timed out (120s)"):
        with make_env(bddl_path) as env:
            obs = env.reset()
            assert isinstance(obs, dict), f"obs is {type(obs)}, expected dict"
            assert "agentview_image" in obs, f"obs keys: {list(obs.keys())}"
            img = obs["agentview_image"]
            assert img.shape == (128, 128, 3), f"Image shape: {img.shape}"
            print(f"  Env created and reset OK")
            print(f"  obs keys: {list(obs.keys())}")
            print(f"  agentview_image shape: {img.shape}, dtype: {img.dtype}")

# ═══════════════════════════════════════════════════════════════
# TEST 4: Rendering (PNG output)
# ═══════════════════════════════════════════════════════════════

def test_rendering(args):
    import cv2
    bddl_path, _ = _get_first_bddl()

    with timeout(120, "Rendering timed out"):
        with make_env(bddl_path) as env:
            obs = env.reset()
            img = obs["agentview_image"]

            # agentview
            out_path = os.path.join(args.output_dir, "test_agentview.png")
            cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            assert os.path.exists(out_path) and os.path.getsize(out_path) > 0
            print(f"  Saved: {out_path} ({os.path.getsize(out_path)} bytes)")

            # eye-in-hand (if available)
            if "robot0_eye_in_hand_image" in obs:
                eih = obs["robot0_eye_in_hand_image"]
                eih_path = os.path.join(args.output_dir, "test_eye_in_hand.png")
                cv2.imwrite(eih_path, cv2.cvtColor(eih, cv2.COLOR_RGB2BGR))
                print(f"  Saved: {eih_path} ({os.path.getsize(eih_path)} bytes)")

# ═══════════════════════════════════════════════════════════════
# TEST 5: Simulation Step
# ═══════════════════════════════════════════════════════════════

def test_simulation_step(args):
    bddl_path, _ = _get_first_bddl()
    n_steps = 10

    with timeout(120, "Simulation step timed out"):
        with make_env(bddl_path) as env:
            obs = env.reset()
            action_dim = env.env.action_dim
            print(f"  Action dim: {action_dim}")

            for i in range(n_steps):
                action = np.zeros(action_dim)
                obs, reward, done, info = env.step(action)
                assert isinstance(obs, dict)
                assert "agentview_image" in obs
                assert obs["agentview_image"].shape == (128, 128, 3)

            print(f"  {n_steps} steps OK")
            print(f"  Last reward: {reward}, done: {done}")

            success = env.check_success()
            print(f"  check_success(): {success}")

# ═══════════════════════════════════════════════════════════════
# TEST 6: LIBERO-PRO Benchmark Loading
# ═══════════════════════════════════════════════════════════════

OOD_SUITES = [
    "libero_goal_swap", "libero_goal_object", "libero_goal_lan",
    "libero_goal_task", "libero_goal_env",
    "libero_spatial_swap", "libero_spatial_object", "libero_spatial_lan",
    "libero_spatial_task", "libero_spatial_env",
    "libero_object_swap", "libero_object_object", "libero_object_lan",
    "libero_object_task", "libero_object_env",
    "libero_10_swap", "libero_10_object", "libero_10_lan",
    "libero_10_task", "libero_10_env",
]

def test_libero_pro_loading(args):
    from libero.libero import benchmark
    bd = benchmark.get_benchmark_dict()

    found = 0
    missing = []
    for name in OOD_SUITES:
        if name in bd:
            suite = bd[name]()
            n = suite.n_tasks
            assert n > 0, f"{name}: 0 tasks"
            found += 1
        else:
            missing.append(name)

    print(f"  OOD suites found: {found}/{len(OOD_SUITES)}")
    if missing:
        print(f"  Missing (non-critical): {missing}")

    assert found > 0, "No OOD benchmark suites found at all!"

# ═══════════════════════════════════════════════════════════════
# TEST 7: Perturbation System (CPU only, in-memory)
# ═══════════════════════════════════════════════════════════════

def test_perturbation(args):
    sys.path.insert(0, "/workspace/LIBERO-PRO")
    from perturbation import (
        BDDLParser,
        SwapPerturbator,
        ObjectReplacePerturbator,
        LanguagePerturbator,
        TaskPerturbator,
        EnvironmentReplacePerturbator,
    )
    from libero.libero import benchmark, get_libero_path

    # 테스트할 task 선택: libero_goal의 첫 번째 task
    bd = benchmark.get_benchmark_dict()
    suite = bd["libero_goal"]()
    task_name = suite.get_task_names()[0]
    bddl_path = suite.get_task_bddl_file_path(0)

    print(f"  Task: {task_name}")
    print(f"  BDDL: {bddl_path}")

    with open(bddl_path, "r") as f:
        content = f.read()

    # BDDLParser
    parser = BDDLParser(content)
    print(f"  BDDLParser: objects={parser.objects_of_interest}, init_states={len(parser.initial_states)} entries")

    ood_dir = "/workspace/LIBERO-PRO/libero_ood"
    suite_name = "libero_goal"

    perturbators = {
        "Swap": (SwapPerturbator, os.path.join(ood_dir, "ood_spatial_relation.yaml"), False),
        "Object": (ObjectReplacePerturbator, os.path.join(ood_dir, "ood_object.yaml"), True),
        "Language": (LanguagePerturbator, os.path.join(ood_dir, "ood_language.yaml"), True),
        "Task": (TaskPerturbator, os.path.join(ood_dir, "ood_task.yaml"), True),
        "Environment": (EnvironmentReplacePerturbator, os.path.join(ood_dir, "ood_environment.yaml"), True),
    }

    for name, (cls, config_path, has_seed) in perturbators.items():
        if not os.path.exists(config_path):
            print(f"  {name}: config missing ({config_path}), SKIP")
            continue

        p = BDDLParser(content)  # fresh parser each time
        perturbator = cls(p, config_path)
        if has_seed:
            result = perturbator.perturb(suite_name, task_name, seed=42)
        else:
            result = perturbator.perturb(suite_name, task_name)

        assert isinstance(result, str), f"{name} returned {type(result)}"
        changed = result != content
        print(f"  {name}: returned str (len={len(result)}), changed={changed}")

# ═══════════════════════════════════════════════════════════════
# TEST 8: OOD bddl/init 파일 검증 (CPU only)
# ═══════════════════════════════════════════════════════════════

# ood_environment 계열은 데이터 미완 (known issue) → 스킵
OOD_ENV_SKIP = {
    "libero_goal_env", "libero_spatial_env",
    "libero_object_env", "libero_10_env",
}

def test_ood_files(args):
    from libero.libero import benchmark
    bd = benchmark.get_benchmark_dict()

    checked = 0
    skipped = 0
    for name in OOD_SUITES:
        if name in OOD_ENV_SKIP:
            print(f"  {name}: SKIP (ood_environment data incomplete)")
            skipped += 1
            continue
        if name not in bd:
            print(f"  {name}: not registered, SKIP")
            skipped += 1
            continue

        suite = bd[name]()
        # bddl 파일 확인 (task 0)
        bddl_path = suite.get_task_bddl_file_path(0)
        assert os.path.exists(bddl_path), f"{name} task 0 bddl missing: {bddl_path}"

        # init_states 파일 확인
        try:
            init_states = suite.get_task_init_states(0)
            assert init_states is not None, f"{name} task 0 init_states is None"
            assert len(init_states) > 0, f"{name} task 0 init_states is empty"
            print(f"  {name}: bddl ✓, init_states ({len(init_states)} states) ✓")
        except FileNotFoundError as e:
            print(f"  {name}: bddl ✓, init_states MISSING ({e})")
            raise

        checked += 1

    print(f"  Checked: {checked}, Skipped: {skipped}")
    assert checked > 0, "No OOD suites were checked!"

# ═══════════════════════════════════════════════════════════════
# TEST 9: OOD 환경 생성 + Step (GPU)
# ═══════════════════════════════════════════════════════════════

def test_ood_env_step(args):
    from libero.libero import benchmark
    bd = benchmark.get_benchmark_dict()

    suite_name = "libero_goal_swap"
    assert suite_name in bd, f"{suite_name} not registered!"
    suite = bd[suite_name]()
    bddl_path = suite.get_task_bddl_file_path(0)
    task_name = suite.get_task_names()[0]
    print(f"  Suite: {suite_name}")
    print(f"  Task: {task_name}")
    print(f"  BDDL: {bddl_path}")

    # init_states 로드
    init_states = suite.get_task_init_states(0)
    print(f"  init_states: {len(init_states)} states, shape={init_states[0].shape}")

    with timeout(120, "OOD env creation + step timed out (120s)"):
        with make_env(bddl_path) as env:
            # set_init_state로 초기 상태 설정
            obs = env.set_init_state(init_states[0])
            print(f"  set_init_state OK")

            obs = env.reset()
            assert isinstance(obs, dict), f"obs is {type(obs)}, expected dict"
            assert "agentview_image" in obs
            img = obs["agentview_image"]
            assert img.shape == (128, 128, 3), f"Image shape: {img.shape}"
            print(f"  Env reset OK, obs keys: {list(obs.keys())}")

            # 5 step zero action
            action_dim = env.env.action_dim
            for i in range(5):
                obs, reward, done, info = env.step(np.zeros(action_dim))
                assert isinstance(obs, dict)
                assert "agentview_image" in obs

            print(f"  5 steps OK, last reward={reward}, done={done}")

# ═══════════════════════════════════════════════════════════════
# TEST 10: Video Rendering (optional)
# ═══════════════════════════════════════════════════════════════

def test_video_rendering(args):
    from libero.libero.utils.video_utils import VideoWriter
    bddl_path, _ = _get_first_bddl()
    video_dir = os.path.join(args.output_dir, "test_video")
    n_frames = 30

    with timeout(120, "Video rendering timed out"):
        with make_env(bddl_path) as env:
            obs = env.reset()
            with VideoWriter(video_dir, save_video=True, fps=30) as writer:
                writer.append_image(obs["agentview_image"][::-1])
                action_dim = env.env.action_dim
                for _ in range(n_frames - 1):
                    obs, _, _, _ = env.step(np.zeros(action_dim))
                    writer.append_image(obs["agentview_image"][::-1])

    video_path = os.path.join(video_dir, "video.mp4")
    assert os.path.exists(video_path), f"Video not found: {video_path}"
    size = os.path.getsize(video_path)
    assert size > 1024, f"Video too small: {size} bytes"
    print(f"  Saved: {video_path} ({size} bytes, {n_frames} frames)")

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def print_summary():
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    for r in results:
        status = "✓ PASS" if r.passed else "✗ FAIL"
        print(f"  {status}  {r.name}  ({r.elapsed:.1f}s)")
        if not r.passed:
            print(f"         {r.detail}")
    print(f"\n  Result: {passed}/{total} passed")
    print(f"{'='*60}")

def main():
    parser = argparse.ArgumentParser(description="LIBERO/LIBERO-PRO local validation")
    parser.add_argument("--output-dir", default="/workspace/LIBERO-PRO/test_outputs")
    parser.add_argument("--skip-video", action="store_true", help="Skip video rendering test")
    parser.add_argument("--skip-pro", action="store_true", help="Skip LIBERO-PRO tests")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # CPU-only tests (1, 2) — 이것들은 GPU 실패와 무관하게 돌아감
    run_test("1. Basic Environment", test_basic_env, args)
    run_test("2. LIBERO Benchmark Loading", test_benchmark_loading, args)

    # GPU tests (3, 4, 5)
    run_test("3. Env Creation + Reset", test_env_creation, args)
    run_test("4. Rendering (PNG)", test_rendering, args)
    run_test("5. Simulation Step", test_simulation_step, args)

    # LIBERO-PRO tests (6, 7, 8, 9)
    if not args.skip_pro:
        run_test("6. LIBERO-PRO Loading", test_libero_pro_loading, args)
        run_test("7. Perturbation System", test_perturbation, args)
        run_test("8. OOD bddl/init Files", test_ood_files, args)
        run_test("9. OOD Env Creation + Step", test_ood_env_step, args)

    # Video (GPU, optional)
    if not args.skip_video:
        run_test("10. Video Rendering", test_video_rendering, args)

    print_summary()

    passed = sum(1 for r in results if r.passed)
    sys.exit(0 if passed == len(results) else 1)

if __name__ == "__main__":
    main()
