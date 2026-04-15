#!/usr/bin/env python3
"""VLA evaluation driver for LIBERO / LIBERO-PRO suites.

Runs inside the `bigenlight/libero-pro` container. Talks to a VLA model server
(pi0.5, X-VLA, ...) over the unified HTTP protocol defined in
`VLA_COMMUNICATION_PROTOCOL.md` at repo root.

The benchmark side stays model-agnostic: we only pack raw robosuite observations
into `observation.*` keys and feed whatever sub-key `action.*` dict comes back
directly into `env.step()` as a 7-D robosuite OSC_POSE action.

Supported suites (any name registered in libero.libero.benchmark is accepted):
  - core: libero_spatial, libero_object, libero_goal, libero_10, libero_90
  - OOD : libero_{goal,spatial,object,10}_{swap,object,lan,task}

Usage (from inside the container, after run.sh --vla-eval libero_spatial):
    python scripts/libero_vla_eval.py \\
        --vla-url http://localhost:8400 \\
        --suite libero_spatial \\
        --num-tasks 2 \\
        --num-trials 1 \\
        --output-dir /workspace/LIBERO-PRO/test_outputs/eval
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import pathlib
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import imageio
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from vla_client import VLAClient  # noqa: E402

logger = logging.getLogger("libero_vla_eval")

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]  # open gripper, no arm motion
DEFAULT_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


@dataclass
class EvalConfig:
    vla_url: str = "http://localhost:8400"
    suite: str = "libero_spatial"
    num_tasks: int = 2
    num_trials: int = 1
    num_steps_wait: int = 10
    resolution: int = 256
    replan_steps: int = 5
    output_dir: str = "/workspace/LIBERO-PRO/test_outputs/eval"
    seed: int = 7
    max_steps_override: Optional[int] = None
    save_video: bool = True
    task_ids: Optional[List[int]] = None  # explicit task ids; overrides num_tasks
    shard_tag: str = ""                    # appended to run dir; useful for parallel shards


@dataclass
class TaskResult:
    task_id: int
    task_description: str
    successes: int = 0
    trials: int = 0
    latencies_ms: List[float] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _read_task_instruction_from_bddl(bddl_path: str, fallback: str) -> str:
    """Workaround for LIBERO-PRO issue #14 (`use_task=True` instruction bug)."""
    try:
        with open(bddl_path, "r") as f:
            content = f.read()
        m = re.search(r"\(:language\s*(.*?)\)", content, re.S)
        if m:
            return m.group(1).strip().strip('"').strip("'")
    except Exception:  # noqa: BLE001
        pass
    return fallback


def _build_states(obs: dict) -> Dict[str, np.ndarray]:
    """Project robosuite obs to unified `observation.state.*` keys.

    The server picks whichever keys it needs — we just hand over everything
    we have. All values are flat float32 arrays.
    """
    out: Dict[str, np.ndarray] = {}

    def put(key: str, value):
        if value is None:
            return
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        out[f"observation.state.{key}"] = arr

    put("eef_pos", obs.get("robot0_eef_pos"))
    put("eef_quat", obs.get("robot0_eef_quat"))  # xyzw
    put("gripper_qpos", obs.get("robot0_gripper_qpos"))
    put("gripper_qvel", obs.get("robot0_gripper_qvel"))
    put("joint_pos", obs.get("robot0_joint_pos"))
    put("joint_vel", obs.get("robot0_joint_vel"))
    return out


def _build_images(obs: dict) -> Dict[str, np.ndarray]:
    """Unified camera keys: `static` ← agentview, `wrist` ← eye-in-hand."""
    images: Dict[str, np.ndarray] = {}
    if "agentview_image" in obs:
        images["static"] = np.ascontiguousarray(obs["agentview_image"])
    if "robot0_eye_in_hand_image" in obs:
        images["wrist"] = np.ascontiguousarray(obs["robot0_eye_in_hand_image"])
    return images


def _assemble_action_from_subkeys(action_dict: Dict[str, np.ndarray]) -> np.ndarray:
    """Rebuild a 7-D robosuite OSC_POSE action from the server sub-keys.

    The unified protocol carries rotation in whichever slot the model provides;
    for pi0.5-LIBERO we asked the server to use `action.eef_euler` as the
    axis-angle delta slot, so a straight concat [pos(3), euler(3), gripper(1)]
    reconstructs the original 7-D action.

    Returns an ndarray of shape `(n_steps, 7)`.
    """
    pos = action_dict.get("action.eef_pos")
    if pos is None:
        raise ValueError(f"Missing action.eef_pos in server response: keys={list(action_dict)}")

    # Rotation: prefer euler (axis-angle slot for pi0.5). Fall back to zeros.
    rot = action_dict.get("action.eef_euler")
    if rot is None:
        rot = action_dict.get("action.eef_axis_angle")
    if rot is None:
        rot = np.zeros((pos.shape[0], 3), dtype=np.float32)

    grip = action_dict.get("action.gripper")
    if grip is None:
        grip = np.full((pos.shape[0], 1), -1.0, dtype=np.float32)  # open

    n = pos.shape[0]
    assert rot.shape[0] == n and grip.shape[0] == n, (
        f"Action chunk length mismatch: pos={pos.shape} rot={rot.shape} grip={grip.shape}"
    )
    return np.concatenate([pos, rot, grip], axis=1).astype(np.float32)


def _flatten_action(action_or_dict, n_action_steps_hint: int = 1) -> np.ndarray:
    if isinstance(action_or_dict, dict):
        return _assemble_action_from_subkeys(action_or_dict)
    arr = np.asarray(action_or_dict, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    return arr


# --------------------------------------------------------------------------- #
# Core eval loop
# --------------------------------------------------------------------------- #
def run_eval(cfg: EvalConfig) -> dict:
    # Lazy imports — these fail outside the libero container, so keep them here.
    from libero.libero import benchmark as _benchmark
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    os.makedirs(cfg.output_dir, exist_ok=True)
    shard_suffix = f"_{cfg.shard_tag}" if cfg.shard_tag else ""
    run_tag = f"{cfg.suite}_{time.strftime('%Y%m%d_%H%M%S')}{shard_suffix}"
    run_dir = pathlib.Path(cfg.output_dir) / run_tag
    video_dir = run_dir / "videos"
    if cfg.save_video:
        video_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(cfg.seed)

    # --- Connect to VLA server -------------------------------------------- #
    client = VLAClient(cfg.vla_url, timeout=120.0)
    logger.info("Waiting for VLA server at %s ...", cfg.vla_url)
    server_info = client.wait_until_ready(max_wait=300.0, poll_interval=3.0)
    logger.info("Server ready: %s", server_info)
    n_action_steps_hint = int(server_info.get("n_action_steps", 1))

    # --- Build suite ------------------------------------------------------ #
    bd = _benchmark.get_benchmark_dict()
    if cfg.suite not in bd:
        raise ValueError(
            f"Unknown suite {cfg.suite!r}. Registered: {sorted(bd.keys())[:10]} ..."
        )
    suite = bd[cfg.suite]()
    n_suite = suite.n_tasks
    if cfg.task_ids:
        task_id_list = [t for t in cfg.task_ids if 0 <= t < n_suite]
        if not task_id_list:
            raise ValueError(f"No valid task ids in {cfg.task_ids} for suite of {n_suite} tasks")
    else:
        n_tasks = min(cfg.num_tasks, n_suite) if cfg.num_tasks > 0 else n_suite
        task_id_list = list(range(n_tasks))
    logger.info(
        "Suite %s: running task ids %s (out of %d)",
        cfg.suite,
        task_id_list,
        n_suite,
    )

    # max_steps: either user override, or the longest demo rule, or fallback 300.
    max_steps = cfg.max_steps_override
    if max_steps is None:
        for k, v in DEFAULT_MAX_STEPS.items():
            if cfg.suite.startswith(k):
                max_steps = v
                break
    if max_steps is None:
        max_steps = 300

    results: List[TaskResult] = []
    total_episodes = 0
    total_successes = 0

    # --- Task loop -------------------------------------------------------- #
    for task_id in task_id_list:
        task = suite.get_task(task_id)
        bddl_path = str(
            pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        )
        task_description = _read_task_instruction_from_bddl(bddl_path, task.language)
        logger.info("Task %d: %s", task_id, task_description)

        init_states = suite.get_task_init_states(task_id)
        tr = TaskResult(task_id=task_id, task_description=task_description)

        for trial_idx in range(cfg.num_trials):
            env = OffScreenRenderEnv(
                bddl_file_name=bddl_path,
                camera_heights=cfg.resolution,
                camera_widths=cfg.resolution,
            )
            try:
                env.seed(cfg.seed)
                env.reset()
                init = init_states[trial_idx % len(init_states)]
                obs = env.set_init_state(init)

                client.reset()
                action_plan: Deque[np.ndarray] = collections.deque()
                replay: List[np.ndarray] = []
                done = False
                steps_executed = 0

                # LIBERO needs a few no-op steps for objects to settle.
                for _ in range(cfg.num_steps_wait):
                    obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

                while steps_executed < max_steps:
                    if not action_plan:
                        images = _build_images(obs)
                        states = _build_states(obs)
                        action_or_dict, latency_ms = client.predict(
                            images, states, task_description
                        )
                        tr.latencies_ms.append(latency_ms)
                        chunk = _flatten_action(action_or_dict, n_action_steps_hint)
                        n_use = min(len(chunk), max(1, cfg.replan_steps))
                        for i in range(n_use):
                            action_plan.append(chunk[i])

                    action = action_plan.popleft()
                    obs, reward, done, info = env.step(action.tolist())
                    if cfg.save_video:
                        replay.append(obs["agentview_image"][::-1])
                    steps_executed += 1
                    if done:
                        break

                tr.trials += 1
                if done:
                    tr.successes += 1
                    total_successes += 1
                total_episodes += 1

                logger.info(
                    "  trial %d: done=%s steps=%d latency_avg=%.1fms",
                    trial_idx,
                    done,
                    steps_executed,
                    float(np.mean(tr.latencies_ms)) if tr.latencies_ms else 0.0,
                )

                if cfg.save_video and replay:
                    safe = re.sub(r"[^a-zA-Z0-9_]+", "_", task_description)[:80]
                    suffix = "success" if done else "failure"
                    out_mp4 = video_dir / f"task{task_id:02d}_t{trial_idx}_{safe}_{suffix}.mp4"
                    imageio.mimwrite(str(out_mp4), replay, fps=10)
            finally:
                env.close()

        results.append(tr)

    # --- Report ----------------------------------------------------------- #
    summary = {
        "suite": cfg.suite,
        "vla_url": cfg.vla_url,
        "server_info": server_info,
        "num_tasks": len(task_id_list),
        "task_ids": task_id_list,
        "num_trials_per_task": cfg.num_trials,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "success_rate": (total_successes / total_episodes) if total_episodes else 0.0,
        "tasks": [
            {
                "task_id": r.task_id,
                "description": r.task_description,
                "successes": r.successes,
                "trials": r.trials,
                "avg_latency_ms": float(np.mean(r.latencies_ms)) if r.latencies_ms else 0.0,
                "num_predictions": len(r.latencies_ms),
            }
            for r in results
        ],
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("=" * 60)
    logger.info(
        "DONE  suite=%s  success=%d/%d (%.1f%%)",
        cfg.suite,
        total_successes,
        total_episodes,
        100.0 * summary["success_rate"],
    )
    logger.info("Wrote %s", run_dir / "summary.json")
    return summary


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vla-url", default=os.environ.get("VLA_SERVER_URL", "http://localhost:8400"))
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--num-tasks", type=int, default=2, help="0 = run all tasks in the suite")
    p.add_argument("--num-trials", type=int, default=1)
    p.add_argument("--num-steps-wait", type=int, default=10)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--replan-steps", type=int, default=5)
    p.add_argument(
        "--output-dir",
        default=os.environ.get("VLA_EVAL_OUTPUT_DIR", "/workspace/LIBERO-PRO/test_outputs/eval"),
    )
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max-steps", type=int, default=None, dest="max_steps_override")
    p.add_argument("--no-video", action="store_true")
    p.add_argument(
        "--task-ids",
        default="",
        help="comma-separated explicit task ids, e.g. '0,1,2,3'. Overrides --num-tasks.",
    )
    p.add_argument(
        "--shard-tag",
        default="",
        help="appended to the run directory name — use per-shard for parallel runs.",
    )
    args = p.parse_args()

    task_ids = None
    if args.task_ids.strip():
        task_ids = [int(x) for x in args.task_ids.split(",") if x.strip()]

    cfg = EvalConfig(
        vla_url=args.vla_url,
        suite=args.suite,
        num_tasks=args.num_tasks,
        num_trials=args.num_trials,
        num_steps_wait=args.num_steps_wait,
        resolution=args.resolution,
        replan_steps=args.replan_steps,
        output_dir=args.output_dir,
        seed=args.seed,
        max_steps_override=args.max_steps_override,
        save_video=not args.no_video,
        task_ids=task_ids,
        shard_tag=args.shard_tag,
    )
    run_eval(cfg)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()
