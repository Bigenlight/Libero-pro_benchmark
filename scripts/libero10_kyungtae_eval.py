#!/usr/bin/env python3
"""LIBERO-10 (standard, no-perturbation) eval driver with full trajectory dump.

Runs inside the `bigenlight/libero-pro` container. Talks to a VLA model server
over the unified HTTP protocol (`VLA_COMMUNICATION_PROTOCOL.md`).

Standard LIBERO-10 ("libero_10" suite) with the canonical pruned init states
— NO Libero-Pro OOD perturbation — i.e. trial k uses `init_states[k]` only.

Output layout (`--output-root /workspace/kyungtae_data`):

    <root>/<model>/<suite>/run_<UTC>/
        config.json
        summary.json
        summary.csv          (one row per episode; flushed every episode)
        run.log
        task<NN>_<bddl_stem>/
            videos/<status>_<TT>_<short>.mp4
            trajectories/<status>_<TT>_<short>.hdf5

Each `.hdf5` holds a single episode with full obs / action / reward / done /
sim_state stacks plus root attrs (task_id, success, init_seed, etc.), so any
episode can be deterministically replayed via
`env.set_init_state(init) + env.set_state(sim_states[t])`.
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as _dt
import json
import logging
import os
import pathlib
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Deque, Dict, List, Optional

import h5py
import imageio
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from vla_client import VLAClient  # noqa: E402

logger = logging.getLogger("libero10_kyungtae_eval")

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]  # open gripper, no arm motion
DEFAULT_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}

TASK_NAME_LEN_CAP = 120  # for the task directory
SHORT_NAME_LEN_CAP = 80  # for the per-episode filename


@dataclass
class EvalConfig:
    vla_url: str = "http://localhost:8700"
    suite: str = "libero_10"
    num_tasks: int = 0  # 0 = all tasks in suite
    num_trials: int = 20
    num_steps_wait: int = 10
    resolution: int = 256
    replan_steps: int = 5
    output_root: str = "/workspace/kyungtae_data"
    model_name: str = "openvla-oft"
    seed: int = 7
    max_steps_override: Optional[int] = None
    task_ids: Optional[List[int]] = None
    run_tag: str = ""
    gzip_level: int = 4
    fps: int = 10


@dataclass
class TaskResult:
    task_id: int
    task_short: str
    task_description: str
    successes: int = 0
    trials: int = 0
    latencies_ms: List[float] = field(default_factory=list)
    steps_success: List[int] = field(default_factory=list)
    steps_fail: List[int] = field(default_factory=list)


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


def _safe_name(s: str, cap: int) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", s)[:cap].strip("_")


def _build_states(obs: dict) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}

    def put(key: str, value):
        if value is None:
            return
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        out[f"observation.state.{key}"] = arr

    put("eef_pos", obs.get("robot0_eef_pos"))
    put("eef_quat", obs.get("robot0_eef_quat"))
    put("gripper_qpos", obs.get("robot0_gripper_qpos"))
    put("gripper_qvel", obs.get("robot0_gripper_qvel"))
    put("joint_pos", obs.get("robot0_joint_pos"))
    put("joint_vel", obs.get("robot0_joint_vel"))
    return out


def _build_images(obs: dict) -> Dict[str, np.ndarray]:
    images: Dict[str, np.ndarray] = {}
    if "agentview_image" in obs:
        images["static"] = np.ascontiguousarray(obs["agentview_image"])
    if "robot0_eye_in_hand_image" in obs:
        images["wrist"] = np.ascontiguousarray(obs["robot0_eye_in_hand_image"])
    return images


def _extract_traj_obs(obs: dict) -> Dict[str, np.ndarray]:
    """Subset of robosuite obs used for the trajectory hdf5."""
    return {
        "eef_pos": np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(3),
        "eef_quat": np.asarray(obs["robot0_eef_quat"], dtype=np.float32).reshape(4),
        "gripper_qpos": np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(2),
        "gripper_qvel": np.asarray(obs["robot0_gripper_qvel"], dtype=np.float32).reshape(2),
        "joint_pos": np.asarray(obs["robot0_joint_pos"], dtype=np.float32).reshape(7),
        "joint_vel": np.asarray(obs["robot0_joint_vel"], dtype=np.float32).reshape(7),
    }


def _assemble_action_from_subkeys(action_dict: Dict[str, np.ndarray]) -> np.ndarray:
    pos = action_dict.get("action.eef_pos")
    if pos is None:
        raise ValueError(f"Missing action.eef_pos in server response: keys={list(action_dict)}")
    rot = action_dict.get("action.eef_euler")
    if rot is None:
        rot = action_dict.get("action.eef_axis_angle")
    if rot is None:
        rot = np.zeros((pos.shape[0], 3), dtype=np.float32)
    grip = action_dict.get("action.gripper")
    if grip is None:
        grip = np.full((pos.shape[0], 1), -1.0, dtype=np.float32)
    n = pos.shape[0]
    assert rot.shape[0] == n and grip.shape[0] == n, (
        f"Action chunk length mismatch: pos={pos.shape} rot={rot.shape} grip={grip.shape}"
    )
    return np.concatenate([pos, rot, grip], axis=1).astype(np.float32)


def _flatten_action(action_or_dict) -> np.ndarray:
    if isinstance(action_or_dict, dict):
        return _assemble_action_from_subkeys(action_or_dict)
    arr = np.asarray(action_or_dict, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    return arr


def _utc_now_iso() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc_stamp() -> str:
    return _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")


# --------------------------------------------------------------------------- #
# HDF5 writer
# --------------------------------------------------------------------------- #
def write_episode_hdf5(
    path: pathlib.Path,
    *,
    actions: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    obs_seq: Dict[str, np.ndarray],
    sim_states: np.ndarray,
    init_state: np.ndarray,
    attrs: Dict[str, object],
    gzip_level: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _ds(parent, name, data):
        # gzip on tiny arrays adds overhead but keeps everything uniform.
        return parent.create_dataset(
            name,
            data=data,
            compression="gzip",
            compression_opts=gzip_level,
        )

    with h5py.File(str(path), "w") as f:
        _ds(f, "actions", actions.astype(np.float32))
        _ds(f, "rewards", rewards.astype(np.float32))
        _ds(f, "dones", dones.astype(np.uint8))
        _ds(f, "sim_states", sim_states.astype(np.float64))
        _ds(f, "init_state", init_state.astype(np.float64))

        obs_grp = f.create_group("obs")
        for k, v in obs_seq.items():
            _ds(obs_grp, k, v.astype(np.float32))

        for k, v in attrs.items():
            if isinstance(v, (dict, list)):
                f.attrs[k] = json.dumps(v)
            elif isinstance(v, bool):
                f.attrs[k] = bool(v)
            elif v is None:
                f.attrs[k] = "null"
            else:
                f.attrs[k] = v


# --------------------------------------------------------------------------- #
# Core eval loop
# --------------------------------------------------------------------------- #
def run_eval(cfg: EvalConfig) -> dict:
    # Lazy imports — these fail outside the libero container.
    from libero.libero import benchmark as _benchmark
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    # --- Connect to VLA server -------------------------------------------- #
    client = VLAClient(cfg.vla_url, timeout=180.0)
    logger.info("Waiting for VLA server at %s ...", cfg.vla_url)
    server_info = client.wait_until_ready(max_wait=300.0, poll_interval=3.0)
    logger.info("Server ready: %s", server_info)

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
    logger.info("Suite %s: running task ids %s (out of %d)", cfg.suite, task_id_list, n_suite)

    # max_steps
    max_steps = cfg.max_steps_override
    if max_steps is None:
        for k, v in DEFAULT_MAX_STEPS.items():
            if cfg.suite.startswith(k):
                max_steps = v
                break
    if max_steps is None:
        max_steps = 300

    # --- Run dir setup ---------------------------------------------------- #
    run_name = f"run_{_utc_stamp()}"
    if cfg.run_tag:
        run_name = f"{run_name}_{_safe_name(cfg.run_tag, 40)}"
    run_dir = pathlib.Path(cfg.output_root) / cfg.model_name / cfg.suite / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # File log
    file_handler = logging.FileHandler(str(run_dir / "run.log"))
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)
    logger.info("Run dir: %s", run_dir)

    started_at = _utc_now_iso()
    started_t0 = time.time()

    # config.json
    cfg_dict = asdict(cfg)
    cfg_dict["max_steps_resolved"] = max_steps
    cfg_dict["server_info"] = server_info
    cfg_dict["task_id_list"] = task_id_list
    cfg_dict["started_at_utc"] = started_at
    with open(run_dir / "config.json", "w") as f:
        json.dump(cfg_dict, f, indent=2, default=str)

    # CSV (header now, append per episode)
    csv_path = run_dir / "summary.csv"
    csv_header = [
        "task_id", "task_short", "trial_idx", "success", "num_steps",
        "init_state_index", "init_seed", "avg_latency_ms", "num_predictions",
        "video_path", "hdf5_path", "started_at", "finished_at", "duration_sec",
    ]
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(csv_header)

    np.random.seed(cfg.seed)

    results: List[TaskResult] = []
    total_episodes = 0
    total_successes = 0

    # --- Task loop -------------------------------------------------------- #
    for task_id in task_id_list:
        task = suite.get_task(task_id)
        bddl_name = task.bddl_file
        bddl_stem = _safe_name(pathlib.Path(bddl_name).stem, TASK_NAME_LEN_CAP)
        bddl_path = str(
            pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / bddl_name
        )
        task_description = _read_task_instruction_from_bddl(bddl_path, task.language)

        task_dir_name = f"task{task_id:02d}_{bddl_stem}"
        task_dir = run_dir / task_dir_name
        videos_dir = task_dir / "videos"
        traj_dir = task_dir / "trajectories"
        videos_dir.mkdir(parents=True, exist_ok=True)
        traj_dir.mkdir(parents=True, exist_ok=True)

        task_short_for_file = _safe_name(bddl_stem, SHORT_NAME_LEN_CAP)
        logger.info("Task %d: %s  (%s)", task_id, task_description, bddl_stem)

        init_states = suite.get_task_init_states(task_id)
        logger.info(
            "  init_states available=%d, using indices 0..%d (mod %d)",
            len(init_states), cfg.num_trials - 1, len(init_states),
        )
        tr = TaskResult(task_id=task_id, task_short=bddl_stem, task_description=task_description)

        for trial_idx in range(cfg.num_trials):
            ep_started = _utc_now_iso()
            ep_t0 = time.time()
            init_state_idx = trial_idx % len(init_states)
            init = init_states[init_state_idx]
            init_flat = np.asarray(init, dtype=np.float64).reshape(-1)

            env = OffScreenRenderEnv(
                bddl_file_name=bddl_path,
                camera_heights=cfg.resolution,
                camera_widths=cfg.resolution,
            )
            try:
                env.seed(cfg.seed)
                env.reset()
                obs = env.set_init_state(init)

                client.reset()
                action_plan: Deque[np.ndarray] = collections.deque()
                replay: List[np.ndarray] = []

                obs_seq_lists: Dict[str, List[np.ndarray]] = {
                    k: [] for k in (
                        "eef_pos", "eef_quat", "gripper_qpos",
                        "gripper_qvel", "joint_pos", "joint_vel",
                    )
                }
                actions_list: List[np.ndarray] = []
                rewards_list: List[float] = []
                dones_list: List[int] = []
                sim_states_list: List[np.ndarray] = []
                latencies: List[float] = []

                # No-op settle steps (not recorded — keeps replay deterministic with same wait)
                for _ in range(cfg.num_steps_wait):
                    obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

                done = False
                steps_executed = 0
                while steps_executed < max_steps:
                    if not action_plan:
                        images = _build_images(obs)
                        states = _build_states(obs)
                        action_or_dict, latency_ms = client.predict(
                            images, states, task_description
                        )
                        latencies.append(latency_ms)
                        chunk = _flatten_action(action_or_dict)
                        n_use = min(len(chunk), max(1, cfg.replan_steps))
                        for i in range(n_use):
                            action_plan.append(chunk[i])

                    # Snapshot obs *before* the step (state at which we decided action)
                    traj_obs = _extract_traj_obs(obs)
                    sim_state_t = np.asarray(env.get_sim_state(), dtype=np.float64).reshape(-1)

                    action = action_plan.popleft()
                    obs, reward, done, info = env.step(action.tolist())

                    # Log this transition
                    actions_list.append(np.asarray(action, dtype=np.float32).reshape(-1))
                    rewards_list.append(float(reward))
                    dones_list.append(1 if done else 0)
                    sim_states_list.append(sim_state_t)
                    for k, v in traj_obs.items():
                        obs_seq_lists[k].append(v)

                    replay.append(obs["agentview_image"][::-1])
                    steps_executed += 1
                    if done:
                        break

                tr.trials += 1
                if done:
                    tr.successes += 1
                    total_successes += 1
                    tr.steps_success.append(steps_executed)
                else:
                    tr.steps_fail.append(steps_executed)
                tr.latencies_ms.extend(latencies)
                total_episodes += 1

                avg_lat = float(np.mean(latencies)) if latencies else 0.0
                logger.info(
                    "  trial %02d: %s steps=%d lat_avg=%.1fms",
                    trial_idx,
                    "SUCCESS" if done else "fail",
                    steps_executed,
                    avg_lat,
                )

                # ---- finalize file paths now that success is known ------ #
                status = "success" if done else "fail"
                fname_base = f"{status}_{trial_idx:02d}_{task_short_for_file}"
                video_path = videos_dir / f"{fname_base}.mp4"
                hdf5_path = traj_dir / f"{fname_base}.hdf5"

                # ---- write video --------------------------------------- #
                if replay:
                    imageio.mimwrite(str(video_path), replay, fps=cfg.fps)

                # ---- write hdf5 ---------------------------------------- #
                actions_arr = np.stack(actions_list, axis=0) if actions_list else np.zeros((0, 7), dtype=np.float32)
                rewards_arr = np.asarray(rewards_list, dtype=np.float32)
                dones_arr = np.asarray(dones_list, dtype=np.uint8)
                obs_seq = {k: (np.stack(v, axis=0) if v else np.zeros((0,), dtype=np.float32))
                           for k, v in obs_seq_lists.items()}
                sim_states_arr = (
                    np.stack(sim_states_list, axis=0)
                    if sim_states_list else np.zeros((0,), dtype=np.float64)
                )

                ep_finished = _utc_now_iso()
                duration = time.time() - ep_t0
                attrs = {
                    "task_id": int(task_id),
                    "task_short": bddl_stem,
                    "task_description": task_description,
                    "task_bddl_file": bddl_name,
                    "trial_idx": int(trial_idx),
                    "success": bool(done),
                    "num_steps": int(steps_executed),
                    "num_steps_wait": int(cfg.num_steps_wait),
                    "init_seed": int(cfg.seed),
                    "init_state_index": int(init_state_idx),
                    "replan_steps": int(cfg.replan_steps),
                    "model_name": cfg.model_name,
                    "vla_url": cfg.vla_url,
                    "server_info_json": server_info,
                    "suite": cfg.suite,
                    "resolution": int(cfg.resolution),
                    "fps": int(cfg.fps),
                    "max_steps": int(max_steps),
                    "num_predictions": int(len(latencies)),
                    "avg_latency_ms": float(avg_lat),
                    "started_at_utc": ep_started,
                    "finished_at_utc": ep_finished,
                    "duration_sec": float(duration),
                    "action_layout": "OSC_POSE [eef_pos(3), eef_euler(3), gripper(1)]",
                    "sim_states_aligned_to": "decision-time, post-settle (after num_steps_wait)",
                }
                write_episode_hdf5(
                    hdf5_path,
                    actions=actions_arr,
                    rewards=rewards_arr,
                    dones=dones_arr,
                    obs_seq=obs_seq,
                    sim_states=sim_states_arr,
                    init_state=init_flat,
                    attrs=attrs,
                    gzip_level=cfg.gzip_level,
                )

                # ---- csv append ---------------------------------------- #
                with open(csv_path, "a", newline="") as f:
                    csv.writer(f).writerow([
                        task_id,
                        bddl_stem,
                        trial_idx,
                        int(bool(done)),
                        steps_executed,
                        init_state_idx,
                        cfg.seed,
                        f"{avg_lat:.3f}",
                        len(latencies),
                        str(video_path.relative_to(run_dir)),
                        str(hdf5_path.relative_to(run_dir)),
                        ep_started,
                        ep_finished,
                        f"{duration:.3f}",
                    ])
            finally:
                env.close()

        results.append(tr)

    # --- Final summary --------------------------------------------------- #
    finished_at = _utc_now_iso()
    total_duration = time.time() - started_t0
    all_latencies = [x for r in results for x in r.latencies_ms]
    summary = {
        "suite": cfg.suite,
        "model_name": cfg.model_name,
        "vla_url": cfg.vla_url,
        "server_info": server_info,
        "run_dir": str(run_dir),
        "num_tasks": len(task_id_list),
        "task_ids": task_id_list,
        "num_trials_per_task": cfg.num_trials,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "success_rate": (total_successes / total_episodes) if total_episodes else 0.0,
        "avg_latency_ms_overall": float(np.mean(all_latencies)) if all_latencies else 0.0,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_sec": total_duration,
        "tasks": [
            {
                "task_id": r.task_id,
                "task_short": r.task_short,
                "description": r.task_description,
                "successes": r.successes,
                "trials": r.trials,
                "success_rate": (r.successes / r.trials) if r.trials else 0.0,
                "avg_latency_ms": float(np.mean(r.latencies_ms)) if r.latencies_ms else 0.0,
                "num_predictions": len(r.latencies_ms),
                "avg_steps_success": float(np.mean(r.steps_success)) if r.steps_success else 0.0,
                "avg_steps_fail": float(np.mean(r.steps_fail)) if r.steps_fail else 0.0,
            }
            for r in results
        ],
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("=" * 60)
    logger.info(
        "DONE  suite=%s model=%s success=%d/%d (%.1f%%)  duration=%.1fs",
        cfg.suite,
        cfg.model_name,
        total_successes,
        total_episodes,
        100.0 * summary["success_rate"],
        total_duration,
    )
    logger.info("Results: %s", run_dir)
    return summary


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vla-url", default=os.environ.get("VLA_SERVER_URL", "http://localhost:8700"))
    p.add_argument("--suite", default="libero_10")
    p.add_argument("--num-tasks", type=int, default=0, help="0 = run all tasks in the suite")
    p.add_argument("--num-trials", type=int, default=20)
    p.add_argument("--num-steps-wait", type=int, default=10)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--replan-steps", type=int, default=5)
    p.add_argument(
        "--output-root",
        default=os.environ.get("KYUNGTAE_OUTPUT_ROOT", "/workspace/kyungtae_data"),
    )
    p.add_argument("--model-name", default="openvla-oft")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max-steps", type=int, default=None, dest="max_steps_override")
    p.add_argument("--task-ids", default="", help="comma-separated explicit task ids")
    p.add_argument("--run-tag", default="", help="suffix appended to run_<ts>")
    p.add_argument("--gzip-level", type=int, default=4)
    p.add_argument("--fps", type=int, default=10)
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
        output_root=args.output_root,
        model_name=args.model_name,
        seed=args.seed,
        max_steps_override=args.max_steps_override,
        task_ids=task_ids,
        run_tag=args.run_tag,
        gzip_level=args.gzip_level,
        fps=args.fps,
    )
    run_eval(cfg)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()
