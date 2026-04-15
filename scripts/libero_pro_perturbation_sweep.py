#!/usr/bin/env python3
"""LIBERO-pro perturbation sweep against a single base task.

Picks one libero_spatial base task (the 'akita black bowl between plate and
ramekin' tabletop scene) and applies every LIBERO-pro perturbation type to it
with multiple seeds, deduplicates identical outputs, evaluates each unique
variant against a running pi0.5 VLA server over the unified HTTP protocol, and
writes rich per-episode metadata to CSV plus a rollout video per run.

Design
------
- Four perturbation types are swept: swap, object, language, task.
  (environment is skipped: LIBERO-pro's EnvironmentReplacePerturbator is
  hardcoded to "living_room_table" and the upstream BDDL files are unreleased
  — see README.md and LIBERO_PRO_PLAN.md.)
- For each type we try a list of candidate seeds (1..10, plus None/0 for the
  base-file call for SwapPerturbator which ignores seed), dedupe perturbed BDDL
  strings by SHA-1, and keep up to `--variants-per-type` unique ones.
- For each unique variant we run ONE rollout with a fresh environment, read
  the task language from the perturbed BDDL directly (issue #14 workaround),
  and record everything we can measure: success, step count, latency stats,
  objects/regions that changed, the full unified diff as a short fingerprint.

Output layout (under --output-dir, default test_outputs/perturbation_sweep/):
    <timestamp>/
        videos/
            <run_id>.mp4                    # agentview_image rollout
        bddls/
            <run_id>.bddl                   # the exact perturbed bddl used
        diffs/
            <run_id>.diff                   # unified diff vs. the base bddl
        video_index.csv                     # filename → perturbation metadata
        perturbation_sweep_details.csv      # full per-episode record
        run_meta.json                       # global run metadata

Runs inside the `bigenlight/libero-pro` container. The pi0.5 server must be
reachable at --vla-url (default http://localhost:8400).
"""

from __future__ import annotations

import argparse
import collections
import csv
import dataclasses
import difflib
import hashlib
import json
import logging
import os
import pathlib
import re
import sys
import time
import traceback
from typing import Deque, Dict, List, Optional, Tuple

import imageio
import numpy as np

# ---- path setup ----------------------------------------------------------- #
# vla_client.py lives next to this script.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from vla_client import VLAClient  # noqa: E402

# LIBERO-pro's perturbation engine is at /workspace/LIBERO-PRO inside the image.
LIBERO_PRO_ROOT = "/workspace/LIBERO-PRO"
sys.path.insert(0, LIBERO_PRO_ROOT)

logger = logging.getLogger("libero_pro_perturbation_sweep")

# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #
SUITE = "libero_spatial"
BASE_TASK_NAME = (
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"
)
OOD_YAML_DIR = f"{LIBERO_PRO_ROOT}/libero_ood"
PERTURBATION_CONFIGS = {
    "swap": f"{OOD_YAML_DIR}/ood_spatial_relation.yaml",
    "object": f"{OOD_YAML_DIR}/ood_object.yaml",
    "language": f"{OOD_YAML_DIR}/ood_language.yaml",
    "task": f"{OOD_YAML_DIR}/ood_task.yaml",
}
CANDIDATE_SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
MAX_STEPS_DEFAULT = 220  # libero_spatial longest demo ≈ 193 steps
NUM_STEPS_WAIT = 10


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class VariantResult:
    run_id: str
    perturbation_type: str
    variant_index: int
    seed: Optional[int]
    base_task_name: str
    suite: str
    scene_name: str
    camera_view: str
    init_state_index: int
    bddl_config_path: str
    # Languages / goals
    base_language: str
    perturbed_language: str
    base_goal: str
    perturbed_goal: str
    # Change summary
    perturbation_summary: str
    perturbation_intensity: str
    n_changed_lines: int
    n_diff_additions: int
    n_diff_removals: int
    diff_fingerprint: str
    added_objects: List[str]
    removed_objects: List[str]
    added_regions: List[str]
    removed_regions: List[str]
    base_objects: List[str]
    perturbed_objects: List[str]
    base_regions: List[str]
    perturbed_regions: List[str]
    perturbed_bddl_sha1: str
    bddl_path: str
    diff_path: str
    video_path: str
    # Rollout metrics
    success: bool
    termination_reason: str
    n_env_steps: int
    max_env_steps: int
    n_predictions: int
    latency_avg_ms: float
    latency_min_ms: float
    latency_max_ms: float
    # Runtime / error
    error: str
    timestamp: str
    timestamp_unix: float
    elapsed_sec: float
    # Server info
    server_model: str
    server_action_type: str
    server_n_action_steps: int


# --------------------------------------------------------------------------- #
# BDDL helpers
# --------------------------------------------------------------------------- #
_LANG_RE = re.compile(r"\(:language\s*(.*?)\)", re.S)
_OBJECTS_BLOCK_RE = re.compile(r"\(:objects\s+(.*?)\)\s*\n", re.S)
_REGIONS_BLOCK_RE = re.compile(r"\(:regions(.*?)\)\s*\n\s*\)\s*\n", re.S)
_GOAL_BLOCK_RE = re.compile(r"\(:goal\s*(.*?)\s*\)\s*\n", re.S)
_SCENE_RE = re.compile(r"\(:target\s+([a-zA-Z_][a-zA-Z0-9_]*)")


def read_language(bddl: str) -> str:
    m = _LANG_RE.search(bddl)
    return m.group(1).strip().strip('"').strip("'") if m else ""


def read_objects(bddl: str) -> List[str]:
    m = _OBJECTS_BLOCK_RE.search(bddl)
    if not m:
        return []
    tokens = m.group(1).replace("-", " ").split()
    # Strip punctuation, take only symbol-like tokens.
    return sorted({t for t in tokens if re.match(r"^[a-zA-Z0-9_]+$", t)})


def read_regions(bddl: str) -> List[str]:
    m = _REGIONS_BLOCK_RE.search(bddl)
    if not m:
        return []
    body = m.group(1)
    names = re.findall(r"\(([a-zA-Z_][a-zA-Z0-9_]*)\s*\n\s*\(:target", body)
    return sorted(set(names))


def read_goal(bddl: str) -> str:
    m = _GOAL_BLOCK_RE.search(bddl)
    return m.group(1).strip() if m else ""


def read_scene(bddl: str) -> str:
    """Return the environment fixture name (e.g. 'main_table')."""
    m = _SCENE_RE.search(bddl)
    return m.group(1) if m else ""


def intensity_label(n_changed: int) -> str:
    if n_changed <= 0:
        return "none"
    if n_changed <= 2:
        return "low"
    if n_changed <= 5:
        return "medium"
    return "high"


def perturbation_summary(
    added_obj: List[str],
    removed_obj: List[str],
    added_reg: List[str],
    removed_reg: List[str],
    base_language: str,
    perturbed_language: str,
    base_goal: str,
    perturbed_goal: str,
) -> str:
    parts = []
    if removed_obj or added_obj:
        parts.append(f"objects:[-{','.join(removed_obj) or '_'}|+{','.join(added_obj) or '_'}]")
    if removed_reg or added_reg:
        parts.append(f"regions:[-{','.join(removed_reg) or '_'}|+{','.join(added_reg) or '_'}]")
    if base_language != perturbed_language:
        parts.append(f"language:changed")
    if base_goal != perturbed_goal:
        parts.append(f"goal:changed")
    return " ".join(parts) if parts else "no-op"


def sha1_of(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def unified_diff(base: str, perturbed: str, label_a: str, label_b: str) -> str:
    return "".join(
        difflib.unified_diff(
            base.splitlines(keepends=True),
            perturbed.splitlines(keepends=True),
            fromfile=label_a,
            tofile=label_b,
            n=2,
        )
    )


def diff_fingerprint(diff: str) -> str:
    """Short human-readable summary of a unified diff.

    Counts the `+`/`-` hunks and lists the first few changed tokens so the
    CSV reader has an at-a-glance idea of what actually changed.
    """
    plus = [l[1:].strip() for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++")]
    minus = [l[1:].strip() for l in diff.splitlines() if l.startswith("-") and not l.startswith("---")]
    # Dedup and cap.
    def cap(lst, n=3):
        seen = []
        for x in lst:
            if x and x not in seen:
                seen.append(x)
            if len(seen) >= n:
                break
        return seen
    return f"+{len(plus)}/-{len(minus)}  +[{' | '.join(cap(plus))}]  -[{' | '.join(cap(minus))}]"


# --------------------------------------------------------------------------- #
# Perturbation engine (uses LIBERO-pro's perturbation.py)
# --------------------------------------------------------------------------- #
def make_perturbator(ptype: str, base_bddl: str):
    from perturbation import (
        BDDLParser,
        SwapPerturbator,
        ObjectReplacePerturbator,
        LanguagePerturbator,
        TaskPerturbator,
    )
    parser = BDDLParser(base_bddl)
    cls = {
        "swap": SwapPerturbator,
        "object": ObjectReplacePerturbator,
        "language": LanguagePerturbator,
        "task": TaskPerturbator,
    }[ptype]
    return cls(parser, PERTURBATION_CONFIGS[ptype])


def apply_perturbation(ptype: str, base_bddl: str, suite: str, task_name: str, seed: Optional[int]) -> str:
    p = make_perturbator(ptype, base_bddl)
    if ptype == "swap":
        # Swap signature has no seed kwarg.
        return p.perturb(suite, task_name)
    return p.perturb(suite, task_name, seed=seed)


def discover_unique_variants(
    ptype: str,
    base_bddl: str,
    suite: str,
    task_name: str,
    wanted: int,
    seeds: List[int],
) -> List[Tuple[Optional[int], str]]:
    """Return up to `wanted` unique (seed, perturbed_bddl) pairs."""
    if ptype == "swap":
        try:
            out = apply_perturbation(ptype, base_bddl, suite, task_name, seed=None)
            return [(None, out)]
        except Exception as e:  # noqa: BLE001
            logger.warning("swap failed: %s", e)
            return []

    seen_hashes: set = set()
    variants: List[Tuple[int, str]] = []
    for s in seeds:
        try:
            out = apply_perturbation(ptype, base_bddl, suite, task_name, seed=s)
        except Exception as e:  # noqa: BLE001
            logger.warning("%s seed=%d failed: %s", ptype, s, e)
            continue
        h = sha1_of(out)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        variants.append((s, out))
        if len(variants) >= wanted:
            break
    return variants


# --------------------------------------------------------------------------- #
# Rollout
# --------------------------------------------------------------------------- #
def _build_states(obs: dict) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    def put(key, value):
        if value is None:
            return
        out[f"observation.state.{key}"] = np.asarray(value, dtype=np.float32).reshape(-1)
    put("eef_pos", obs.get("robot0_eef_pos"))
    put("eef_quat", obs.get("robot0_eef_quat"))
    put("gripper_qpos", obs.get("robot0_gripper_qpos"))
    put("joint_pos", obs.get("robot0_joint_pos"))
    return out


def _build_images(obs: dict) -> Dict[str, np.ndarray]:
    images: Dict[str, np.ndarray] = {}
    if "agentview_image" in obs:
        images["static"] = np.ascontiguousarray(obs["agentview_image"])
    if "robot0_eye_in_hand_image" in obs:
        images["wrist"] = np.ascontiguousarray(obs["robot0_eye_in_hand_image"])
    return images


def _assemble_action(action_or_dict) -> np.ndarray:
    if isinstance(action_or_dict, dict):
        pos = action_or_dict["action.eef_pos"]
        rot = action_or_dict.get("action.eef_euler")
        if rot is None:
            rot = np.zeros((pos.shape[0], 3), dtype=np.float32)
        grip = action_or_dict.get("action.gripper")
        if grip is None:
            grip = np.full((pos.shape[0], 1), -1.0, dtype=np.float32)
        return np.concatenate([pos, rot, grip], axis=1).astype(np.float32)
    arr = np.asarray(action_or_dict, dtype=np.float32)
    return arr if arr.ndim == 2 else arr[np.newaxis, :]


def run_rollout(
    client: VLAClient,
    bddl_path: str,
    instruction: str,
    init_state: Optional[np.ndarray],
    resolution: int,
    max_steps: int,
    replan_steps: int,
    seed: int,
) -> Tuple[bool, str, int, List[np.ndarray], List[float]]:
    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    replay: List[np.ndarray] = []
    latencies: List[float] = []
    termination = "max_steps"
    done = False
    steps_executed = 0
    try:
        env.seed(seed)
        env.reset()
        if init_state is not None:
            try:
                obs = env.set_init_state(init_state)
            except Exception as e:  # noqa: BLE001
                logger.warning("set_init_state failed (%s) — falling back to env.reset obs", e)
                obs = env.reset()
                termination = "init_state_error"
        else:
            obs = env.reset()

        client.reset()
        for _ in range(NUM_STEPS_WAIT):
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

        action_plan: Deque[np.ndarray] = collections.deque()
        while steps_executed < max_steps:
            if not action_plan:
                images = _build_images(obs)
                states = _build_states(obs)
                action_or_dict, latency_ms = client.predict(images, states, instruction)
                latencies.append(latency_ms)
                chunk = _assemble_action(action_or_dict)
                for i in range(min(len(chunk), max(1, replan_steps))):
                    action_plan.append(chunk[i])

            action = action_plan.popleft()
            obs, reward, done, info = env.step(action.tolist())
            replay.append(obs["agentview_image"][::-1])
            steps_executed += 1
            if done:
                termination = "env_done_success"
                break
    except Exception as e:  # noqa: BLE001
        termination = f"exception:{type(e).__name__}"
        logger.error("rollout error: %s\n%s", e, traceback.format_exc())
    finally:
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass
    return done, termination, steps_executed, replay, latencies


# --------------------------------------------------------------------------- #
# Main sweep
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vla-url", default=os.environ.get("VLA_SERVER_URL", "http://localhost:8400"))
    p.add_argument("--suite", default=SUITE)
    p.add_argument("--base-task", default=BASE_TASK_NAME)
    p.add_argument("--variants-per-type", type=int, default=4)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--max-steps", type=int, default=MAX_STEPS_DEFAULT)
    p.add_argument("--replan-steps", type=int, default=5)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--output-dir",
        default="/workspace/LIBERO-PRO/test_outputs/perturbation_sweep",
    )
    p.add_argument(
        "--skip-types",
        default="",
        help="comma-separated list of perturbation types to skip (e.g. 'task')",
    )
    args = p.parse_args()

    # Include millisecond suffix so rapid reruns don't collide.
    run_dir = pathlib.Path(args.output_dir) / (
        time.strftime("%Y%m%d_%H%M%S_") + f"{int(time.time() * 1000) % 1000:03d}"
    )
    (run_dir / "videos").mkdir(parents=True, exist_ok=True)
    (run_dir / "bddls").mkdir(parents=True, exist_ok=True)
    (run_dir / "diffs").mkdir(parents=True, exist_ok=True)

    # --- Load base task ------------------------------------------------------ #
    from libero.libero import benchmark as _benchmark
    from libero.libero import get_libero_path

    bd = _benchmark.get_benchmark_dict()
    suite_obj = bd[args.suite]()
    names = suite_obj.get_task_names()
    try:
        base_task_id = names.index(args.base_task)
    except ValueError:
        raise SystemExit(
            f"Base task {args.base_task!r} not found in {args.suite}. Available: {names}"
        )
    task = suite_obj.get_task(base_task_id)
    base_bddl_path = str(
        pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    )
    with open(base_bddl_path, "r") as f:
        base_bddl = f.read()
    base_language = read_language(base_bddl)
    base_objects = read_objects(base_bddl)
    base_regions = read_regions(base_bddl)
    base_goal = read_goal(base_bddl)
    scene_name = read_scene(base_bddl) or "main_table"
    logger.info("Base task: %s  (scene=%s)", args.base_task, scene_name)
    logger.info("Base language: %s", base_language)

    init_states = suite_obj.get_task_init_states(base_task_id)
    logger.info("Available init states: %d", len(init_states))

    # --- Connect to VLA server ---------------------------------------------- #
    client = VLAClient(args.vla_url, timeout=120.0)
    logger.info("Waiting for VLA server at %s ...", args.vla_url)
    server_info = client.wait_until_ready(max_wait=300.0, poll_interval=3.0)
    logger.info("Server: %s", server_info)

    skip_types = {t.strip() for t in args.skip_types.split(",") if t.strip()}
    types_to_run = [t for t in ["swap", "object", "language", "task"] if t not in skip_types]

    # --- Discover unique variants ------------------------------------------- #
    per_type_variants: Dict[str, List[Tuple[Optional[int], str]]] = {}
    for ptype in types_to_run:
        logger.info("Discovering %s variants ...", ptype)
        variants = discover_unique_variants(
            ptype=ptype,
            base_bddl=base_bddl,
            suite=args.suite,
            task_name=args.base_task,
            wanted=args.variants_per_type,
            seeds=CANDIDATE_SEEDS,
        )
        logger.info("  %s: %d unique variant(s)", ptype, len(variants))
        per_type_variants[ptype] = variants

    # --- Execute ------------------------------------------------------------- #
    results: List[VariantResult] = []

    for ptype in types_to_run:
        for idx, (seed, perturbed_bddl) in enumerate(per_type_variants[ptype], start=1):
            run_id = f"{ptype}_v{idx}_s{seed if seed is not None else 'x'}"
            logger.info("=== %s ===", run_id)
            t0 = time.time()

            bddl_out_path = run_dir / "bddls" / f"{run_id}.bddl"
            diff_out_path = run_dir / "diffs" / f"{run_id}.diff"
            video_out_path = run_dir / "videos" / f"{run_id}.mp4"

            bddl_out_path.write_text(perturbed_bddl)
            diff_text = unified_diff(base_bddl, perturbed_bddl, "base.bddl", f"{run_id}.bddl")
            diff_out_path.write_text(diff_text)

            perturbed_language = read_language(perturbed_bddl)
            perturbed_objects = read_objects(perturbed_bddl)
            perturbed_regions = read_regions(perturbed_bddl)
            perturbed_goal = read_goal(perturbed_bddl)
            added_obj = sorted(set(perturbed_objects) - set(base_objects))
            removed_obj = sorted(set(base_objects) - set(perturbed_objects))
            added_reg = sorted(set(perturbed_regions) - set(base_regions))
            removed_reg = sorted(set(base_regions) - set(perturbed_regions))
            # Header markers `---`/`+++` don't count as changes.
            n_additions = sum(
                1 for l in diff_text.splitlines() if l.startswith("+") and not l.startswith("+++")
            )
            n_removals = sum(
                1 for l in diff_text.splitlines() if l.startswith("-") and not l.startswith("---")
            )
            n_changed_lines = n_additions + n_removals
            fp = diff_fingerprint(diff_text)
            summary = perturbation_summary(
                added_obj, removed_obj, added_reg, removed_reg,
                base_language, perturbed_language, base_goal, perturbed_goal,
            )
            intensity = intensity_label(n_changed_lines)

            # Rollout
            init_state = init_states[0] if len(init_states) else None
            success = False
            termination = "not_run"
            n_steps = 0
            replay: List[np.ndarray] = []
            latencies: List[float] = []
            error = ""
            try:
                success, termination, n_steps, replay, latencies = run_rollout(
                    client=client,
                    bddl_path=str(bddl_out_path),
                    instruction=perturbed_language or base_language,
                    init_state=init_state,
                    resolution=args.resolution,
                    max_steps=args.max_steps,
                    replan_steps=args.replan_steps,
                    seed=args.seed,
                )
            except Exception as e:  # noqa: BLE001
                error = f"{type(e).__name__}: {e}"
                logger.error("rollout error: %s", error)

            if replay:
                try:
                    imageio.mimwrite(str(video_out_path), replay, fps=10)
                except Exception as e:  # noqa: BLE001
                    logger.warning("video write failed: %s", e)

            result = VariantResult(
                run_id=run_id,
                perturbation_type=ptype,
                variant_index=idx,
                seed=seed,
                base_task_name=args.base_task,
                suite=args.suite,
                scene_name=scene_name,
                camera_view="agentview_image",
                init_state_index=0,
                bddl_config_path=PERTURBATION_CONFIGS[ptype],
                base_language=base_language,
                perturbed_language=perturbed_language,
                base_goal=base_goal,
                perturbed_goal=perturbed_goal,
                perturbation_summary=summary,
                perturbation_intensity=intensity,
                n_changed_lines=n_changed_lines,
                n_diff_additions=n_additions,
                n_diff_removals=n_removals,
                diff_fingerprint=fp,
                added_objects=added_obj,
                removed_objects=removed_obj,
                added_regions=added_reg,
                removed_regions=removed_reg,
                base_objects=base_objects,
                perturbed_objects=perturbed_objects,
                base_regions=base_regions,
                perturbed_regions=perturbed_regions,
                perturbed_bddl_sha1=sha1_of(perturbed_bddl),
                bddl_path=str(bddl_out_path.relative_to(run_dir)),
                diff_path=str(diff_out_path.relative_to(run_dir)),
                video_path=str(video_out_path.relative_to(run_dir)) if replay else "",
                success=success,
                termination_reason=termination,
                n_env_steps=n_steps,
                max_env_steps=args.max_steps,
                n_predictions=len(latencies),
                latency_avg_ms=float(np.mean(latencies)) if latencies else 0.0,
                latency_min_ms=float(np.min(latencies)) if latencies else 0.0,
                latency_max_ms=float(np.max(latencies)) if latencies else 0.0,
                error=error,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                timestamp_unix=round(t0, 3),
                elapsed_sec=round(time.time() - t0, 2),
                server_model=str(server_info.get("model", "")),
                server_action_type=str(server_info.get("action_type", "")),
                server_n_action_steps=int(server_info.get("n_action_steps", 1)),
            )
            results.append(result)

    # --- Write CSVs + meta --------------------------------------------------- #
    detail_csv = run_dir / "perturbation_sweep_details.csv"
    video_csv = run_dir / "video_index.csv"

    field_names = [f.name for f in dataclasses.fields(VariantResult)]
    with open(detail_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writeheader()
        for r in results:
            row = dataclasses.asdict(r)
            for k, v in list(row.items()):
                if isinstance(v, list):
                    # JSON-encode list columns so that any delimiter/comma inside
                    # a token is safe and the CSV stays parseable.
                    row[k] = json.dumps(v, ensure_ascii=False)
            writer.writerow(row)

    with open(video_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "run_id",
                "video_filename",
                "perturbation_type",
                "variant_index",
                "seed",
                "success",
                "perturbation_intensity",
                "perturbation_summary",
                "perturbed_language",
                "n_env_steps",
                "termination_reason",
                "diff_fingerprint",
            ]
        )
        for r in results:
            w.writerow(
                [
                    r.run_id,
                    pathlib.Path(r.video_path).name if r.video_path else "",
                    r.perturbation_type,
                    r.variant_index,
                    r.seed if r.seed is not None else "",
                    "success" if r.success else "fail",
                    r.perturbation_intensity,
                    r.perturbation_summary,
                    r.perturbed_language,
                    r.n_env_steps,
                    r.termination_reason,
                    r.diff_fingerprint,
                ]
            )

    run_meta = {
        "suite": args.suite,
        "base_task": args.base_task,
        "base_language": base_language,
        "base_bddl_sha1": sha1_of(base_bddl),
        "variants_per_type": args.variants_per_type,
        "candidate_seeds": CANDIDATE_SEEDS,
        "skip_types": sorted(skip_types),
        "types_run": types_to_run,
        "per_type_variant_count": {k: len(v) for k, v in per_type_variants.items()},
        "vla_url": args.vla_url,
        "server_info": server_info,
        "total_runs": len(results),
        "successes": sum(1 for r in results if r.success),
    }
    (run_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2, default=str))

    logger.info("=" * 60)
    logger.info("Sweep complete. %d/%d successes.", run_meta["successes"], run_meta["total_runs"])
    logger.info("Run dir: %s", run_dir)
    logger.info("Video index CSV: %s", video_csv)
    logger.info("Detail CSV: %s", detail_csv)

    # Echo paths to stdout for the host-side collator to pick up.
    print(f"RUN_DIR={run_dir}")
    print(f"VIDEO_CSV={video_csv}")
    print(f"DETAIL_CSV={detail_csv}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()
