#!/usr/bin/env python3
"""pi05_libero × LIBERO-pro `libero_10_swap` 3-task × 3-trial evaluator.

Loads pre-generated swap BDDL + init files; does NOT call SwapPerturbator at
runtime.  The three target tasks are:
  1. KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it
  2. LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate
  3. STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy

For each task the driver:
  - Reads the pre-generated perturbed (swap) and base BDDL files from disk.
  - Builds a per-task output directory with perturbed.bddl, base.bddl,
    diff_vs_base.diff, and perturbation_info.json.
  - Runs `--num-trials` rollouts using different init-state indices.
  - Saves each rollout video prefixed with success_ or fail_.
  - Writes a top-level summary.csv and run_meta.json.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import logging
import os
import pathlib
import re
import shutil
import sys
import time
import traceback
from typing import Deque, Dict, List, Optional, Tuple

import imageio
import numpy as np
import torch

# ---- path setup ------------------------------------------------------------ #
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from vla_client import VLAClient  # noqa: E402
from libero_pro_perturbation_sweep import (  # noqa: E402
    read_language,
    read_objects,
    read_regions,
    read_goal,
    read_scene,
    unified_diff,
    diff_fingerprint,
    sha1_of,
    intensity_label,
    perturbation_summary,
    _build_states,
    _build_images,
    _assemble_action,
    run_rollout,
    NUM_STEPS_WAIT,
    LIBERO_DUMMY_ACTION,
)

logger = logging.getLogger("libero10_swap_3task_eval")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
LIBERO_PRO_ROOT = "/workspace/LIBERO-PRO"
BASE_SUITE_DIR = "libero_10"
SWAP_SUITE_DIR = "libero_10_swap"
BDDL_BASE_DIR = "/workspace/LIBERO-PRO/libero/libero/bddl_files"
INIT_BASE_DIR = "/workspace/LIBERO-PRO/libero/libero/init_files"
MAX_STEPS_DEFAULT = 520
DEFAULT_TASKS_CSV = (
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it,"
    "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate,"
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy"
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _short_task_name(task_stem: str) -> str:
    """Convert a long task stem to a compact filesystem-safe label.

    Algorithm:
      1. Lowercase the full stem.
      2. Match the leading scene token via regex ``^([a-z_]+_scene\\d+)``
         (e.g. ``kitchen_scene3``).
      3. From the remainder, split on ``_``, drop stopwords, keep the first
         3 surviving tokens.
      4. Concatenate ``<scene>_<tok1>_<tok2>_<tok3>``, strip trailing
         underscores, and truncate at 60 chars.

    Examples::

        >>> _short_task_name(
        ...     "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"
        ... )
        'kitchen_scene3_turn_stove_put'

        >>> _short_task_name(
        ...     "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_"
        ...     "put_the_yellow_and_white_mug_on_the_right_plate"
        ... )
        'living_room_scene5_put_white_mug'

        >>> _short_task_name(
        ...     "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_"
        ...     "compartment_of_the_caddy"
        ... )
        'study_scene1_pick_up_book'
    """
    stopwords = {"the", "and", "a", "an", "of", "on", "in", "to", "is", "it",
                 "its", "with", "both", "into"}
    s = task_stem.lower()
    m = re.match(r"^([a-z_]+_scene\d+)", s)
    if m:
        scene = m.group(1)
        remainder = s[m.end():].lstrip("_")
    else:
        # Fallback: no scene prefix found — treat whole string as remainder.
        scene = ""
        remainder = s

    tokens = [t for t in remainder.split("_") if t and t not in stopwords]
    kept = tokens[:3]
    parts = [scene] + kept if scene else kept
    label = "_".join(parts).rstrip("_")
    return label[:60]


def _load_init_states(path: str) -> np.ndarray:
    """Load a LIBERO-pro .pruned_init file and return a 2-D ndarray (N, D)."""
    state = torch.load(path, map_location="cpu")
    if isinstance(state, torch.Tensor):
        arr = state.cpu().numpy()
    elif isinstance(state, dict) and "init_states" in state:
        arr = state["init_states"]
        if isinstance(arr, torch.Tensor):
            arr = arr.cpu().numpy()
        arr = np.asarray(arr)
    else:
        arr = np.asarray(state)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    return arr


def _clamp_index(idx: int, n: int, label: str) -> int:
    """Clamp *idx* to the valid range ``[0, n-1]``, logging a warning if needed."""
    if idx < 0:
        return 0
    if idx >= n:
        logger.warning(
            "init_state_index %d out of range for %s (n=%d) — clamping to %d",
            idx, label, n, n - 1,
        )
        return n - 1
    return idx


def _build_perturbation_info(
    base_bddl: str,
    perturbed_bddl: str,
    task_stem: str,
) -> dict:
    """Compute rich perturbation metadata from two BDDL strings.

    Returns a JSON-serialisable dict suitable for ``perturbation_info.json``.
    """
    base_language = read_language(base_bddl)
    perturbed_language = read_language(perturbed_bddl)
    base_goal = read_goal(base_bddl)
    perturbed_goal = read_goal(perturbed_bddl)
    base_objects = read_objects(base_bddl)
    perturbed_objects = read_objects(perturbed_bddl)
    base_regions = read_regions(base_bddl)
    perturbed_regions = read_regions(perturbed_bddl)
    base_scene = read_scene(base_bddl)
    perturbed_scene = read_scene(perturbed_bddl)

    added_obj = sorted(set(perturbed_objects) - set(base_objects))
    removed_obj = sorted(set(base_objects) - set(perturbed_objects))
    added_reg = sorted(set(perturbed_regions) - set(base_regions))
    removed_reg = sorted(set(base_regions) - set(perturbed_regions))

    diff_text = unified_diff(base_bddl, perturbed_bddl, "base.bddl", "perturbed.bddl")
    n_additions = sum(
        1 for line in diff_text.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    n_removals = sum(
        1 for line in diff_text.splitlines()
        if line.startswith("-") and not line.startswith("---")
    )
    n_changed_lines = n_additions + n_removals

    fp = diff_fingerprint(diff_text)
    summary = perturbation_summary(
        added_obj, removed_obj, added_reg, removed_reg,
        base_language, perturbed_language, base_goal, perturbed_goal,
    )
    intensity = intensity_label(n_changed_lines)
    perturbed_sha1 = sha1_of(perturbed_bddl)

    return {
        "task_stem": task_stem,
        "perturbation_type": "swap",
        "base_language": base_language,
        "perturbed_language": perturbed_language,
        "base_goal": base_goal,
        "perturbed_goal": perturbed_goal,
        "base_scene": base_scene,
        "perturbed_scene": perturbed_scene,
        "base_objects": base_objects,
        "perturbed_objects": perturbed_objects,
        "added_objects": added_obj,
        "removed_objects": removed_obj,
        "base_regions": base_regions,
        "perturbed_regions": perturbed_regions,
        "added_regions": added_reg,
        "removed_regions": removed_reg,
        "perturbation_summary": summary,
        "perturbation_intensity": intensity,
        "n_changed_lines": n_changed_lines,
        "n_diff_additions": n_additions,
        "n_diff_removals": n_removals,
        "diff_fingerprint": fp,
        "perturbed_bddl_sha1": perturbed_sha1,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="pi05_libero × LIBERO-pro libero_10_swap 3-task × 3-trial evaluator",
    )
    parser.add_argument(
        "--vla-url",
        default=os.environ.get("VLA_SERVER_URL", "http://localhost:8400"),
        help="URL of the pi0.5 VLA HTTP server (default: %(default)s)",
    )
    parser.add_argument(
        "--tasks",
        default=DEFAULT_TASKS_CSV,
        help="Comma-separated task stems to evaluate",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=3,
        help="Number of trials (rollouts) per task",
    )
    parser.add_argument(
        "--init-state-indices",
        default="0,1,2",
        help="Comma-separated init-state indices, one per trial",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=MAX_STEPS_DEFAULT,
        help="Maximum env steps per rollout",
    )
    parser.add_argument(
        "--replan-steps",
        type=int,
        default=5,
        help="Re-plan every N env steps",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="Camera resolution (square)",
    )
    parser.add_argument(
        "--output-dir",
        default="/workspace/LIBERO-PRO/test_outputs/pi05_libero10_swap",
        help="Root output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Base random seed (seed + trial_idx used per trial)",
    )
    parser.add_argument(
        "--server-wait-sec",
        type=float,
        default=60.0,
        help="Max seconds to wait for VLA server to become ready",
    )
    args = parser.parse_args()

    # --- Parse task list and init-state indices ----------------------------- #
    tasks: List[str] = [t.strip() for t in args.tasks.split(",") if t.strip()]
    init_state_indices: List[int] = [
        int(x.strip()) for x in args.init_state_indices.split(",") if x.strip()
    ]
    if len(init_state_indices) != args.num_trials:
        parser.error(
            f"--init-state-indices has {len(init_state_indices)} entries but "
            f"--num-trials is {args.num_trials}"
        )

    # --- Logging ------------------------------------------------------------ #
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # --- Output directory --------------------------------------------------- #
    run_dir = pathlib.Path(args.output_dir) / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run directory: %s", run_dir)

    # --- Connect to VLA server --------------------------------------------- #
    client = VLAClient(args.vla_url, timeout=120.0)
    logger.info("Waiting for VLA server at %s (max %.0f s) …", args.vla_url, args.server_wait_sec)
    try:
        server_info = client.wait_until_ready(
            max_wait=args.server_wait_sec, poll_interval=2.0
        )
        logger.info("Server ready: %s", server_info)
    except (TimeoutError, Exception) as exc:
        logger.error(
            "VLA server not reachable at %s: %s\n"
            "Start the server with:\n"
            "  docker run --gpus all -p 8400:8400 <pi05_image> python server.py\n"
            "or set VLA_SERVER_URL to the correct address.",
            args.vla_url, exc,
        )
        sys.exit(2)

    # --- Main loop ---------------------------------------------------------- #
    summary_rows: List[dict] = []
    total_successes = 0
    total_episodes = 0

    for task_stem in tasks:
        base_path = f"{BDDL_BASE_DIR}/{BASE_SUITE_DIR}/{task_stem}.bddl"
        perturbed_path = f"{BDDL_BASE_DIR}/{SWAP_SUITE_DIR}/{task_stem}.bddl"
        init_path = f"{INIT_BASE_DIR}/{SWAP_SUITE_DIR}/{task_stem}.pruned_init"

        missing = [p for p in (base_path, perturbed_path, init_path) if not os.path.isfile(p)]
        if missing:
            logger.error(
                "task=%s  Skipping — missing files: %s", task_stem, missing
            )
            continue

        # Read BDDL files.
        with open(base_path, "r") as fh:
            base_bddl = fh.read()
        with open(perturbed_path, "r") as fh:
            perturbed_bddl = fh.read()

        # Build perturbation metadata.
        info = _build_perturbation_info(base_bddl, perturbed_bddl, task_stem)
        perturbed_bddl_sha1: str = info["perturbed_bddl_sha1"]

        # Create per-task output directory.
        short_name = _short_task_name(task_stem)
        task_dir = run_dir / short_name / "swap"
        task_dir.mkdir(parents=True, exist_ok=True)

        # Copy BDDL files and write diff + info.
        shutil.copy2(perturbed_path, task_dir / "perturbed.bddl")
        shutil.copy2(base_path, task_dir / "base.bddl")
        diff_text = unified_diff(base_bddl, perturbed_bddl, "base.bddl", "perturbed.bddl")
        (task_dir / "diff_vs_base.diff").write_text(diff_text)
        (task_dir / "perturbation_info.json").write_text(
            json.dumps(info, indent=2, default=str)
        )
        logger.info(
            "task=%s  perturbation_summary=%s intensity=%s",
            task_stem, info["perturbation_summary"], info["perturbation_intensity"],
        )

        # Load init states.
        try:
            states = _load_init_states(init_path)
        except Exception as exc:
            logger.error("task=%s  Failed to load init states from %s: %s", task_stem, init_path, exc)
            continue
        n_init_states = len(states)
        logger.info("task=%s  n_init_states=%d", task_stem, n_init_states)

        # Determine task instruction from perturbed BDDL (fall back to base).
        instruction = read_language(perturbed_bddl) or read_language(base_bddl)
        perturbed_language: str = info["perturbed_language"]

        # --- Trial loop ------------------------------------------------------ #
        for trial_idx, raw_state_idx in enumerate(init_state_indices):
            state_idx = _clamp_index(raw_state_idx, n_init_states, f"{task_stem}[trial={trial_idx}]")
            init_state = states[state_idx]

            logger.info(
                "task=%s  trial=%d/%d  init_state_idx=%d  instruction=%r",
                short_name, trial_idx + 1, args.num_trials, state_idx, instruction,
            )

            success = False
            termination = "not_run"
            n_steps = 0
            latency_avg_ms = 0.0
            video_name = ""
            t0 = time.time()

            try:
                success, termination, n_steps, replay, latencies = run_rollout(
                    client=client,
                    bddl_path=str(task_dir / "perturbed.bddl"),
                    instruction=instruction,
                    init_state=init_state,
                    resolution=args.resolution,
                    max_steps=args.max_steps,
                    replan_steps=args.replan_steps,
                    seed=args.seed + trial_idx,
                )
                latency_avg_ms = float(np.mean(latencies)) if latencies else 0.0

                prefix = "success" if success else "fail"
                video_name = f"{prefix}_trial{trial_idx}_{short_name}.mp4"
                video_path = task_dir / video_name
                try:
                    imageio.mimwrite(str(video_path), replay, fps=10)
                except Exception as vid_exc:
                    logger.warning(
                        "task=%s  trial=%d  video write failed: %s",
                        short_name, trial_idx, vid_exc,
                    )
                    video_name = ""
            except Exception as exc:  # noqa: BLE001
                termination = f"exception:{type(exc).__name__}"
                logger.error(
                    "task=%s  trial=%d  rollout exception:\n%s",
                    short_name, trial_idx, traceback.format_exc(),
                )

            elapsed = round(time.time() - t0, 2)
            total_episodes += 1
            if success:
                total_successes += 1

            logger.info(
                "task=%s  trial=%d  success=%s  n_steps=%d  "
                "termination=%s  latency_avg_ms=%.1f  elapsed=%.1fs",
                short_name, trial_idx, success, n_steps,
                termination, latency_avg_ms, elapsed,
            )

            summary_rows.append({
                "task": task_stem,
                "short_name": short_name,
                "trial": trial_idx,
                "init_state_idx": state_idx,
                "success": success,
                "n_env_steps": n_steps,
                "termination_reason": termination,
                "latency_avg_ms": round(latency_avg_ms, 2),
                "video_filename": video_name,
                "perturbation_type": "swap",
                "perturbed_language": perturbed_language,
                "perturbed_bddl_sha1": perturbed_bddl_sha1,
            })

    # --- Write summary.csv -------------------------------------------------- #
    summary_csv = run_dir / "summary.csv"
    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with open(summary_csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        logger.info("summary.csv written: %s", summary_csv)
    else:
        logger.warning("No trials completed — summary.csv not written.")

    # --- Write run_meta.json ------------------------------------------------ #
    success_rate = total_successes / total_episodes if total_episodes > 0 else 0.0
    run_meta = {
        "args": vars(args),
        "vla_url": args.vla_url,
        "server_info": server_info,
        "tasks": tasks,
        "num_tasks": len(tasks),
        "trials_per_task": args.num_trials,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "success_rate": round(success_rate, 4),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run_dir": str(run_dir),
    }
    run_meta_path = run_dir / "run_meta.json"
    run_meta_path.write_text(json.dumps(run_meta, indent=2, default=str))
    logger.info("run_meta.json written: %s", run_meta_path)

    # --- Final summary ------------------------------------------------------ #
    logger.info(
        "Eval complete. success_rate=%.1f%% (%d/%d)  run_dir=%s",
        success_rate * 100, total_successes, total_episodes, run_dir,
    )
    print(f"RUN_DIR={run_dir}")
    print(f"SUMMARY_CSV={summary_csv}")
    print(f"SUCCESS_RATE={success_rate:.4f}")


if __name__ == "__main__":
    main()
