#!/usr/bin/env python3
"""pi05_libero × LIBERO-pro `libero_10` baseline (no perturbation) evaluator.

For each of 6 libero_10 tasks, iterates over init_states[0..max_trials-1].
Stops at the first success and saves that video; if all trials fail, saves all
failure videos.  Mirrors the swap driver's output layout but under
``pi05_libero10_baseline/``.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import pathlib
import shutil
import sys
import time
import traceback
from typing import Dict, List, Optional

import imageio
import numpy as np

# ---- path setup ------------------------------------------------------------ #
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from vla_client import VLAClient  # noqa: E402
from libero_pro_perturbation_sweep import (  # noqa: E402
    read_language,
    read_objects,
    read_regions,
    read_goal,
    read_scene,
    sha1_of,
    run_rollout,
)
from libero10_swap_3task_eval import _short_task_name  # noqa: E402

logger = logging.getLogger("libero10_baseline_eval")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
LIBERO_PRO_ROOT = "/workspace/LIBERO-PRO"
SUITE_NAME = "libero_10"
MAX_STEPS_DEFAULT = 520
DEFAULT_TASKS_CSV = (
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it,"
    "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate,"
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy,"
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it,"
    "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it,"
    "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket"
)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="pi05_libero × LIBERO-pro libero_10 baseline (no perturbation) evaluator",
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
        "--max-trials",
        type=int,
        default=10,
        help="Maximum number of trials (rollouts) per task; stops early on first success",
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
        default="/workspace/LIBERO-PRO/test_outputs/pi05_libero10_baseline",
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

    # --- Logging ------------------------------------------------------------ #
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # --- Parse task list ---------------------------------------------------- #
    tasks: List[str] = [t.strip() for t in args.tasks.split(",") if t.strip()]

    # --- Output directory --------------------------------------------------- #
    run_dir = pathlib.Path(args.output_dir) / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run directory: %s", run_dir)

    # --- Connect to VLA server --------------------------------------------- #
    client = VLAClient(args.vla_url, timeout=120.0)
    logger.info("Waiting for VLA server at %s (max %.0f s) ...", args.vla_url, args.server_wait_sec)
    try:
        server_info = client.wait_until_ready(
            max_wait=args.server_wait_sec, poll_interval=2.0
        )
        logger.info("Server ready: %s", server_info)
    except Exception as exc:
        logger.error(
            "VLA server not reachable at %s: %s\n"
            "Start the pi0.5 server with:\n"
            "  docker run --gpus all -p 8400:8400 <pi05_image> python server.py\n"
            "or set VLA_SERVER_URL to the correct address.",
            args.vla_url, exc,
        )
        sys.exit(2)

    # --- Load benchmark suite ONCE ----------------------------------------- #
    sys.path.insert(0, LIBERO_PRO_ROOT)
    from libero.libero import benchmark as _benchmark  # noqa: E402
    from libero.libero import get_libero_path  # noqa: E402

    bd = _benchmark.get_benchmark_dict()
    suite = bd[SUITE_NAME]()
    names = suite.get_task_names()
    bddl_root = get_libero_path("bddl_files")

    # --- Main loop ---------------------------------------------------------- #
    summary_rows: List[dict] = []
    trial_rows: List[dict] = []

    for task_stem in tasks:
        # Look up task id by name.
        try:
            task_id = names.index(task_stem)
        except ValueError:
            logger.error(
                "task=%s  Not found in suite %s (available: %d tasks). Skipping.",
                task_stem, SUITE_NAME, len(names),
            )
            continue

        task = suite.get_task(task_id)
        base_path = str(pathlib.Path(bddl_root) / task.problem_folder / task.bddl_file)

        # Read base BDDL text.
        with open(base_path, "r") as fh:
            base_bddl = fh.read()

        base_language = read_language(base_bddl)
        base_goal = read_goal(base_bddl)
        base_objects = read_objects(base_bddl)
        base_regions = read_regions(base_bddl)
        base_scene = read_scene(base_bddl)

        # Load init states from the suite object.
        init_states = suite.get_task_init_states(task_id)
        n_init_states = len(init_states)

        short = _short_task_name(task_stem)
        task_dir = run_dir / short / "baseline"
        task_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(base_path, task_dir / "base.bddl")

        n_attempts = min(args.max_trials, n_init_states)
        if n_attempts < args.max_trials:
            logger.info(
                "task=%s  max_trials=%d clipped to n_init_states=%d",
                short, args.max_trials, n_init_states,
            )

        # --- Trial loop with early exit on success -------------------------- #
        success_at_trial: Optional[int] = None
        total_steps_to_success: Optional[int] = None
        video_filenames: List[str] = []
        latencies_avg_per_trial: List[float] = []
        n_trials_attempted = 0
        final_status = "all_failed"

        for trial_idx in range(n_attempts):
            init_state = init_states[trial_idx]
            n_trials_attempted += 1
            t0 = time.time()
            try:
                success, termination, n_steps, replay, latencies = run_rollout(
                    client=client,
                    bddl_path=str(task_dir / "base.bddl"),
                    instruction=base_language,
                    init_state=init_state,
                    resolution=args.resolution,
                    max_steps=args.max_steps,
                    replan_steps=args.replan_steps,
                    seed=args.seed + trial_idx,
                )
            except Exception as exc:
                logger.error("trial exception: %s\n%s", exc, traceback.format_exc())
                success, termination, n_steps, replay, latencies = (
                    False, f"exception:{type(exc).__name__}", 0, [], []
                )

            latency_avg = float(np.mean(latencies)) if latencies else 0.0
            latencies_avg_per_trial.append(latency_avg)

            prefix = "success" if success else "fail"
            video_name = f"{prefix}_trial{trial_idx}_{short}.mp4"
            video_path = task_dir / video_name
            if replay:
                try:
                    imageio.mimwrite(str(video_path), replay, fps=10)
                    video_filenames.append(video_name)
                except Exception as vid_exc:
                    logger.warning("video write failed: %s", vid_exc)
                    video_name = ""
            else:
                video_name = ""

            # Per-trial row.
            trial_rows.append({
                "task": task_stem,
                "short_name": short,
                "trial": trial_idx,
                "success": success,
                "n_env_steps": n_steps,
                "termination_reason": termination,
                "latency_avg_ms": round(latency_avg, 2),
                "video_filename": video_name,
                "elapsed_sec": round(time.time() - t0, 2),
            })

            logger.info(
                "task=%s  trial=%d  success=%s  n_steps=%d  termination=%s  "
                "latency_avg_ms=%.1f  elapsed=%.1fs",
                short, trial_idx, success, n_steps, termination,
                latency_avg, time.time() - t0,
            )

            if success:
                success_at_trial = trial_idx
                total_steps_to_success = n_steps
                final_status = "succeeded"
                break  # EARLY EXIT

        # --- Write task_info.json ------------------------------------------- #
        task_info = {
            "task_stem": task_stem,
            "short_name": short,
            "base_language": base_language,
            "base_goal": base_goal,
            "base_scene": base_scene,
            "base_objects": base_objects,
            "base_regions": base_regions,
            "n_init_states_available": n_init_states,
            "n_trials_attempted": n_trials_attempted,
            "success_at_trial": success_at_trial,
            "total_steps_to_success": total_steps_to_success,
            "video_filenames": video_filenames,
            "final_status": final_status,
            "base_bddl_sha1": sha1_of(base_bddl),
        }
        (task_dir / "task_info.json").write_text(json.dumps(task_info, indent=2, default=str))

        # --- Per-task aggregate summary row --------------------------------- #
        summary_rows.append({
            "task": task_stem,
            "short_name": short,
            "n_init_states_available": n_init_states,
            "n_trials_attempted": n_trials_attempted,
            "success_at_trial": "" if success_at_trial is None else success_at_trial,
            "total_steps_to_success": "" if total_steps_to_success is None else total_steps_to_success,
            "latency_avg_ms": round(
                float(np.mean(latencies_avg_per_trial)) if latencies_avg_per_trial else 0.0, 2
            ),
            "video_filenames": json.dumps(video_filenames),
            "final_status": final_status,
            "base_language": base_language,
            "base_bddl_sha1": sha1_of(base_bddl),
        })

        logger.info(
            "task=%s  final_status=%s  success_at_trial=%s  n_trials_attempted=%d",
            short, final_status, success_at_trial, n_trials_attempted,
        )

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
        logger.warning("No tasks completed — summary.csv not written.")

    # --- Write trials.csv --------------------------------------------------- #
    trials_csv = run_dir / "trials.csv"
    if trial_rows:
        fieldnames_t = list(trial_rows[0].keys())
        with open(trials_csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames_t)
            writer.writeheader()
            writer.writerows(trial_rows)
        logger.info("trials.csv written: %s", trials_csv)
    else:
        logger.warning("No trials completed — trials.csv not written.")

    # --- Write run_meta.json ------------------------------------------------ #
    total_tasks_with_success = sum(1 for r in summary_rows if r["final_status"] == "succeeded")
    total_trials_run = sum(r["n_trials_attempted"] for r in summary_rows)
    success_rate_per_task = (
        total_tasks_with_success / len(summary_rows) if summary_rows else 0.0
    )
    run_meta = {
        "args": vars(args),
        "vla_url": args.vla_url,
        "server_info": server_info,
        "tasks": tasks,
        "num_tasks": len(tasks),
        "max_trials": args.max_trials,
        "total_tasks_with_success": total_tasks_with_success,
        "total_trials_run": total_trials_run,
        "success_rate_per_task": round(success_rate_per_task, 4),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run_dir": str(run_dir),
    }
    run_meta_path = run_dir / "run_meta.json"
    run_meta_path.write_text(json.dumps(run_meta, indent=2, default=str))
    logger.info("run_meta.json written: %s", run_meta_path)

    # --- Final summary ------------------------------------------------------ #
    logger.info(
        "Eval complete. tasks_succeeded=%d/%d  total_trials=%d  "
        "success_rate_per_task=%.1f%%  run_dir=%s",
        total_tasks_with_success, len(summary_rows), total_trials_run,
        success_rate_per_task * 100, run_dir,
    )
    print(f"RUN_DIR={run_dir}")
    print(f"SUMMARY_CSV={summary_csv}")
    print(f"SUCCESS_RATE={success_rate_per_task:.4f}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()
