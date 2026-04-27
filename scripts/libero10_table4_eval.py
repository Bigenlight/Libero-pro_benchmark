#!/usr/bin/env python3
"""libero10_table4_eval.py — Per-shard unified eval driver for Table 4.

Runs all 5 perturbation conditions (ori, pos, obj, sem, task) over a
subset of libero_10 tasks.  Outputs go to a SHARED timestamped run directory
(provided by the orchestrator via --output-dir) but to disjoint subdirs so
multiple shards can run in parallel without collision.

Output layout (under --output-dir / <ptype> / <short> / <subdir>):
    ori/<short>/baseline/
        base.bddl
        task_info.json
        success_trial0_<short>.mp4  (or fail_…)
        ...
    pos/<short>/variant0/
        base.bddl
        perturbed.bddl
        diff_vs_base.diff
        perturbation_info.json
        success_trial0_<short>.mp4
        ...
    obj|sem|task/<short>/variant{N}/
        same layout as pos

Top-level per-shard files:
    summary_<shard_tag>.csv
    run_meta_<shard_tag>.json
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import pathlib
import sys
import time
import traceback
from typing import Dict, List, Optional, Tuple

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
    unified_diff,
    diff_fingerprint,
    intensity_label,
    perturbation_summary,
    run_rollout,
    discover_unique_variants,
)
from libero10_swap_3task_eval import (  # noqa: E402
    _short_task_name,
    _build_perturbation_info,
    _clamp_index,
)

logger = logging.getLogger("libero10_table4_eval")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
LIBERO_PRO_ROOT = "/workspace/LIBERO-PRO"
SUITE_NAME = "libero_10"
OOD_BDDL_ROOT = "/workspace/LIBERO-PRO/libero/libero/bddl_files"
PRE_GENERATED_DIRS = {
    "pos":  f"{OOD_BDDL_ROOT}/libero_10_swap",
    "obj":  f"{OOD_BDDL_ROOT}/libero_10_object",
    "sem":  f"{OOD_BDDL_ROOT}/libero_10_lan",
    "task": f"{OOD_BDDL_ROOT}/libero_10_task",
}
PERTURBATOR_TYPE = {
    "pos": "swap",
    "obj": "object",
    "sem": "language",
    "task": "task",
}

# CSV columns — exactly this order.
CSV_COLUMNS = [
    "task",
    "short_name",
    "perturbation",
    "variant_idx",
    "variant_seed",
    "trial",
    "init_state_idx",
    "success",
    "n_env_steps",
    "termination_reason",
    "latency_avg_ms",
    "latency_min_ms",
    "latency_max_ms",
    "n_predictions",
    "video_filename",
    "perturbed_bddl_sha1",
    "perturbed_language",
    "instruction",
    "n_changed_lines",
    "perturbation_intensity",
    "error",
    "elapsed_sec",
    "timestamp",
    "shard_tag",
]


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="libero10 Table-4 per-shard unified eval driver",
    )
    p.add_argument(
        "--vla-url",
        default=os.environ.get("VLA_SERVER_URL", "http://localhost:8400"),
        help="URL of the VLA HTTP server (default: %(default)s)",
    )
    p.add_argument(
        "--tasks",
        required=True,
        help="REQUIRED. Comma-separated task stems to evaluate.",
    )
    p.add_argument(
        "--perturbations",
        default="ori,pos,obj,sem,task",
        help="Comma-separated perturbation types to run (default: %(default)s)",
    )
    p.add_argument(
        "--variants-obj",
        type=int,
        default=5,
        help="Number of unique obj variants per task (default: %(default)s)",
    )
    p.add_argument(
        "--variants-sem",
        type=int,
        default=3,
        help="Number of unique sem variants per task (default: %(default)s)",
    )
    p.add_argument(
        "--variants-task",
        type=int,
        default=5,
        help="Number of unique task variants per task (default: %(default)s)",
    )
    p.add_argument(
        "--trials-ori",
        type=int,
        default=10,
        help="Trials per task for ori condition (default: %(default)s)",
    )
    p.add_argument(
        "--trials-pos",
        type=int,
        default=10,
        help="Trials per task/variant for pos condition (default: %(default)s)",
    )
    p.add_argument(
        "--trials-other",
        type=int,
        default=5,
        help="Trials per variant for obj/sem/task conditions (default: %(default)s)",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=520,
        help="Maximum env steps per rollout (default: %(default)s)",
    )
    p.add_argument(
        "--replan-steps",
        type=int,
        default=5,
        help="Re-plan every N env steps (default: %(default)s)",
    )
    p.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="Camera resolution (square) (default: %(default)s)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Base random seed (default: %(default)s)",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="REQUIRED. Shared <timestamp> run directory (created by orchestrator).",
    )
    p.add_argument(
        "--shard-tag",
        default="shard0",
        help="Identifier for this shard (default: %(default)s)",
    )
    p.add_argument(
        "--server-wait-sec",
        type=float,
        default=600.0,
        help="Max seconds to wait for VLA server to become ready (default: %(default)s)",
    )
    p.add_argument(
        "--variant-seeds-start",
        type=int,
        default=1,
        help="First seed passed to discover_unique_variants (default: %(default)s)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip actual rollouts — log intent only.",
    )
    return p


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _trials_for(ptype: str, args: argparse.Namespace) -> int:
    if ptype == "ori":
        return args.trials_ori
    if ptype == "pos":
        return args.trials_pos
    return args.trials_other


def _wanted_variants(ptype: str, args: argparse.Namespace) -> int:
    return {"obj": args.variants_obj, "sem": args.variants_sem, "task": args.variants_task}[ptype]


def _collect_variants(
    ptype: str,
    task_stem: str,
    base_bddl: str,
    args: argparse.Namespace,
) -> List[Tuple[Optional[int], str]]:
    """Return list of (seed, bddl_text) for the given perturbation type.

    For ori: returns [(None, base_bddl, "baseline")] — caller handles specially.
    For pos: reads pre-generated file → 1 variant.
    For obj/sem/task: pre-generated file as variant 0, then discover_unique_variants.
    """
    if ptype == "ori":
        # Caller uses the base bddl as-is; seed=None, label="baseline".
        return [(None, base_bddl)]

    pre_dir = PRE_GENERATED_DIRS[ptype]
    pre_path = pathlib.Path(pre_dir) / f"{task_stem}.bddl"

    if ptype == "pos":
        if pre_path.is_file():
            try:
                return [(None, pre_path.read_text())]
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "task=%s ptype=%s  could not read pre-generated file %s: %s",
                    task_stem, ptype, pre_path, e,
                )
                return []
        else:
            logger.warning(
                "task=%s ptype=pos  pre-generated file not found: %s", task_stem, pre_path
            )
            return []

    # obj / sem / task — try to seed variant 0 from pre-generated file.
    wanted = _wanted_variants(ptype, args)
    variants: List[Tuple[Optional[int], str]] = []
    seen_sha1s: set = set()

    if pre_path.is_file():
        try:
            pre_bddl = pre_path.read_text()
            h = sha1_of(pre_bddl)
            seen_sha1s.add(h)
            variants.append((None, pre_bddl))
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "task=%s ptype=%s  could not read pre-generated file %s: %s",
                task_stem, ptype, pre_path, e,
            )

    remaining = wanted - len(variants)
    if remaining > 0:
        seed_range = range(
            args.variant_seeds_start,
            args.variant_seeds_start + 15,
        )
        try:
            discovered = discover_unique_variants(
                ptype=PERTURBATOR_TYPE[ptype],
                base_bddl=base_bddl,
                suite=SUITE_NAME,
                task_name=task_stem,
                wanted=remaining + len(variants),  # over-request, then dedupe
                seeds=list(seed_range),
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "task=%s ptype=%s  discover_unique_variants failed: %s",
                task_stem, ptype, e,
            )
            discovered = []

        for seed, bddl_text in discovered:
            if len(variants) >= wanted:
                break
            h = sha1_of(bddl_text)
            if h in seen_sha1s:
                continue
            seen_sha1s.add(h)
            variants.append((seed, bddl_text))

    return variants


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Validate task list.
    tasks: List[str] = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if not tasks:
        logger.error("--tasks is empty or invalid: %r", args.tasks)
        sys.exit(1)

    perturbations: List[str] = [p.strip() for p in args.perturbations.split(",") if p.strip()]

    # Output directory — orchestrator creates it; we just use it.
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Shard=%s  output_dir=%s", args.shard_tag, output_dir)

    # Connect to VLA server.
    client: Optional[VLAClient] = None
    server_info: dict = {}
    if not args.dry_run:
        client = VLAClient(args.vla_url, timeout=300.0)
        logger.info(
            "Waiting for VLA server at %s (max %.0f s) ...", args.vla_url, args.server_wait_sec
        )
        try:
            server_info = client.wait_until_ready(
                max_wait=args.server_wait_sec, poll_interval=3.0
            )
            logger.info("Server ready: %s", server_info)
        except Exception as exc:
            logger.error(
                "VLA server not reachable at %s: %s\n"
                "Set VLA_SERVER_URL or pass --vla-url.",
                args.vla_url, exc,
            )
            sys.exit(2)

    # Load benchmark suite once.
    sys.path.insert(0, LIBERO_PRO_ROOT)
    from libero.libero import benchmark as _benchmark  # noqa: E402
    from libero.libero import get_libero_path  # noqa: E402

    bd = _benchmark.get_benchmark_dict()
    suite = bd[SUITE_NAME]()
    names = suite.get_task_names()
    bddl_root = get_libero_path("bddl_files")

    summary_rows: List[dict] = []
    global_trial_counter = 0
    t_start_global = time.time()

    # ------------------------------------------------------------------ #
    # Task loop
    # ------------------------------------------------------------------ #
    for task_stem in tasks:
        # Look up task.
        try:
            task_id = names.index(task_stem)
        except ValueError:
            logger.error(
                "task=%s  Not found in suite %s. Skipping.", task_stem, SUITE_NAME
            )
            continue

        task = suite.get_task(task_id)
        base_path = pathlib.Path(bddl_root) / task.problem_folder / task.bddl_file

        try:
            base_bddl = base_path.read_text()
        except Exception as e:
            logger.error(
                "task=%s  Cannot read base BDDL from %s: %s. Skipping.", task_stem, base_path, e
            )
            continue

        base_language = read_language(base_bddl)
        base_goal = read_goal(base_bddl)
        base_objects = read_objects(base_bddl)
        base_regions = read_regions(base_bddl)
        base_scene = read_scene(base_bddl)
        base_bddl_sha1 = sha1_of(base_bddl)

        init_states = suite.get_task_init_states(task_id)
        n_init = len(init_states)
        short = _short_task_name(task_stem)

        logger.info(
            "task=%s  short=%s  n_init=%d  perturbations=%s",
            task_stem, short, n_init, perturbations,
        )

        # ---------------------------------------------------------------- #
        # Perturbation-type loop
        # ---------------------------------------------------------------- #
        for ptype in perturbations:

            # Collect variants.
            try:
                variants = _collect_variants(ptype, task_stem, base_bddl, args)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "task=%s ptype=%s  variant collection failed: %s. Skipping ptype.",
                    task_stem, ptype, e,
                )
                continue

            if not variants:
                logger.warning("task=%s ptype=%s  no variants found. Skipping.", task_stem, ptype)
                continue

            trials = _trials_for(ptype, args)

            for idx, (variant_seed, perturbed_bddl) in enumerate(variants):
                # Determine subdir.
                subdir_name = "baseline" if ptype == "ori" else f"variant{idx}"
                task_dir = output_dir / ptype / short / subdir_name
                task_dir.mkdir(parents=True, exist_ok=True)

                if ptype == "ori":
                    # Write base.bddl and task_info.json.
                    (task_dir / "base.bddl").write_text(base_bddl)
                    task_info = {
                        "task_stem": task_stem,
                        "short_name": short,
                        "perturbation": "ori",
                        "base_language": base_language,
                        "base_goal": base_goal,
                        "base_scene": base_scene,
                        "base_objects": base_objects,
                        "base_regions": base_regions,
                        "base_bddl_sha1": base_bddl_sha1,
                        "n_init_states_available": n_init,
                        "n_trials_planned": trials,
                    }
                    (task_dir / "task_info.json").write_text(
                        json.dumps(task_info, indent=2, default=str)
                    )
                    # For ori: perturbed == base.
                    perturbed_language = base_language
                    n_changed_lines = 0
                    perturbation_intensity_str = "none"
                    perturbed_bddl_sha1 = base_bddl_sha1
                    variant_seed_str = ""
                    rollout_bddl_path = str(task_dir / "base.bddl")

                else:
                    # Write base.bddl, perturbed.bddl, diff, perturbation_info.
                    (task_dir / "base.bddl").write_text(base_bddl)
                    (task_dir / "perturbed.bddl").write_text(perturbed_bddl)
                    diff_text = unified_diff(
                        base_bddl, perturbed_bddl, "base.bddl", "perturbed.bddl"
                    )
                    (task_dir / "diff_vs_base.diff").write_text(diff_text)

                    try:
                        pinfo = _build_perturbation_info(base_bddl, perturbed_bddl, task_stem)
                    except Exception as e:  # noqa: BLE001
                        logger.warning(
                            "task=%s ptype=%s variant=%d  _build_perturbation_info failed: %s",
                            task_stem, ptype, idx, e,
                        )
                        pinfo = {
                            "perturbed_bddl_sha1": sha1_of(perturbed_bddl),
                            "perturbed_language": read_language(perturbed_bddl),
                            "n_changed_lines": 0,
                            "perturbation_intensity": "unknown",
                        }

                    pinfo["variant_idx"] = idx
                    pinfo["variant_seed"] = variant_seed if variant_seed is not None else ""
                    (task_dir / "perturbation_info.json").write_text(
                        json.dumps(pinfo, indent=2, default=str)
                    )

                    perturbed_language = pinfo.get("perturbed_language", read_language(perturbed_bddl))
                    n_changed_lines = pinfo.get("n_changed_lines", 0)
                    perturbation_intensity_str = pinfo.get("perturbation_intensity", "unknown")
                    perturbed_bddl_sha1 = pinfo.get("perturbed_bddl_sha1", sha1_of(perturbed_bddl))
                    variant_seed_str = str(variant_seed) if variant_seed is not None else ""
                    rollout_bddl_path = str(task_dir / "perturbed.bddl")

                # -------------------------------------------------------- #
                # Trial loop
                # -------------------------------------------------------- #
                for trial_idx in range(trials):
                    init_idx = _clamp_index(
                        trial_idx, n_init, f"{short}/{ptype}/v{idx}"
                    )
                    init_state = init_states[init_idx]
                    instruction = read_language(perturbed_bddl) or base_language

                    t0 = time.time()
                    success = False
                    n_steps = 0
                    termination = "not_run"
                    latency_avg_ms = 0.0
                    latency_min_ms = 0.0
                    latency_max_ms = 0.0
                    n_predictions = 0
                    video_filename = ""
                    error_str = ""

                    if args.dry_run:
                        logger.info(
                            "[DRY-RUN] task=%s ptype=%s variant=%d trial=%d — skipping rollout",
                            task_stem, ptype, idx, trial_idx,
                        )
                        termination = "dry_run"
                        # Write placeholder video file name (empty file not written).
                        prefix = "fail"
                        video_filename = f"{prefix}_trial{trial_idx}_{short}.mp4"

                    else:
                        # Real rollout.
                        try:
                            success, termination, n_steps, replay, latencies = run_rollout(
                                client=client,
                                bddl_path=rollout_bddl_path,
                                instruction=instruction,
                                init_state=init_state,
                                resolution=args.resolution,
                                max_steps=args.max_steps,
                                replan_steps=args.replan_steps,
                                seed=args.seed + trial_idx,
                            )
                            if latencies:
                                latency_avg_ms = float(np.mean(latencies))
                                latency_min_ms = float(np.min(latencies))
                                latency_max_ms = float(np.max(latencies))
                                n_predictions = len(latencies)

                            prefix = "success" if success else "fail"
                            video_filename = f"{prefix}_trial{trial_idx}_{short}.mp4"
                            video_path = task_dir / video_filename
                            if replay:
                                try:
                                    imageio.mimwrite(str(video_path), replay, fps=10)
                                except Exception as vid_exc:
                                    logger.warning(
                                        "task=%s ptype=%s variant=%d trial=%d  video write failed: %s",
                                        task_stem, ptype, idx, trial_idx, vid_exc,
                                    )
                                    video_filename = ""

                        except Exception as exc:  # noqa: BLE001
                            termination = f"exception:{type(exc).__name__}"
                            error_str = f"{type(exc).__name__}: {exc}"
                            logger.error(
                                "task=%s ptype=%s variant=%d trial=%d  rollout exception:\n%s",
                                task_stem, ptype, idx, trial_idx, traceback.format_exc(),
                            )
                            prefix = "fail"
                            video_filename = f"{prefix}_trial{trial_idx}_{short}.mp4"

                    elapsed = round(time.time() - t0, 2)
                    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
                    global_trial_counter += 1

                    logger.info(
                        "task=%s  ptype=%s  variant=%d  trial=%d  success=%s  "
                        "n_steps=%d  termination=%s  latency_avg_ms=%.1f  elapsed=%.1fs",
                        task_stem, ptype, idx, trial_idx, success,
                        n_steps, termination, latency_avg_ms, elapsed,
                    )

                    summary_rows.append({
                        "task": task_stem,
                        "short_name": short,
                        "perturbation": ptype,
                        "variant_idx": idx,
                        "variant_seed": variant_seed_str,
                        "trial": trial_idx,
                        "init_state_idx": init_idx,
                        "success": success,
                        "n_env_steps": n_steps,
                        "termination_reason": termination,
                        "latency_avg_ms": round(latency_avg_ms, 2),
                        "latency_min_ms": round(latency_min_ms, 2),
                        "latency_max_ms": round(latency_max_ms, 2),
                        "n_predictions": n_predictions,
                        "video_filename": video_filename,
                        "perturbed_bddl_sha1": perturbed_bddl_sha1,
                        "perturbed_language": perturbed_language,
                        "instruction": instruction,
                        "n_changed_lines": n_changed_lines,
                        "perturbation_intensity": perturbation_intensity_str,
                        "error": error_str,
                        "elapsed_sec": elapsed,
                        "timestamp": ts,
                        "shard_tag": args.shard_tag,
                    })

    # ---------------------------------------------------------------------- #
    # Write per-shard outputs (always, even if exceptions occurred above)
    # ---------------------------------------------------------------------- #
    total_episodes = len(summary_rows)
    total_successes = sum(1 for r in summary_rows if r["success"])
    success_rate = round(total_successes / total_episodes, 4) if total_episodes > 0 else 0.0
    elapsed_total = round(time.time() - t_start_global, 2)

    # summary CSV
    summary_csv = output_dir / f"summary_{args.shard_tag}.csv"
    try:
        with open(summary_csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(summary_rows)
        logger.info("summary CSV written: %s", summary_csv)
    except Exception as e:
        logger.error("Failed to write summary CSV: %s", e)

    # run_meta JSON
    run_meta_path = output_dir / f"run_meta_{args.shard_tag}.json"
    run_meta = {
        "args": vars(args),
        "vla_url": args.vla_url,
        "server_info": server_info,
        "tasks": tasks,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "success_rate": success_rate,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_sec": elapsed_total,
        "shard_tag": args.shard_tag,
    }
    try:
        run_meta_path.write_text(json.dumps(run_meta, indent=2, default=str))
        logger.info("run_meta written: %s", run_meta_path)
    except Exception as e:
        logger.error("Failed to write run_meta JSON: %s", e)

    logger.info(
        "Shard=%s done.  success_rate=%.1f%% (%d/%d)  elapsed=%.0fs  output_dir=%s",
        args.shard_tag, success_rate * 100, total_successes, total_episodes,
        elapsed_total, output_dir,
    )
    print(f"SHARD_TAG={args.shard_tag}")
    print(f"OUTPUT_DIR={output_dir}")
    print(f"SUMMARY_CSV={summary_csv}")
    print(f"SUCCESS_RATE={success_rate:.4f}")


if __name__ == "__main__":
    main()
