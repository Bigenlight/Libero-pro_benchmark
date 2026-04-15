#!/usr/bin/env python3
"""Merge multiple sharded libero_vla_eval summary.json files into one aggregate.

Used by the multi-GPU orchestrator to turn per-shard outputs (e.g. running
task ids [0-3], [4-6], [7-9] on three parallel GPUs) into a single
benchmark-level score for the whole suite.

Usage:
    python scripts/merge_eval_summaries.py \\
        --inputs outputs/shard_g1/summary.json \\
                 outputs/shard_g2/summary.json \\
                 outputs/shard_g3/summary.json \\
        --output outputs/libero_spatial_full_summary.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import List


def merge(input_paths: List[pathlib.Path]) -> dict:
    shards = []
    for p in input_paths:
        with open(p) as f:
            shards.append(json.load(f))
    if not shards:
        raise SystemExit("no input shards")

    # Basic invariants: same suite + same server_info.model
    suite = shards[0]["suite"]
    model = shards[0].get("server_info", {}).get("model", "")
    for s in shards[1:]:
        if s["suite"] != suite:
            raise SystemExit(f"suite mismatch: {s['suite']!r} vs {suite!r}")
        if s.get("server_info", {}).get("model", "") != model:
            sys.stderr.write(
                f"WARN: server model mismatch across shards ({model!r} vs "
                f"{s.get('server_info', {}).get('model', '')!r})\n"
            )

    # De-duplicate task entries by task_id (if a task was accidentally run
    # in multiple shards, prefer the first and flag it).
    merged_tasks: dict = {}
    for s in shards:
        for t in s.get("tasks", []):
            tid = int(t["task_id"])
            if tid in merged_tasks:
                sys.stderr.write(f"WARN: duplicate task_id {tid} across shards — keeping first\n")
                continue
            merged_tasks[tid] = dict(t)

    tasks_sorted = [merged_tasks[k] for k in sorted(merged_tasks)]
    total_successes = sum(int(t.get("successes", 0)) for t in tasks_sorted)
    total_trials = sum(int(t.get("trials", 0)) for t in tasks_sorted)
    weighted_latency_num = 0.0
    weighted_latency_den = 0
    for t in tasks_sorted:
        n = int(t.get("num_predictions", 0))
        weighted_latency_num += float(t.get("avg_latency_ms", 0.0)) * n
        weighted_latency_den += n
    avg_latency_ms = weighted_latency_num / weighted_latency_den if weighted_latency_den else 0.0

    return {
        "suite": suite,
        "server_model": model,
        "n_tasks": len(tasks_sorted),
        "total_episodes": total_trials,
        "total_successes": total_successes,
        "success_rate": (total_successes / total_trials) if total_trials else 0.0,
        "avg_latency_ms": avg_latency_ms,
        "shard_count": len(shards),
        "shard_sources": [str(p) for p in input_paths],
        "tasks": [
            {
                "task_id": t["task_id"],
                "description": t.get("description", ""),
                "successes": t.get("successes", 0),
                "trials": t.get("trials", 0),
                "success_rate": (
                    t.get("successes", 0) / t.get("trials", 0) if t.get("trials") else 0.0
                ),
                "avg_latency_ms": t.get("avg_latency_ms", 0.0),
                "num_predictions": t.get("num_predictions", 0),
            }
            for t in tasks_sorted
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="paths to shard summary.json files")
    ap.add_argument("--output", required=True, help="merged summary output path")
    args = ap.parse_args()

    merged = merge([pathlib.Path(p) for p in args.inputs])
    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, indent=2))

    # Terse stdout line for the orchestrator.
    print(
        f"{merged['suite']}  {merged['total_successes']}/{merged['total_episodes']}  "
        f"({100.0 * merged['success_rate']:.1f}%)  avg_latency={merged['avg_latency_ms']:.1f}ms"
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
