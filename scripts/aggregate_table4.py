"""
aggregate_table4.py — Aggregate per-shard summary CSVs from a LIBERO-pro run
and produce summary.csv, cell_aggregate.csv, and table4_reproduction.txt.

Usage:
    python3 aggregate_table4.py --run-dir /path/to/run_dir [--paper-pi05 CSV]
"""

import argparse
import logging
import os
import sys
from glob import glob

logger = logging.getLogger(__name__)

CANONICAL_TASKS = [
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
    "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
    "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
    "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
    "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
    "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
    "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
    "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
]

PERTURBATIONS = ["ori", "obj", "pos", "sem", "task"]  # paper column order
PERT_HEADERS = ["Ori", "Obj", "Pos", "Sem", "Task"]

DEFAULT_PAPER_PI05 = "0.93,0.92,0.08,0.93,0.01"


# ---------------------------------------------------------------------------
# Helpers — parse success value regardless of type
# ---------------------------------------------------------------------------

def to_bool(val):
    """Convert 'True'/'False' strings or Python bools/ints to int (0 or 1)."""
    if isinstance(val, bool):
        return int(val)
    if isinstance(val, int):
        return val
    s = str(val).strip().lower()
    if s in ("true", "1", "yes"):
        return 1
    if s in ("false", "0", "no"):
        return 0
    raise ValueError(f"Cannot parse success value: {val!r}")


def truncate(name, maxlen=50):
    """Truncate a string to maxlen chars, appending '..' if truncated."""
    if len(name) <= maxlen:
        return name
    return name[: maxlen - 2] + ".."


# ---------------------------------------------------------------------------
# Shard discovery
# ---------------------------------------------------------------------------

def find_shard_files(run_dir):
    """Return shard CSV paths sorted by shard tag (g1, g2, g3, ...)."""
    pattern = os.path.join(run_dir, "summary_*.csv")
    files = glob(pattern)
    if not files:
        return []

    def shard_key(p):
        base = os.path.splitext(os.path.basename(p))[0]  # e.g. summary_g1
        tag = base.replace("summary_", "")               # e.g. g1
        # Try to extract numeric suffix for sorting
        digits = "".join(c for c in tag if c.isdigit())
        try:
            return (tag[0], int(digits)) if digits else (tag, 0)
        except Exception:
            return (tag, 0)

    files.sort(key=shard_key)
    return files


# ---------------------------------------------------------------------------
# Pandas path
# ---------------------------------------------------------------------------

def run_with_pandas(run_dir, paper_avgs):
    import pandas as pd

    shard_files = find_shard_files(run_dir)
    if not shard_files:
        logger.error("No summary_*.csv files found in %s", run_dir)
        sys.exit(1)

    logger.info("Found %d shard file(s): %s", len(shard_files), shard_files)

    # 1. Load and concatenate
    dfs = []
    for f in shard_files:
        try:
            df = pd.read_csv(f, dtype=str)
            dfs.append(df)
            logger.info("  Loaded %s (%d rows)", f, len(df))
        except Exception as e:
            logger.warning("  Skipping %s: %s", f, e)

    if not dfs:
        logger.error("All shard files failed to load.")
        sys.exit(1)

    combined = pd.concat(dfs, ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates()
    after = len(combined)
    if before != after:
        logger.info("Dropped %d duplicate rows.", before - after)

    # Write summary.csv
    summary_path = os.path.join(run_dir, "summary.csv")
    combined.to_csv(summary_path, index=False)
    logger.info("Wrote %s (%d rows)", summary_path, len(combined))

    # Parse success
    combined["success_int"] = combined["success"].apply(to_bool)

    # 2. Cell aggregate
    agg = (
        combined.groupby(["task", "perturbation"], sort=False)
        .agg(
            short_name=("short_name", "first"),
            n_episodes=("success_int", "count"),
            n_success=("success_int", "sum"),
            success_rate=("success_int", "mean"),
        )
        .reset_index()
    )

    # Sort: CANONICAL_TASKS order, PERTURBATIONS order
    task_order = {t: i for i, t in enumerate(CANONICAL_TASKS)}
    pert_order = {p: i for i, p in enumerate(PERTURBATIONS)}

    agg["_task_ord"] = agg["task"].map(lambda t: task_order.get(t, 999))
    agg["_pert_ord"] = agg["perturbation"].map(lambda p: pert_order.get(p, 999))
    agg = agg.sort_values(["_task_ord", "_pert_ord"]).drop(columns=["_task_ord", "_pert_ord"])

    agg = agg[["task", "short_name", "perturbation", "n_episodes", "n_success", "success_rate"]]
    cell_path = os.path.join(run_dir, "cell_aggregate.csv")
    agg.to_csv(cell_path, index=False)
    logger.info("Wrote %s (%d rows)", cell_path, len(agg))

    # 3. Build cell dict: {task: {pert: (rate, n_success, n_episodes)}}
    cell = {}
    for _, row in agg.iterrows():
        t = row["task"]
        p = row["perturbation"]
        cell.setdefault(t, {})[p] = (float(row["success_rate"]), int(row["n_success"]), int(row["n_episodes"]))

    total_episodes = int(combined["success_int"].count())
    table_text = build_table(cell, total_episodes, run_dir, paper_avgs)

    table_path = os.path.join(run_dir, "table4_reproduction.txt")
    with open(table_path, "w") as f:
        f.write(table_text)
    logger.info("Wrote %s", table_path)
    print(table_text)


# ---------------------------------------------------------------------------
# Pure-csv fallback path
# ---------------------------------------------------------------------------

def run_with_csv(run_dir, paper_avgs):
    import csv

    shard_files = find_shard_files(run_dir)
    if not shard_files:
        logger.error("No summary_*.csv files found in %s", run_dir)
        sys.exit(1)

    logger.info("Found %d shard file(s) (csv-module path)", len(shard_files))

    all_rows = []
    fieldnames = None
    seen = set()  # for dedup

    for f in shard_files:
        try:
            with open(f, newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                if fieldnames is None:
                    fieldnames = reader.fieldnames
                for row in reader:
                    key = tuple(sorted(row.items()))
                    if key not in seen:
                        seen.add(key)
                        all_rows.append(row)
            logger.info("  Loaded %s", f)
        except Exception as e:
            logger.warning("  Skipping %s: %s", f, e)

    if not all_rows:
        logger.error("All shard files failed to load.")
        sys.exit(1)

    # Write summary.csv
    summary_path = os.path.join(run_dir, "summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames or list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    logger.info("Wrote %s (%d rows)", summary_path, len(all_rows))

    # Accumulate: {(task, pert): {"short_name", "total", "successes"}}
    acc = {}
    for row in all_rows:
        t = row.get("task", "")
        p = row.get("perturbation", "")
        sn = row.get("short_name", "")
        try:
            s = to_bool(row.get("success", "False"))
        except ValueError:
            s = 0
        key = (t, p)
        if key not in acc:
            acc[key] = {"short_name": sn, "total": 0, "successes": 0}
        acc[key]["total"] += 1
        acc[key]["successes"] += s

    # Write cell_aggregate.csv
    task_order = {t: i for i, t in enumerate(CANONICAL_TASKS)}
    pert_order = {p: i for i, p in enumerate(PERTURBATIONS)}

    sorted_keys = sorted(acc.keys(), key=lambda k: (task_order.get(k[0], 999), pert_order.get(k[1], 999)))

    cell_path = os.path.join(run_dir, "cell_aggregate.csv")
    with open(cell_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["task", "short_name", "perturbation", "n_episodes", "n_success", "success_rate"])
        for (t, p) in sorted_keys:
            d = acc[(t, p)]
            n = d["total"]
            s = d["successes"]
            rate = s / n if n > 0 else 0.0
            writer.writerow([t, d["short_name"], p, n, s, f"{rate:.6f}"])
    logger.info("Wrote %s", cell_path)

    # Build cell dict
    cell = {}
    for (t, p), d in acc.items():
        n = d["total"]
        s = d["successes"]
        rate = s / n if n > 0 else 0.0
        cell.setdefault(t, {})[p] = (rate, s, n)

    total_episodes = sum(d["total"] for d in acc.values())
    table_text = build_table(cell, total_episodes, run_dir, paper_avgs)

    table_path = os.path.join(run_dir, "table4_reproduction.txt")
    with open(table_path, "w") as f:
        f.write(table_text)
    logger.info("Wrote %s", table_path)
    print(table_text)


# ---------------------------------------------------------------------------
# Table renderer — stdlib only
# ---------------------------------------------------------------------------

def build_table(cell, total_episodes, run_dir, paper_avgs):
    """Render the ASCII table as a string."""

    lines = []

    def add(s=""):
        lines.append(s)

    # --- Header
    add("LIBERO-pro Table 4 — pi0.5 reproduction (libero_10, env-skipped)")
    add(f"Run dir: {run_dir}")
    add(f"Tasks: {len(CANONICAL_TASKS)}  |  Perturbations: {len(PERTURBATIONS)}  |  Total episodes: {total_episodes}")
    add()

    # Column widths
    W_IDX = 4    # "#   "
    W_TASK = 50  # task name
    W_CELL = 5   # "0.NN"
    W_CELL_EP = 7  # "10/10  "

    SEP_RATE = _make_sep(W_IDX, W_TASK, [W_CELL] * len(PERTURBATIONS))
    SEP_EP   = _make_sep(W_IDX, W_TASK, [W_CELL_EP] * len(PERTURBATIONS))

    # Header row
    add(SEP_RATE)
    header = _make_row(W_IDX, W_TASK, [W_CELL] * len(PERTURBATIONS),
                       "#", "Task", PERT_HEADERS)
    add(header)
    add(SEP_RATE)

    # Per-task rows — collect avgs
    pert_sums = {p: 0.0 for p in PERTURBATIONS}
    pert_counts = {p: 0 for p in PERTURBATIONS}
    missing_notes = []  # (task_num, task_name, pert)

    task_rows = []
    for idx, task in enumerate(CANONICAL_TASKS, start=1):
        task_cell = cell.get(task, {})
        cells_vals = []
        for p in PERTURBATIONS:
            if p in task_cell:
                rate, _, _ = task_cell[p]
                cells_vals.append(f"{rate:.2f}")
                pert_sums[p] += rate
                pert_counts[p] += 1
            else:
                cells_vals.append("--")
                missing_notes.append((idx, truncate(task, 30), p))
        row = _make_row(W_IDX, W_TASK, [W_CELL] * len(PERTURBATIONS),
                        str(idx), truncate(task, W_TASK), cells_vals)
        task_rows.append(row)

    for row in task_rows:
        add(row)

    add(SEP_RATE)

    # Avg row
    avg_vals = []
    for p in PERTURBATIONS:
        if pert_counts[p] > 0:
            avg_vals.append(f"{pert_sums[p] / pert_counts[p]:.2f}")
        else:
            avg_vals.append("--")

    avg_row = _make_row(W_IDX, W_TASK, [W_CELL] * len(PERTURBATIONS),
                        "Avg", "", avg_vals)
    add(avg_row)
    add(SEP_RATE)
    add()

    # Paper reference + delta
    paper_labels = ["Ori", "Obj", "Pos", "Sem", "Task"]
    paper_parts = "  ".join(f"{lbl}={paper_avgs[i]:.2f}" for i, lbl in enumerate(paper_labels))
    add("Paper Pi0.5 reference (Table 4, libero-10, env-skipped column):")
    add(f"  Avg: {paper_parts}")
    add()

    # Delta
    our_avgs = []
    for p in PERTURBATIONS:
        if pert_counts[p] > 0:
            our_avgs.append(pert_sums[p] / pert_counts[p])
        else:
            our_avgs.append(None)

    delta_parts = []
    for i, lbl in enumerate(paper_labels):
        if our_avgs[i] is not None:
            d = our_avgs[i] - paper_avgs[i]
            delta_parts.append(f"{lbl}={d:+.2f}")
        else:
            delta_parts.append(f"{lbl}=N/A ")
    add("Delta (ours - paper):")
    add("  Avg: " + "  ".join(delta_parts))
    add()

    # Episodes-per-cell table
    add("Episodes per cell:")
    add(SEP_EP)
    ep_header = _make_row(W_IDX, W_TASK, [W_CELL_EP] * len(PERTURBATIONS),
                          "#", "Task", PERT_HEADERS)
    add(ep_header)
    add(SEP_EP)

    for idx, task in enumerate(CANONICAL_TASKS, start=1):
        task_cell = cell.get(task, {})
        ep_vals = []
        for p in PERTURBATIONS:
            if p in task_cell:
                _, ns, ne = task_cell[p]
                ep_vals.append(f"{ns}/{ne}")
            else:
                ep_vals.append("--")
        row = _make_row(W_IDX, W_TASK, [W_CELL_EP] * len(PERTURBATIONS),
                        str(idx), truncate(task, W_TASK), ep_vals)
        add(row)

    add(SEP_EP)
    add()

    # Notes
    add("Notes:")
    add("- Missing cells shown as \"--\".")
    if missing_notes:
        for task_num, task_short, p in missing_notes:
            add(f"- Task {task_num} ({task_short}) missing for perturbation '{p}' (possible shard failure).")

    return "\n".join(lines) + "\n"


def _make_sep(w_idx, w_task, w_cells):
    """Build a separator line like +----+----+..."""
    parts = ["+" + "-" * (w_idx + 1)]
    parts.append("-" * (w_task + 2) + "+")
    for w in w_cells:
        parts.append("-" * (w + 2) + "+")
    return "".join(parts)


def _make_row(w_idx, w_task, w_cells, idx_val, task_val, cell_vals):
    """Build a data row like | idx | task | c1 | c2 | ..."""
    row = f"| {idx_val:<{w_idx}} "
    row += f"| {task_val:<{w_task}} "
    for i, cv in enumerate(cell_vals):
        row += f"| {cv:<{w_cells[i]}} "
    row += "|"
    return row


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate LIBERO-pro shard summaries into Table 4 format."
    )
    parser.add_argument(
        "--run-dir", required=True,
        help="Path to the run directory containing summary_*.csv files.",
    )
    parser.add_argument(
        "--paper-pi05",
        default=DEFAULT_PAPER_PI05,
        help=(
            "Comma-separated paper Pi0.5 averages for ori,obj,pos,sem,task "
            f"(default: {DEFAULT_PAPER_PI05})"
        ),
    )
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args = parse_args()
    run_dir = os.path.abspath(args.run_dir)

    if not os.path.isdir(run_dir):
        logger.error("Run directory does not exist: %s", run_dir)
        sys.exit(1)

    # Parse paper reference values
    try:
        paper_avgs = [float(x.strip()) for x in args.paper_pi05.split(",")]
        if len(paper_avgs) != len(PERTURBATIONS):
            raise ValueError(f"Expected {len(PERTURBATIONS)} values, got {len(paper_avgs)}")
    except Exception as e:
        logger.error("Invalid --paper-pi05 value: %s", e)
        sys.exit(1)

    # Check shard files exist before choosing backend
    shard_files = find_shard_files(run_dir)
    if not shard_files:
        logger.error(
            "No summary_*.csv files found in: %s\n"
            "Expected files like summary_g1.csv, summary_g2.csv, ...",
            run_dir,
        )
        sys.exit(1)

    # Try pandas, fall back to csv module
    try:
        import pandas  # noqa: F401
        logger.info("Using pandas backend.")
        run_with_pandas(run_dir, paper_avgs)
    except ImportError:
        logger.info("pandas not available, using csv-module fallback.")
        run_with_csv(run_dir, paper_avgs)


if __name__ == "__main__":
    main()
