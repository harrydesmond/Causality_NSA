"""Create mock calibration figure from analyse_mocks.py results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_TRUNC_LIMIT = 14


def _resolve_default_input(trunc_limit: int) -> Path:
    return (Path.cwd() / f"penalty_results_trunc{trunc_limit}.txt").resolve()


def _resolve_paper_plots_dir() -> Path:
    out_dir = Path("Plots_paper").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def read_results(path: Path, trunc_limit: int):
    rows = []
    penalties = []
    malformed = 0

    with open(path, "r") as f:
        _ = f.readline()  # header
        for line in f:
            parts = line.strip().split()
            if len(parts) != 7:
                malformed += 1
                continue
            _, penalty, seed, prec, rec, f1, trunc_lim = parts
            try:
                penalty_i = int(penalty)
                seed_i = int(seed)
                prec_f = float(prec)
                rec_f = float(rec)
                f1_f = float(f1)
                trunc_i = int(trunc_lim)
            except ValueError:
                malformed += 1
                continue

            if trunc_i != trunc_limit:
                continue

            rows.append((penalty_i, seed_i, prec_f, rec_f, f1_f))
            penalties.append(penalty_i)

    if not rows:
        raise RuntimeError(f"No valid rows parsed from {path}")

    if malformed > 0:
        print(f"Skipped {malformed} malformed rows from {path}", flush=True)

    return rows, sorted(set(penalties))


def aggregate(rows, penalties):
    stats = {p: {"prec": [], "rec": [], "f1": []} for p in penalties}
    for penalty, _, prec, rec, f1 in rows:
        stats[penalty]["prec"].append(prec)
        stats[penalty]["rec"].append(rec)
        stats[penalty]["f1"].append(f1)

    agg = {p: {} for p in penalties}
    for p in penalties:
        for key in ("prec", "rec", "f1"):
            arr = np.asarray(stats[p][key], dtype=float)
            agg[p][key] = (
                float(np.mean(arr)),
                float(np.percentile(arr, 16)),
                float(np.percentile(arr, 84)),
            )
    return agg


def make_paper_plot(penalties, agg, out_pdf: Path):
    means_prec = [agg[p]["prec"][0] for p in penalties]
    means_rec = [agg[p]["rec"][0] for p in penalties]
    means_f1 = [agg[p]["f1"][0] for p in penalties]

    lo_prec = [agg[p]["prec"][1] for p in penalties]
    hi_prec = [agg[p]["prec"][2] for p in penalties]
    lo_rec = [agg[p]["rec"][1] for p in penalties]
    hi_rec = [agg[p]["rec"][2] for p in penalties]
    lo_f1 = [agg[p]["f1"][1] for p in penalties]
    hi_f1 = [agg[p]["f1"][2] for p in penalties]

    plt.figure(figsize=(8, 5))
    plt.plot(penalties, means_prec, color="tab:blue", label="Precision", linewidth=2.5)
    plt.fill_between(penalties, lo_prec, hi_prec, color="tab:blue", alpha=0.15, linewidth=0)
    plt.plot(penalties, means_rec, color="tab:orange", label="Recall", linewidth=2.5)
    plt.fill_between(penalties, lo_rec, hi_rec, color="tab:orange", alpha=0.15, linewidth=0)
    plt.plot(penalties, means_f1, color="tab:green", label="F1 Score", linewidth=2.5)
    plt.fill_between(penalties, lo_f1, hi_f1, color="tab:green", alpha=0.15, linewidth=0)
    plt.xlabel("Penalty Discount", fontsize=16)
    plt.ylabel("Score", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(min(penalties), max(penalties))
    plt.ylim(0.5, 1.0)
    plt.legend(fontsize=14, ncol=3)
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the mock-performance figure from metrics results."
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to penalty_results_trunc<truncation_limit>.txt",
    )
    parser.add_argument(
        "--truncation_limit",
        type=int,
        default=DEFAULT_TRUNC_LIMIT,
        help="Truncation limit to select from results (default: 14, paper setting)",
    )
    args = parser.parse_args()

    trunc_limit = args.truncation_limit
    if trunc_limit <= 0:
        raise ValueError("--truncation_limit must be a positive integer")

    if trunc_limit != DEFAULT_TRUNC_LIMIT:
        print("Note: truncation_limit=14 was used in the paper.", flush=True)

    if args.input is None:
        input_path = _resolve_default_input(trunc_limit)
    else:
        input_path = Path(args.input).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input results file not found: {input_path}")

    paper_plots_dir = _resolve_paper_plots_dir()
    rows, penalties = read_results(input_path, trunc_limit)
    agg = aggregate(rows, penalties)

    if trunc_limit == DEFAULT_TRUNC_LIMIT:
        out_pdf = paper_plots_dir / "mocks.pdf"
    else:
        out_pdf = paper_plots_dir / f"mocks_trunc{trunc_limit}.pdf"
    make_paper_plot(penalties, agg, out_pdf)

    print(f"Loaded {len(rows)} valid rows from {input_path}", flush=True)
    print(f"Saved paper figure: {out_pdf}", flush=True)


if __name__ == "__main__":
    main()
