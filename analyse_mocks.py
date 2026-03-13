"""Run FCIT on all generated mock datasets and write paper calibration metrics.

Default paper setup:
- truncation_limit = 14
- p-value threshold = 0.01
- penalty sweep across 30 values from 1 to 200

Writes penalty_results_trunc<truncation_limit>.txt for use in plot_mock_results.py
"""

from __future__ import annotations

import ast
import argparse
import contextlib
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytetrad.tools.TetradSearch as ts
from mpi4py import MPI


DEFAULT_TRUNC_LIMIT = 14
PVAL_THRESHOLD = 0.01
SILENCE_FCIT = os.environ.get("SILENCE_FCIT", "1") not in ("0", "false", "False")
PENALTIES = np.unique(np.linspace(1, 200, 30).round().astype(int))

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


@contextlib.contextmanager
def silence_stdout_stderr(active: bool = True):
    if not active:
        yield
        return

    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass

    devnull = open(os.devnull, "w")
    try:
        old_stdout_fd = os.dup(1)
        old_stderr_fd = os.dup(2)
    except OSError:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            yield
        return

    try:
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")
        try:
            yield
        finally:
            try:
                sys.stdout.close()
                sys.stderr.close()
            except Exception:
                pass
            sys.stdout, sys.stderr = old_stdout, old_stderr
    finally:
        try:
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)
            devnull.close()
        except Exception:
            pass


def get_edges_from_graph(graph) -> set[tuple[str, str]]:
    edges: set[tuple[str, str]] = set()
    for edge in graph.getEdges():
        node1 = edge.getNode1().getName()
        node2 = edge.getNode2().getName()
        ep1 = str(edge.getEndpoint1())
        ep2 = str(edge.getEndpoint2())
        if ep1 == "TAIL" and ep2 == "ARROW":
            edges.add((node1, node2))
        elif ep1 == "CIRCLE" and ep2 == "ARROW":
            edges.add((node1, node2))
        elif ep1 == "CIRCLE" and ep2 == "CIRCLE":
            edges.add((node1, node2))
            edges.add((node2, node1))
        elif ep1 == "ARROW" and ep2 == "CIRCLE":
            edges.add((node2, node1))
    return edges


def get_edges_from_dict(graph_dict: dict[str, list[str]]) -> set[tuple[str, str]]:
    return {(parent, child) for parent, children in graph_dict.items() for child in children}


def calculate_precision_recall_f1(reconstructed_graph, ground_truth_graph: dict[str, list[str]]):
    reconstructed_edges = get_edges_from_graph(reconstructed_graph)
    ground_truth_edges = get_edges_from_dict(ground_truth_graph)

    tp = len(reconstructed_edges & ground_truth_edges)
    fp = len(reconstructed_edges - ground_truth_edges)
    fn = len(ground_truth_edges - reconstructed_edges)

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def load_ground_truths(path: Path) -> list[dict[str, list[str]]]:
    graph_gts: list[dict[str, list[str]]] = []
    with open(path, "r") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                d = ast.literal_eval(line)
            except Exception as exc:
                print(f"[WARN] Skipping malformed CPDAG line {lineno}: {exc}", flush=True)
                continue
            graph_gts.append(d)
    return graph_gts


def run_for_penalty_and_seed(
    penalty_discount: int,
    seed_id: int,
    seed_to_gt: dict[int, dict[str, list[str]]],
    trunc_limit: int,
):
    filename_npy = Path(f"simulated_data/data_{seed_id}.npy")
    filename_txt = Path(f"simulated_data/data_{seed_id}.txt")
    if not filename_npy.exists() or not filename_txt.exists():
        raise FileNotFoundError(f"Missing mock data files for seed {seed_id}")

    data = np.load(filename_npy)
    var_names = np.genfromtxt(filename_txt, delimiter="\t", max_rows=1, dtype=str)
    data_df = pd.DataFrame(data, columns=var_names)

    search = ts.TetradSearch(data_df)
    search.set_verbose(False)
    search.use_basis_function_lrt(truncation_limit=trunc_limit, alpha=PVAL_THRESHOLD)
    search.use_basis_function_bic(
        truncation_limit=trunc_limit,
        penalty_discount=penalty_discount,
    )

    with silence_stdout_stderr(SILENCE_FCIT):
        search.run_fcit()

    graph_java = search.get_java()
    graph_gt = seed_to_gt[seed_id]
    return calculate_precision_recall_f1(graph_java, graph_gt)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run FCIT over all mock datasets and write calibration metrics."
    )
    parser.add_argument(
        "--truncation_limit",
        type=int,
        default=DEFAULT_TRUNC_LIMIT,
        help="FCIT truncation limit (default: 14, paper setting)",
    )
    args = parser.parse_args()

    trunc_limit = args.truncation_limit
    if trunc_limit <= 0:
        raise ValueError("--truncation_limit must be a positive integer")

    gt_file = Path("simulated_data/cpdag_dicts_by_rank.txt")
    if not gt_file.exists():
        raise FileNotFoundError(
            "Missing simulated_data/cpdag_dicts_by_rank.txt. Run make_mocks.py first."
        )

    graph_gts_all = load_ground_truths(gt_file)
    seed_to_gt = {idx: gt for idx, gt in enumerate(graph_gts_all)}
    seeds = sorted(seed_to_gt.keys())

    if rank == 0:
        print(f"Using truncation limit {trunc_limit}", flush=True)
        if trunc_limit != DEFAULT_TRUNC_LIMIT:
            print(
                "Note: truncation_limit=14 was used in the paper.",
                flush=True,
            )
        print(f"Using p-value threshold {PVAL_THRESHOLD}", flush=True)
        print(f"Loaded {len(seeds)} ground-truth mock graphs", flush=True)
        print(
            f"Running with {size} ranks, {len(PENALTIES)} penalties, {len(seeds)} datasets",
            flush=True,
        )

    jobs = [(p, seed) for p in PENALTIES for seed in seeds]
    my_jobs = jobs[rank::size]

    rank_start = time.time()
    my_results: list[tuple[int, int, float, float, float]] = []
    for idx, (penalty, seed_id) in enumerate(my_jobs, start=1):
        precision, recall, f1 = run_for_penalty_and_seed(
            penalty, seed_id, seed_to_gt, trunc_limit
        )
        my_results.append((penalty, seed_id, precision, recall, f1))
        if idx == 1 or idx % 10 == 0 or idx == len(my_jobs):
            print(
                f"[Progress] Rank {rank}: {idx}/{len(my_jobs)} jobs, elapsed {time.time() - rank_start:.1f}s",
                flush=True,
            )

    # Write rank-local results first to avoid large gather payloads on rank 0.
    tmp_dir = Path("simulated_data") / ".tmp_analyse_results"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    rank_tmp_path = tmp_dir / f"rank_{rank}.txt"
    with open(rank_tmp_path, "w") as f:
        for penalty, seed_id, precision, recall, f1 in my_results:
            f.write(f"{penalty} {seed_id} {precision} {recall} {f1}\n")

    comm.Barrier()

    if rank == 0:
        flat_results = []
        for r in range(size):
            p = tmp_dir / f"rank_{r}.txt"
            if not p.exists():
                continue
            with open(p, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    penalty, seed_id, precision, recall, f1 = parts
                    flat_results.append(
                        (int(penalty), int(seed_id), float(precision), float(recall), float(f1))
                    )

        flat_results.sort(key=lambda x: (x[0], x[1]))

        output_file = Path.cwd() / f"penalty_results_trunc{trunc_limit}.txt"
        with open(output_file, "w") as f:
            f.write("Rank Penalty_Seed Precision Recall F1_Score TruncLimit\n")
            for penalty, seed_id, precision, recall, f1 in flat_results:
                # Rank is not meaningful after gather; keep column for compatibility.
                f.write(
                    f"0 {penalty} {seed_id} {precision} {recall} {f1} {trunc_limit}\n"
                )

        print(f"Wrote {len(flat_results)} rows to {output_file}", flush=True)

        # Best-effort cleanup of temporary shard files.
        for r in range(size):
            p = tmp_dir / f"rank_{r}.txt"
            try:
                if p.exists():
                    p.unlink()
            except OSError:
                pass
        try:
            tmp_dir.rmdir()
        except OSError:
            pass


if __name__ == "__main__":
    main()
