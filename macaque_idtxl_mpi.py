#!/usr/bin/env python3
"""MPI-enabled multivariate TE analysis for the macaque fMRI benchmark.

The script mirrors the exploratory notebook `macaque_idtxl_multivariate_te.ipynb`
but wraps it in a command-line utility that can be launched on HPC systems
using MPI (e.g. via `mpiexec`, `srun`, or `python -m mpi4py.futures`).
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import math
np.math = math
import pandas as pd

try:
    from mpi4py import MPI  # type: ignore
except ImportError:  # pragma: no cover
    MPI = None  # type: ignore

from idtxl.data import Data
from idtxl.multivariate_te import MultivariateTE


def load_macaque_data_replications(
    path: Path,
    *,
    samples_per_replication: int,
    n_replications: int,
    var_subset: Optional[Union[int, Sequence[Union[int, str]]]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Load BOLD signal, z-score channels, and reshape into replications."""
    df = pd.read_csv(path, sep=r"\s+|\t", engine="python")
    cols = sorted(
        [c for c in df.columns if str(c).startswith("X")], key=lambda name: int(name[1:])
    )
    if isinstance(var_subset, int):
        cols = cols[:var_subset]
    elif isinstance(var_subset, Iterable) and not isinstance(var_subset, (str, bytes)):
        target_names = {str(v) for v in var_subset}
        cols = [c for c in cols if c in target_names]

    total_samples = samples_per_replication * n_replications
    df = df.iloc[:total_samples]
    data = df[cols].to_numpy(dtype=float)
    data = (data - data.mean(axis=0, keepdims=True)) / (
        data.std(axis=0, keepdims=True) + 1e-12
    )
    data = data.reshape((n_replications, samples_per_replication, len(cols)))
    return data, cols


def parse_truth_edges(path: Path, var_names: Sequence[str]) -> Set[Tuple[int, int]]:
    """Parse ground-truth graph into 0-based edge index tuples."""
    name_to_idx = {name: idx for idx, name in enumerate(var_names)}
    edges: Set[Tuple[int, int]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if "-->" not in raw:
                continue
            line = raw.replace(".", "")
            left, right = line.split("-->")
            src = left.strip().split()[-1]
            dst = right.strip().split()[0]
            if src in name_to_idx and dst in name_to_idx:
                edges.add((name_to_idx[src], name_to_idx[dst]))
    return edges


def benjamini_hochberg(pvalues: np.ndarray, alpha: float) -> np.ndarray:
    """Classic BH-FDR control for independent tests."""
    pvalues = np.asarray(pvalues, dtype=float)
    m = pvalues.size
    order = np.argsort(pvalues)
    sorted_p = pvalues[order]
    thresholds = alpha * (np.arange(1, m + 1) / m)
    below = sorted_p <= thresholds
    if not below.any():
        return np.zeros_like(pvalues, dtype=bool)
    k = np.max(np.nonzero(below))
    mask = np.zeros_like(pvalues, dtype=bool)
    mask[order[: k + 1]] = True
    return mask


def run_idtxl_te_replications(data_array: np.ndarray, settings: dict) -> Any:
    """Execute IDTxl multivariate TE using replication-aware dimensional order."""
    data_obj = Data(data_array, dim_order="rsp", normalise=False)
    analysis = MultivariateTE()
    return analysis.analyse_network(settings=settings, data=data_obj)


def collect_edge_tests(results: Any) -> pd.DataFrame:
    """Extract per-edge lag/TE/p-value summaries from an IDTxl result object."""
    records = []
    for target in results.targets_analysed:
        view = results.get_single_target(target, fdr=False)
        selected = view.get("selected_vars_sources") or []
        if not selected:
            continue
        pvals = view.get("selected_sources_pval")
        te_vals = view.get("selected_sources_te")
        current_sample = view.get("current_value")[1]
        for idx, (source, sample_idx) in enumerate(selected):
            lag = current_sample - sample_idx
            pval = float(pvals[idx]) if pvals is not None else np.nan
            te_val = float(te_vals[idx]) if te_vals is not None else np.nan
            records.append(
                {
                    "source": int(source),
                    "target": int(target),
                    "lag": int(lag),
                    "pvalue": pval,
                    "te": te_val,
                }
            )
    edge_df = pd.DataFrame(records)
    if edge_df.empty:
        edge_df["significant"] = False
        edge_df["bh_rank"] = np.nan
        return edge_df
    edge_df = edge_df.sort_values("pvalue", na_position="last")
    aggregated = edge_df.groupby(["source", "target"], as_index=False).first()
    aggregated = aggregated[aggregated["pvalue"].notna()].copy()
    if aggregated.empty:
        aggregated["significant"] = False
        aggregated["bh_rank"] = np.nan
        return aggregated
    return aggregated


def mark_significant_edges(edge_df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Apply BH-FDR and annotate the edge table."""
    if edge_df.empty or edge_df["pvalue"].isna().all():
        edge_df["significant"] = False
        edge_df["bh_rank"] = np.nan
        return edge_df
    mask = benjamini_hochberg(edge_df["pvalue"].to_numpy(), alpha)
    edge_df = edge_df.copy()
    edge_df["significant"] = mask
    edge_df["bh_rank"] = np.arange(1, edge_df.shape[0] + 1)
    return edge_df


def evaluate_predictions(
    edge_df: pd.DataFrame, truth_edges: Set[Tuple[int, int]], n_nodes: int
) -> dict:
    """Compute precision/recall metrics for directed edges and skeleton."""
    pred = np.zeros((n_nodes, n_nodes), dtype=bool)
    lag_map: dict = {}
    for row in edge_df.itertuples(index=False):
        if getattr(row, "significant", False):
            pred[row.source, row.target] = True
            lag_map[(row.source, row.target)] = row.lag

    truth_matrix = np.zeros_like(pred)
    for src, dst in truth_edges:
        truth_matrix[src, dst] = True

    def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        return prec, rec, f1

    tp = int((pred & truth_matrix).sum())
    fp = int((pred & ~truth_matrix).sum())
    fn = int((~pred & truth_matrix).sum())
    prec, rec, f1 = prf(tp, fp, fn)

    pred_skel = np.logical_or(pred, pred.T)
    truth_skel = np.logical_or(truth_matrix, truth_matrix.T)
    tp_s = fp_s = fn_s = 0
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            pred_val = pred_skel[i, j]
            truth_val = truth_skel[i, j]
            if pred_val and truth_val:
                tp_s += 1
            elif pred_val and not truth_val:
                fp_s += 1
            elif (not pred_val) and truth_val:
                fn_s += 1
    prec_s, rec_s, f1_s = prf(tp_s, fp_s, fn_s)

    lag_counter: Counter = Counter()
    for (src, dst), lag in lag_map.items():
        if (src, dst) in truth_edges:
            lag_counter["lag<=1" if lag <= 1 else "lag>1"] += 1

    two_pred = []
    two_true = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if pred[i, j] and pred[j, i]:
                two_pred.append((i, j))
            if truth_matrix[i, j] and truth_matrix[j, i]:
                two_true.append((i, j))
    two_tp = [edge for edge in two_pred if edge in two_true]
    two_fp = [edge for edge in two_pred if edge not in two_true]
    two_fn = [edge for edge in two_true if edge not in two_pred]

    return {
        "dir": {"tp": tp, "fp": fp, "fn": fn, "prec": prec, "rec": rec, "f1": f1},
        "skel": {"tp": tp_s, "fp": fp_s, "fn": fn_s, "prec": prec_s, "rec": rec_s, "f1": f1_s},
        "lag_counter": lag_counter,
        "two_cycles": {"tp": len(two_tp), "fp": len(two_fp), "fn": len(two_fn)},
        "pred_matrix": pred,
    }


def print_summary(label: str, metrics: dict, run_cfg: dict, var_names: Sequence[str], alpha: float) -> None:
    """Human-readable recap of the run outcome."""
    dir_m = metrics["dir"]
    sk_m = metrics["skel"]
    lag_counts = metrics["lag_counter"]
    cycles = metrics["two_cycles"]
    print(f"=== IDTxl mTE ({label}) vs. Ground Truth ===")
    print(
        f"Samples={run_cfg['samples']}  Vars={len(var_names)}  tau_max={run_cfg['tau_max']}  "
        f"alpha={alpha:.3f} (BH)"
    )
    print("Directed edges (pair-level union):")
    print(f"  TP={dir_m['tp']}  FP={dir_m['fp']}  FN={dir_m['fn']}")
    print(f"  Precision={dir_m['prec']:.3f}  Recall={dir_m['rec']:.3f}  F1={dir_m['f1']:.3f}")
    print("Skeleton (undirected union):")
    print(f"  TP={sk_m['tp']}  FP={sk_m['fp']}  FN={sk_m['fn']}")
    print(f"  Precision={sk_m['prec']:.3f}  Recall={sk_m['rec']:.3f}  F1={sk_m['f1']:.3f}")
    print("Where were TPs found?")
    print(f"  lag <= 1 : {lag_counts.get('lag<=1', 0)}")
    print(f"  lag > 1  : {lag_counts.get('lag>1', 0)}")
    print("2-cycles (mutual edges):")
    print(f"  TP={cycles['tp']}  FP={cycles['fp']}  FN={cycles['fn']}")


def parse_var_subset(raw: Optional[str]) -> Optional[Union[int, List[str]]]:
    """Translate CLI option into either an int or list of names."""
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        items = [token.strip() for token in raw.split(",") if token.strip()]
        return items or None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run multivariate TE on macaque fMRI using MPI-enabled IDTxl."
    )
    parser.add_argument("--bold-path", type=Path, required=True, help="Path to concatenated BOLD time series.")
    parser.add_argument(
        "--truth-path",
        type=Path,
        required=True,
        help="Path to ground-truth edge list (e.g. Macaque_SmallDegree_graph.txt).",
    )
    parser.add_argument(
        "--samples-per-replication",
        type=int,
        default=500,
        help="Samples per replication used to segment the time series (default: 500).",
    )
    parser.add_argument(
        "--n-replications",
        type=int,
        default=10,
        help="Number of replications to build from the concatenated time series (default: 10).",
    )
    parser.add_argument(
        "--var-subset",
        type=str,
        default=None,
        help="Limit variables by count (int) or comma-separated variable names.",
    )
    parser.add_argument(
        "--tau-max",
        type=int,
        default=2,
        help="Maximum source/target lag to consider (default: 2).",
    )
    parser.add_argument(
        "--tau-min",
        type=int,
        default=0,
        help="Minimum source lag to consider (default: 0).",
    )
    parser.add_argument(
        "--n-perm",
        type=int,
        default=200,
        help="Number of permutations for all permutation-based tests (default: 200).",
    )
    parser.add_argument(
        "--cmi-estimator",
        type=str,
        default="JidtKraskovCMI",
        help="Conditional MI estimator to use (default: JidtKraskovCMI).",
    )
    parser.add_argument(
        "--kraskov-k",
        type=int,
        default=6,
        help="k-nearest neighbours for Kraskov estimators (default: 6).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Alpha level for Benjamini-Hochberg FDR post-processing (default: 0.05).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=0,
        help="Number of MPI workers to hand to IDTxl (0 disables MPI, default: 0).",
    )
    parser.add_argument(
        "--analysis-label",
        type=str,
        default="macaque_idtxl_mpi",
        help="Label for output artefacts (default: macaque_idtxl_mpi).",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("result_figures"),
        help="Directory for CSV/NumPy outputs (default: result_figures).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Turn on IDTxl internal verbosity.",
    )
    args = parser.parse_args()

    if MPI is not None:
        assert MPI.COMM_WORLD.Get_rank() == 0, "Run this script on the MPI root rank only."

    if args.max_workers < 0:
        raise ValueError("--max-workers must be non-negative.")

    var_subset = parse_var_subset(args.var_subset)
    data_array, var_names = load_macaque_data_replications(
        args.bold_path,
        samples_per_replication=args.samples_per_replication,
        n_replications=args.n_replications,
        var_subset=var_subset,
    )
    truth_edges = parse_truth_edges(args.truth_path, var_names)

    settings = {
        "cmi_estimator": args.cmi_estimator,
        "kraskov_k": args.kraskov_k,
        "min_lag_sources": args.tau_min,
        "max_lag_sources": args.tau_max,
        "max_lag_target": args.tau_max,
        "verbose": args.verbose,
    }

    for key in ["n_perm_max_stat", "n_perm_min_stat", "n_perm_omnibus", "n_perm_max_seq"]:
        settings[key] = args.n_perm

    if args.max_workers:
        settings["MPI"] = True
        settings["max_workers"] = args.max_workers

    results_obj = run_idtxl_te_replications(data_array, settings)
    edge_tests = collect_edge_tests(results_obj)
    edge_tests = mark_significant_edges(edge_tests, alpha=args.alpha)

    metrics = evaluate_predictions(edge_tests, truth_edges, len(var_names))
    run_cfg = {
        "label": args.analysis_label,
        "samples": args.samples_per_replication,
        "tau_max": args.tau_max,
    }
    print_summary(args.analysis_label, metrics, run_cfg, var_names, args.alpha)

    edges_named = edge_tests.copy()
    if not edges_named.empty:
        edges_named["source_name"] = edges_named["source"].map(lambda idx: var_names[idx])
        edges_named["target_name"] = edges_named["target"].map(lambda idx: var_names[idx])

    summary_row = {
        "label": args.analysis_label,
        "samples": args.samples_per_replication,
        "vars": len(var_names),
        "tau_max": args.tau_max,
        "n_perm": args.n_perm,
        "dir_TP": metrics["dir"]["tp"],
        "dir_FP": metrics["dir"]["fp"],
        "dir_FN": metrics["dir"]["fn"],
        "dir_Precision": metrics["dir"]["prec"],
        "dir_Recall": metrics["dir"]["rec"],
        "dir_F1": metrics["dir"]["f1"],
        "skel_TP": metrics["skel"]["tp"],
        "skel_FP": metrics["skel"]["fp"],
        "skel_FN": metrics["skel"]["fn"],
        "skel_Precision": metrics["skel"]["prec"],
        "skel_Recall": metrics["skel"]["rec"],
        "skel_F1": metrics["skel"]["f1"],
        "tp_lag_le1": metrics["lag_counter"].get("lag<=1", 0),
        "tp_lag_gt1": metrics["lag_counter"].get("lag>1", 0),
        "cycle_TP": metrics["two_cycles"]["tp"],
        "cycle_FP": metrics["two_cycles"]["fp"],
        "cycle_FN": metrics["two_cycles"]["fn"],
        "edges_tested": int(edge_tests.shape[0]),
        "edges_significant": int(metrics["pred_matrix"].sum()),
    }
    summary_df = pd.DataFrame([summary_row])

    args.result_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.result_dir / f"{args.analysis_label}_summary.csv"
    edges_path = args.result_dir / f"{args.analysis_label}_edges.csv"
    matrix_path = args.result_dir / f"{args.analysis_label}_adjacency.npy"

    summary_df.to_csv(summary_path, index=False)
    edges_named.to_csv(edges_path, index=False)
    np.save(matrix_path, metrics["pred_matrix"])
    print(f"Saved summary to {summary_path}")
    print(f"Saved edge table to {edges_path}")
    print(f"Saved adjacency matrix to {matrix_path}")


if __name__ == "__main__":
    main()
