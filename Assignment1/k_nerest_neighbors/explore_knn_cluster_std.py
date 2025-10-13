"""
Exploration script for studying CLUSTER_STD vs best k in kNN.

Usage (repo root):
    python Assignment1/k_nerest_neighbors/explore_knn_cluster_std.py

Outputs are written to:
    Assignment1/k_nerest_neighbors/output/explore_cluster_std/std_<value>/
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from data_generate import (
    generate_and_save,
    load_prepared_dataset,
    RANDOM_STATE,
    N_SAMPLES,
    N_CLASSES,
    TEST_SIZE,
    VAL_SIZE,
)
from knn_student import select_k_by_validation, knn_predict
from viz_knn import plot_k_curve, plot_decision_boundary_multi


BASE_DIR = Path(__file__).parent
OUTPUT_ROOT = BASE_DIR / "output" / "explore_cluster_std"
DATA_ROOT = BASE_DIR / "input_knn"


@dataclass
class ExperimentResult:
    cluster_std: float
    ks: List[int]
    accs: List[float]
    best_k: int
    val_best_acc: float
    test_acc: float
    curve_path: Path
    boundary_path: Path


def run_single_experiment(cluster_std: float, ks: List[int]) -> ExperimentResult:
    # regenerate dataset with desired cluster_std (other params fixed)
    generate_and_save(
        data_dir=str(DATA_ROOT),
        n_samples=N_SAMPLES,
        n_classes=N_CLASSES,
        cluster_std=cluster_std,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        force=True,
    )

    X_train, y_train, X_val, y_val, X_test, y_test = load_prepared_dataset(str(DATA_ROOT))
    best_k, accs = select_k_by_validation(
        X_train, y_train, X_val, y_val, ks, metric="l2", mode="no_loops"
    )

    X_trv = np.vstack([X_train, X_val])
    y_trv = np.hstack([y_train, y_val])
    y_test_pred = knn_predict(X_test, X_trv, y_trv, best_k, metric="l2", mode="no_loops")
    test_acc = float((y_test_pred == y_test).mean())
    val_best = float(accs[ks.index(best_k)])

    run_dir = OUTPUT_ROOT / f"std_{cluster_std:.2f}"
    run_dir.mkdir(parents=True, exist_ok=True)
    curve_path = run_dir / "knn_k_curve.png"
    boundary_path = run_dir / "knn_boundary_grid.png"

    plot_k_curve(ks, accs, str(curve_path))

    def predict_fn_for_k(k):
        return lambda Xq: knn_predict(Xq, X_trv, y_trv, k, metric="l2", mode="no_loops")

    ks_panel = sorted(set(ks + [best_k]))
    plot_decision_boundary_multi(
        predict_fn_for_k,
        X_train,
        y_train,
        X_test,
        y_test,
        ks=ks_panel,
        out_path=str(boundary_path),
        grid_n=200,
        batch_size=4096,
    )

    return ExperimentResult(
        cluster_std=cluster_std,
        ks=ks,
        accs=[float(a) for a in accs],
        best_k=best_k,
        val_best_acc=val_best,
        test_acc=test_acc,
        curve_path=curve_path,
        boundary_path=boundary_path,
    )


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    ks = [1, 3, 5, 7, 9, 11, 13, 15, 17]
    cluster_std_values = [1.0, 2.0, 3.0, 4.0, 5.0]

    results: List[ExperimentResult] = []
    for std in cluster_std_values:
        print(f"\n=== Experiment: CLUSTER_STD={std:.2f} ===")
        res = run_single_experiment(std, ks)
        results.append(res)
        print(
            f"best_k={res.best_k} | val_acc={res.val_best_acc:.4f} | "
            f"test_acc={res.test_acc:.4f}"
        )
        print(f"curve saved to: {res.curve_path}")
        print(f"boundary saved to: {res.boundary_path}")

    summary_path = OUTPUT_ROOT / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("CLUSTER_STD, best_k, val_acc(best_k), test_acc\n")
        for res in results:
            f.write(
                f"{res.cluster_std:.2f}, {res.best_k}, "
                f"{res.val_best_acc:.4f}, {res.test_acc:.4f}\n"
            )
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
