"""
Grid search utility for DecisionTreeClassifier.

Loops over a predefined parameter grid, evaluates accuracy on the iris
train/test split provided in dataset/, and logs results to stdout and file.
"""

from itertools import product
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from decision_tree import DecisionTreeClassifier


DATA_DIR = Path(__file__).parent / "dataset"
LOG_PATH = Path(__file__).parent / "output" / "tune_log.txt"


def load_iris_split():
    train_path = DATA_DIR / "iris_train.csv"
    test_path = DATA_DIR / "iris_test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Missing iris_train.csv or iris_test.csv under dataset/")

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    feature_names = list(df_train.columns[:-1])
    X_train = df_train[feature_names].to_numpy(dtype=float)
    y_train = df_train["label"].to_numpy(dtype=int)
    X_test = df_test[feature_names].to_numpy(dtype=float)
    y_test = df_test["label"].to_numpy(dtype=int)
    return X_train, y_train, X_test, y_test


def format_result(idx: int, params: Dict[str, Any], acc: float, depth: int, leaves: int) -> str:
    param_str = ", ".join(f"{k}={v}" for k, v in params.items())
    return f"[{idx:03d}] acc={acc:.4f} depth={depth:<3d} leaves={leaves:<3d} | {param_str}"


def main():
    X_train, y_train, X_test, y_test = load_iris_split()

    param_grid = {
        "criterion": ["info_gain", "info_gain_ratio", "gini", "error_rate"],
        "splitter": ["best", "random"],
        "max_depth": [None, 6, 4],
        "min_samples_split": [2, 4, 8],
        "min_impurity_split": [0.0, 1e-3],
        "max_features": [None, "sqrt", 2],
    }

    grid_keys = list(param_grid.keys())
    grid_values = [param_grid[k] for k in grid_keys]

    results = []
    for combo in product(*grid_values):
        params = dict(zip(grid_keys, combo))
        clf = DecisionTreeClassifier(**params, random_state=0)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = (preds == y_test).mean()
        depth = getattr(clf, "tree_depth", -1)
        leaves = getattr(clf, "tree_leaf_num", -1)
        results.append((params, acc, depth, leaves))

    # sort by accuracy desc, then shallower depth (ascending), fewer leaves (ascending)
    results.sort(key=lambda x: (-x[1], x[2], x[3]))

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("w", encoding="utf-8") as f:
        for idx, (params, acc, depth, leaves) in enumerate(results, start=1):
            line = format_result(idx, params, acc, depth, leaves)
            print(line)
            f.write(line + "\n")

    print(f"\nTop 5 configurations:")
    for idx, (params, acc, depth, leaves) in enumerate(results[:5], start=1):
        print(format_result(idx, params, acc, depth, leaves))
    print(f"\nFull log saved to {LOG_PATH}")


if __name__ == "__main__":
    main()
