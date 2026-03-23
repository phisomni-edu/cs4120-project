from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    hamming_loss,
    precision_recall_fscore_support,
)


ArrayLike = Iterable[Iterable[int]] | np.ndarray | pd.DataFrame


def _to_binary_numpy(values: ArrayLike, name: str) -> np.ndarray:
    """Convert array-like inputs to a 2D integer numpy array."""
    if isinstance(values, pd.DataFrame):
        arr = values.to_numpy()
    else:
        arr = np.asarray(values)

    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D (n_samples, n_labels). Got shape {arr.shape}.")

    # Coerce boolean/float/int predictions into {0, 1} ints.
    arr = (arr > 0).astype(int)
    return arr


def _validate_binary_inputs(y_true: np.ndarray, y_pred: np.ndarray, label_names: list[str]) -> None:
    """Validate shape agreement and label cardinality."""
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred shape mismatch: {y_true.shape} vs {y_pred.shape}."
        )

    if y_true.shape[1] != len(label_names):
        raise ValueError(
            "label_names length must match number of columns in y_true/y_pred. "
            f"Got len(label_names)={len(label_names)} and n_labels={y_true.shape[1]}."
        )


def compute_overall_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    zero_division: int = 0,
) -> Dict[str, float]:
    """
    Compute overall multilabel metrics.

    Returns:
    - accuracy: exact-match subset accuracy (for multilabel)
    - macro_f1: unweighted F1 across labels
    - micro_f1: globally aggregated F1
    - hamming_loss: average label-wise error
    """
    y_true_np = _to_binary_numpy(y_true, "y_true")
    y_pred_np = _to_binary_numpy(y_pred, "y_pred")

    if y_true_np.shape != y_pred_np.shape:
        raise ValueError(
            f"y_true and y_pred shape mismatch: {y_true_np.shape} vs {y_pred_np.shape}."
        )

    accuracy = accuracy_score(y_true_np, y_pred_np)
    macro_f1 = precision_recall_fscore_support(
        y_true_np,
        y_pred_np,
        average="macro",
        zero_division=zero_division,
    )[2]
    micro_f1 = precision_recall_fscore_support(
        y_true_np,
        y_pred_np,
        average="micro",
        zero_division=zero_division,
    )[2]
    ham_loss = hamming_loss(y_true_np, y_pred_np)

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "hamming_loss": float(ham_loss),
    }


def compute_per_class_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    label_names: list[str],
    *,
    zero_division: int = 0,
) -> pd.DataFrame:
    """
    Compute per-class precision/recall/F1/support and confusion counts.

    Returns columns:
    emotion, precision, recall, f1, support, tp, fp, fn, tn
    """
    y_true_np = _to_binary_numpy(y_true, "y_true")
    y_pred_np = _to_binary_numpy(y_pred, "y_pred")
    _validate_binary_inputs(y_true_np, y_pred_np, label_names)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_np,
        y_pred_np,
        average=None,
        zero_division=zero_division,
    )

    per_class_rows = []
    for idx, emotion in enumerate(label_names):
        y_t = y_true_np[:, idx]
        y_p = y_pred_np[:, idx]

        tp = int(np.sum((y_t == 1) & (y_p == 1)))
        fp = int(np.sum((y_t == 0) & (y_p == 1)))
        fn = int(np.sum((y_t == 1) & (y_p == 0)))
        tn = int(np.sum((y_t == 0) & (y_p == 0)))

        per_class_rows.append(
            {
                "emotion": emotion,
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )

    return pd.DataFrame(per_class_rows)


def build_overall_report(
    *,
    method: str,
    data_fraction: float,
    seed: int,
    y_true: ArrayLike,
    y_pred: ArrayLike,
    zero_division: int = 0,
) -> pd.DataFrame:
    """Build one-row overall report with experiment metadata."""
    metrics = compute_overall_metrics(y_true, y_pred, zero_division=zero_division)
    row = {
        "method": method,
        "data_fraction": float(data_fraction),
        "seed": int(seed),
        **metrics,
    }
    return pd.DataFrame([row])


def build_per_class_report(
    *,
    method: str,
    data_fraction: float,
    seed: int,
    label_names: list[str],
    y_true: ArrayLike,
    y_pred: ArrayLike,
    zero_division: int = 0,
) -> pd.DataFrame:
    """
    Build per-class report with experiment metadata.

    Output format is compatible with project result aggregation:
    method, data_fraction, seed, emotion, f1, precision, recall
    (plus support/tp/fp/fn/tn for debugging/analysis)
    """
    df = compute_per_class_metrics(
        y_true,
        y_pred,
        label_names,
        zero_division=zero_division,
    )

    df.insert(0, "seed", int(seed))
    df.insert(0, "data_fraction", float(data_fraction))
    df.insert(0, "method", method)
    return df


def evaluate_run(
    *,
    method: str,
    data_fraction: float,
    seed: int,
    label_names: list[str],
    y_true: ArrayLike,
    y_pred: ArrayLike,
    zero_division: int = 0,
) -> Dict[str, Any]:
    """Run shared evaluation and return both overall and per-class reports."""
    y_true_np = _to_binary_numpy(y_true, "y_true")
    y_pred_np = _to_binary_numpy(y_pred, "y_pred")
    _validate_binary_inputs(y_true_np, y_pred_np, label_names)

    overall_df = build_overall_report(
        method=method,
        data_fraction=data_fraction,
        seed=seed,
        y_true=y_true_np,
        y_pred=y_pred_np,
        zero_division=zero_division,
    )
    per_class_df = build_per_class_report(
        method=method,
        data_fraction=data_fraction,
        seed=seed,
        label_names=label_names,
        y_true=y_true_np,
        y_pred=y_pred_np,
        zero_division=zero_division,
    )

    return {
        "overall": overall_df,
        "per_class": per_class_df,
    }


def save_results_csv(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Save a dataframe to CSV and return resolved path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path.resolve()


def save_json_summary(payload: Dict[str, Any], output_path: str | Path) -> Path:
    """Save JSON summary and return resolved path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return output_path.resolve()


def save_evaluation_outputs(
    evaluation: Dict[str, pd.DataFrame],
    *,
    method: str,
    output_dir: str | Path = "results",
) -> Dict[str, Path]:
    """
    Save shared evaluation outputs to standard paths.

    Writes:
    - {output_dir}/{method}_overall.csv
    - {output_dir}/{method}_per_class.csv
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    overall_path = save_results_csv(
        evaluation["overall"],
        output_dir / f"{method}_overall.csv",
    )
    per_class_path = save_results_csv(
        evaluation["per_class"],
        output_dir / f"{method}_per_class.csv",
    )

    return {
        "overall_csv": overall_path,
        "per_class_csv": per_class_path,
    }
