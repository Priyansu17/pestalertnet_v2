"""Frozen test-split evaluation for Phase 5.

Provides:
- list_test_split(...)
- predict_proba(...)
- evaluate_test_split(...)

These utilities operate ONLY on frozen Phase 2/4 artifacts and the frozen
checkpoint. No retraining, no checkpoint modification, no output files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

from .checkpoint import load_best_model

# Frozen evaluation invariants.
TEST_SPLIT = 2


def _project_root() -> Path:
    """Return the PestAlertNet project root (two levels above this file)."""
    from pathlib import Path

    return Path(__file__).resolve().parents[2]


def _device() -> torch.device:
    """CUDA when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def list_test_split(
    sequences_path: str | Path | None = None,
    return_indices: bool = False,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the frozen test split without modifying any artifact.

    Args:
        sequences_path:
            Optional path to `sequences.npz`. Defaults to the frozen
            `data/processed/sequences.npz`.
        return_indices:
            If True, also return the integer array of test sample indices
            (positions inside the full artifact).

    Returns:
        X_test: np.ndarray, shape [N_test, 21, 25], float32
        y_test: np.ndarray, shape [N_test], int8, values in {0,1}
        If return_indices is True:
            idx: np.ndarray, shape [N_test], int64

    Raises
    ------
    FileNotFoundError
        If the frozen sequences artifact is missing.
    ValueError
        If the test split is empty, X shape is wrong, or y has values outside {0,1}.
    """
    if sequences_path is None:
        candidate = _project_root() / "data" / "processed" / "sequences.npz"
        if not candidate.exists():
            raise FileNotFoundError(
                "Frozen test split artifact not found at:\n"
                f"  {candidate}\n"
                f"Expected the Phase 2 artifact data/processed/sequences.npz to exist."
            )
        sequences_path = candidate
    else:
        p = Path(sequences_path)
        if not p.is_absolute():
            p = _project_root() / p
        if not p.exists():
            raise FileNotFoundError(f"Sequences artifact not found: {p}")
    p = Path(sequences_path)
    data = np.load(p, mmap_mode="r")
    try:
        X = data["X"].astype(np.float32, copy=False)
        y = data["y"].astype(np.int8, copy=False)
        split = data["split"].astype(np.int8, copy=False)
    finally:
        if hasattr(data, "close"):
            data.close()

    if X.ndim != 3 or X.shape[1] != 21 or X.shape[2] != 25:
        raise ValueError(
            f"expected X shape [N, 21, 25], got {X.shape}. "
            f"Phase 2 feature order and input window are frozen."
        )

    if not np.all(np.isin(y, [0, 1])):
        bad = np.unique(y[~np.isin(y, [0, 1])])
        raise ValueError(f"y contains non-binary values: {bad}")

    mask = split == TEST_SPLIT
    if int(mask.sum()) == 0:
        raise ValueError("Frozen test split is empty (no split == 2 samples).")

    idx = np.flatnonzero(mask)
    X_test = X[mask]
    y_test = y[mask]

    if return_indices:
        return X_test, y_test, idx
    return X_test, y_test


def predict_proba(
    X_test: np.ndarray,
    *,
    model_path: str | Path | None = None,
    device: torch.device | None = None,
) -> np.ndarray:
    """Return frozen-test probabilities for the saved best checkpoint.

    Uses:
      checkpoint.load_best_model(...)
      model.eval()
      torch.no_grad()
      sigmoid on logits (if forward returns raw logits)

    No retraining, fine-tuning, optimization, or checkpoint mutation.
    """
    X = np.asarray(X_test, dtype=np.float32)
    # if X.ndim != 3 or X.shape[1] != 21 or X.shape[2] != 25:
    #     raise ValueError(
    #         f"expected X shape [N, 21, 25], got {X.shape}. "
    #         f"Phase 2 input window and order are frozen."
    #     )
    if X.ndim != 3 or X.shape[1:] != (21, 25):
        raise ValueError(
            f"expected X shape [N, 21, 25], got {X.shape}. "
            "Phase 2 input window and order are frozen."
        )
    if X.shape[0] == 0:
        raise ValueError("X must contain at least one sample.")
    if not np.isfinite(X).all():
        raise ValueError("X must not contain NaN or Inf values.")

    model, ckpt = load_best_model(model_path=model_path, device=device)
    model.eval()

    if device is None:
        device = _device()
    elif isinstance(device, str):
        device = torch.device(device)

    with torch.no_grad():
        logits = model(torch.from_numpy(X).to(device))
        probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)

    if probs.ndim != 1 or probs.shape[0] != X.shape[0]:
        raise RuntimeError(
            "predict_proba produced unexpected output shape: "
            f"{getattr(probs, 'shape', None)}"
        )

    return probs


def evaluate_test_split(
    sequences_path: str | Path | None = None,
    *,
    model_path: str | Path | None = None,
    threshold: float | None = None,
    return_labels: bool = False,
) -> Dict[str, Any]:
    """Evaluate the frozen Phase 2 TEST split with the saved best checkpoint.

    Uses:
      - Frozen test split (only split == 2).
      - Frozen checkpoint (models/best_model.pt) via checkpoint.load_best_model.
      - BCE-with-logits loss with the checkpoint's stored pos_weight where present.
      - Decision threshold: effective_threshold = ckpt threshold unless an
        explicit threshold override is supplied.
      - ROC-AUC from sklearn.metrics.roc_auc_score.

    Returns a structured dict:
      {
        "n_test": int,
        "loss": float | None,
        "accuracy": float,
        "precision": float,
        "recall": float,
        "f1": float,
        "auc": float,
        "threshold": float,
        "confusion_matrix": {"tp": int, "fp": int, "tn": int, "fn": int},
        "true_labels": np.ndarray | None,
        "probabilities": np.ndarray | None,
      }
    """
    from sklearn.metrics import roc_auc_score

    X_test, y_test = list_test_split(sequences_path=sequences_path)

    # model, ckpt = load_best_model(model_path=model_path)
    # device = _device()
    device = _device()
    model, ckpt = load_best_model(model_path=model_path, device=device)
    model.eval()

    effective_threshold = (
        #float(ckpt["threshold"]) if threshold is None else float(threshold)
        float(ckpt.get("threshold", 0.5)) if threshold is None else float(threshold)
    )

    with torch.no_grad():
        logits = model(torch.from_numpy(X_test).to(device))

    # pos_weight = (
    #     float(ckpt["pos_weight"]) if ckpt.get("pos_weight") is not None else 0.0
    # )
    
    pos_weight = ckpt.get("pos_weight")
    
    # if pos_weight > 0.0:
    #     pos_weight_t = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    #     criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
    # else:
    #     criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
    if pos_weight is not None and float(pos_weight) > 0.0:
        pos_weight_t = torch.tensor(
            [float(pos_weight)], dtype=torch.float32, device=device
        )
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
        loss_note = None
    else:
        criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
        loss_note = (
            "Checkpoint has no positive-class weight; loss uses unweighted BCE "
            "and is not directly comparable to the weighted training loss."
        )
    # model.eval()
    with torch.no_grad():
        loss = criterion(
            logits,
            torch.from_numpy(y_test)
            .to(device=device, dtype=torch.float32),
        ).item()

    probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
    preds = (probs >= effective_threshold).astype(np.int8)

    tp = int(np.sum((y_test == 1) & (preds == 1)))
    fp = int(np.sum((y_test == 0) & (preds == 1)))
    tn = int(np.sum((y_test == 0) & (preds == 0)))
    fn = int(np.sum((y_test == 1) & (preds == 0)))

    accuracy = float((tp + tn) / max(tp + fp + tn + fn, 1))
    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    f1 = float(2 * precision * recall / max(precision + recall, 1e-9))

    if int((y_test == 0).sum()) == 0 or int((y_test == 1).sum()) == 0:
        auc = float("nan")
    else:
        try:
            auc = float(roc_auc_score(y_test, probs))
        except ValueError:
            auc = float("nan")
        if not (0.0 <= auc <= 1.0) or np.isnan(auc):
            auc = float("nan")

    result = {
        "n_test": int(y_test.shape[0]),
        "loss": float(loss) if not np.isnan(loss) else None,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
        "threshold": float(effective_threshold),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "loss_note": loss_note,
    }

    if return_labels:
        result["true_labels"] = np.asarray(y_test, dtype=np.int8)
        result["probabilities"] = probs

    return result


__all__ = [
    "list_test_split",
    "predict_proba",
    "evaluate_test_split",
]