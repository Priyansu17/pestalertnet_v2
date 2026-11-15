"""Frozen outbreak-risk inference for the PestAlertNet best checkpoint."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .features import load_feature_manifest, EXPECTED_FEATURE_NAMES


def _validate_shape_and_finite(X: np.ndarray) -> np.ndarray:
    """Validate input shape and finite values; return float32 [B, 21, 25]."""
    x = np.asarray(X, dtype=np.float32)

    if x.ndim == 2 and x.shape == (21, 25):
        x = x.reshape(1, 21, 25)
    elif x.ndim != 3 or x.shape[1:] != (21, 25):
        raise ValueError(
            "X must have shape (21, 25) or (N, 21, 25)."
        )

    if not np.isfinite(x).all():
        raise ValueError("X must not contain NaN or Inf values.")

    return x


def predict_outbreak(
    X: np.ndarray,
    *,
    model_path: str | Path | None = None,
    device: str | None = None,
) -> np.ndarray:
    """Return outbreak probabilities with shape (N,)."""
    x = _validate_shape_and_finite(X)

    manifest = load_feature_manifest()
    if tuple(manifest.feature_names) != tuple(EXPECTED_FEATURE_NAMES):
        raise RuntimeError(
            "Frozen feature order does not match the approved Phase 5 contract."
        )

    # from .evaluate import predict_proba as _eval_predict_proba

    # # return _eval_predict_proba(
    # #     x,
    # #     model_path=model_path,
    # #     device=device,
    # #     threshold=None,
    # # )
    # return _eval_predict_proba(
    # x,
    # model_path=model_path,
    # device=device,
    # )
    import torch
    from .checkpoint import load_best_model


    if device is None:
        device_obj = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device_obj = torch.device(device)

    model, _ = load_best_model(
        model_path=model_path,
        device=device_obj,
    )
    model.eval()

    with torch.no_grad():
        logits = model(torch.from_numpy(x).to(device_obj))
        probabilities = torch.sigmoid(logits)

    result = probabilities.detach().cpu().numpy().astype(np.float32)

    if result.shape != (x.shape[0],):
        raise RuntimeError(
            f"Expected probabilities of shape ({x.shape[0]},), "
            f"got {result.shape}"
        )

    return result


__all__ = ["predict_outbreak"]
