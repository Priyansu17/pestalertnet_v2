"""Explainability for the frozen PestAlertNet best checkpoint.

Uses Captum Integrated Gradients to attribute a batch of frozen checkpoint
predictions to the 21 time steps and 25 frozen features. No artifacts or
plots are created.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np
import torch

from captum.attr import IntegratedGradients

from .features import load_feature_manifest

# Frozen evaluation invariants.
INPUT_TIMESTEPS = 21
INPUT_DIM = 25
FROZEN_SAMPLE_SHAPE = (INPUT_TIMESTEPS, INPUT_DIM)

__all__ = [
    "integrated_gradients_attr",
    "aggregate_attribution",
    "feature_names",
]


def integrated_gradients_attr(
    model: torch.nn.Module,
    X: Union[np.ndarray, torch.Tensor],
    baseline: Optional[Union[np.ndarray, torch.Tensor]] = None,
    n_steps: int = 50,
    device: Optional[str] = None,
) -> np.ndarray:
    """Integrated Gradients attribution for frozen best-model logits.

    The frozen model returns one raw logit per sample with shape [N].
    We therefore use a small forward wrapper so Captum receives exactly one
    scalar logit per sample, and call Captum with target=None.

    Args:
        model: Frozen HybridOutbreakModel, already in eval() mode.
        X: [N, 21, 25] or [21, 25] float32/64 input.
        baseline: [N, 21, 25] or [21, 25]. Defaults to an all-zero tensor in
            the frozen Phase 2 feature space.
        n_steps: IG steps.
        device: optional device string.

    Returns:
        attributions: np.ndarray [N, 21, 25].
    """
    if not isinstance(X, torch.Tensor):
        X_tensor = torch.as_tensor(X, dtype=torch.float32)
    else:
        X_tensor = X.detach().float()

    if X_tensor.dim() == 2:
        X_tensor = X_tensor.unsqueeze(0)

    if X_tensor.shape[1:] != (INPUT_TIMESTEPS, INPUT_DIM):
        raise ValueError(
            f"expected [N,21,25] or [21,25], got {tuple(X_tensor.shape)}"
        )

    if not np.isfinite(X_tensor.detach().cpu().numpy()).all():
        raise ValueError("Input contains NaN/inf values.")

    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    model_device = next(model.parameters()).device
    target_device = torch.device(device) if device is not None else model_device
    X_tensor = X_tensor.to(target_device)
    if baseline is None:
        baseline_tensor = torch.zeros_like(X_tensor)
    else:
        baseline_tensor = torch.as_tensor(
            baseline,
            dtype=X_tensor.dtype,
            device=X_tensor.device,
        )

        if baseline_tensor.shape == (INPUT_TIMESTEPS, INPUT_DIM):
            baseline_tensor = baseline_tensor.unsqueeze(0).expand(
                X_tensor.shape[0], -1, -1
            )

        if baseline_tensor.shape != X_tensor.shape:
            raise ValueError(
                f"baseline shape after expansion must match X shape; "
                f"got {tuple(baseline_tensor.shape)}, expected {tuple(X_tensor.shape)}"
            )

        if not torch.isfinite(baseline_tensor).all():
            raise ValueError("baseline contains NaN or Inf values.")

    def forward_logits(x: torch.Tensor) -> torch.Tensor:
        return model(x)

    # ig = IntegratedGradients(forward_logits)
    # attributions = ig.attribute(
    #     X_tensor,
    #     baselines=baseline_tensor,
    #     n_steps=n_steps,
    # )
    ig = IntegratedGradients(forward_logits)

    # cuDNN RNN backward can fail in eval mode during input attribution.
    # Disable cuDNN only for this gradient calculation.
    with torch.backends.cudnn.flags(enabled=False):
      attributions = ig.attribute(
        X_tensor,
        baselines=baseline_tensor,
        n_steps=n_steps,
    )

    if attributions.requires_grad:
        attributions = attributions.detach()

    attr = attributions.detach().cpu().numpy()

    if attr.ndim == 2:
        attr = attr.reshape(1, INPUT_TIMESTEPS, INPUT_DIM)
    elif attr.ndim != 3 or attr.shape[1:] != (INPUT_TIMESTEPS, INPUT_DIM):
        raise RuntimeError(f"unexpected IG output shape: {attr.shape}")

    return attr.astype(np.float64)


def aggregate_attribution(
    attributions: np.ndarray,
    axis_time: int = -2,
    keepdim: bool = False,
) -> np.ndarray:
    """Aggregate absolute attribution over time to one value per feature.

    Args:
        attributions: [N, 21, 25].
        axis_time: axis to aggregate (default -2 = time).
        keepdim: if True, keep time dimension as 1.

    Returns:
        [N, 25] or [1, 25] per sample/feature.
    """
    if attributions.ndim != 3 or attributions.shape[1:] != (INPUT_TIMESTEPS, INPUT_DIM):
        raise ValueError(f"expected [N,21,25], got {attributions.shape}")

    weights = np.abs(attributions)
    kept = np.sum(weights, axis=axis_time, keepdims=keepdim)

    if kept.ndim == 3 and kept.shape[1] == 1:
        return kept[:, 0, :]  # [N, 25]
    if kept.ndim == 2:
        return kept
    return kept


def feature_names() -> List[str]:
    """Return the exact 25 frozen feature names from the Phase 2 manifest.

    Never a hard-coded reordered list.
    """
    from .features import load_feature_manifest

    return list(load_feature_manifest().feature_names)
