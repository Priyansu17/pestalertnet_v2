"""Frozen-checkpoint loading for Phase 5.

Exposes:
- get_checkpoint(...)
- load_best_model(...)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch

# Root of the PestAlertNet project (two levels above this file).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pt"


class CheckpointConfigError(ValueError):
    """Raised when a stored checkpoint config does not match the frozen spec."""


def get_checkpoint(
    *,
    model_path: Path | None = None,
    device: str | None = None,
    map_location: str | None = None,
) -> Dict[str, Any]:
    """Load and validate the frozen best checkpoint.

    Returns the full checkpoint dict (metadata + state_dict), but does NOT
    reconstruct the model. Use :func:`load_best_model` for the model.
    """
    if model_path is None:
        model_path = MODEL_PATH
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Frozen checkpoint not found: {model_path}\n"
            f"Expected this file to exist and be readable. "
            f"Run `python -m phase5.checkpoint` once to confirm the path."
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Explicit weights_only=False for compatibility with the saved Phase 4 checkpoint.
    ckpt: Dict[str, Any] = torch.load(
        model_path,
        map_location=map_location or device,
        weights_only=False,
    )

    if not isinstance(ckpt, dict):
        raise TypeError(f"Checkpoint must be a dict, got {type(ckpt).__name__}")

    return ckpt


def load_best_model(
    *,
    model_path: Path | None = None,
    device: str | None = None,
    map_location: str | None = None,
) -> tuple["HybridOutbreakModel", Dict[str, Any]]:
    """Return a fresh HybridOutbreakModel in eval mode.

    The model is rebuilt from checkpoint metadata and the state dict is loaded
    with strict=True. The checkpoint is never modified or saved.
    """
    from src.model.hybrid_outbreak_model import HybridOutbreakModel

    ckpt = get_checkpoint(
        model_path=model_path,
        device=device,
        map_location=map_location,
    )

    # required = {
    #     "input_dim",
    #     "seq_len",
    #     "hidden_size",
    #     "num_layers",
    #     "num_heads",
    #     "d_model",
    #     "dim_feedforward",
    #     "num_transformer_layers",
    #     "mlp_hidden",
    #     "threshold",
    # }
    # missing = required - set(ckpt.keys())
    # if missing:
    #     raise CheckpointConfigError(
    #         f"Checkpoint metadata is missing required keys: {sorted(missing)}"
    #     )

    # cfg = {
    #     "input_dim": int(ckpt["input_dim"]),
    #     "seq_len": int(ckpt["seq_len"]),
    #     "hidden_size": int(ckpt["hidden_size"]),
    #     "num_layers": int(ckpt["num_layers"]),
    #     "num_heads": int(ckpt["num_heads"]),
    #     "ff_hidden": int(ckpt["dim_feedforward"]),
    #     "d_model": int(ckpt["d_model"]),
    #     "num_transformer_layers": int(ckpt["num_transformer_layers"]),
    #     "dim_feedforward": int(ckpt["dim_feedforward"]),
    #     "mlp_hidden": int(ckpt["mlp_hidden"]),
    #     "dropout": 0.1,  # verified frozen dropout, not present in the checkpoint
    #     "threshold": float(ckpt.get("threshold", 0.5)),
    # }

    # # Freeze configuration invariants.
    # if cfg["input_dim"] != 25 or cfg["seq_len"] != 21:
    #     raise CheckpointConfigError(
    #         f"Checkpoint config must match the frozen 21x25 feature manifest. Got "
    #         f"input_dim={cfg['input_dim']}, seq_len={cfg['seq_len']}"
    #     )
    # if cfg["num_layers"] != 2:
    #     raise CheckpointConfigError(
    #         f"Expected a 2-layer LSTM/Transformer, got num_layers={cfg['num_layers']}"
    #     )
    # if cfg["num_heads"] != 4:
    #     raise CheckpointConfigError(
    #         f"Expected 4 attention heads, got num_heads={cfg['num_heads']}"
    #     )
    # if cfg["ff_hidden"] != 256:
    #     raise CheckpointConfigError(
    #         f"Expected transformer FFN dimension 256, got dim_feedforward={cfg['ff_hidden']}"
    #     )
    # if cfg["num_transformer_layers"] != 2:
    #     raise CheckpointConfigError(
    #         f"Expected a 2-layer Transformer encoder, got num_transformer_layers={cfg['num_transformer_layers']}"
    #     )
    # if cfg["d_model"] != 256:
    #     raise CheckpointConfigError(
    #         f"Expected d_model=256, got d_model={cfg['d_model']}"
    #     )

    # model = HybridOutbreakModel(
    #     input_dim=cfg["input_dim"],
    #     seq_len=cfg["seq_len"],
    #     hidden_size=cfg["hidden_size"],
    #     num_layers=cfg["num_layers"],
    #     num_heads=cfg["num_heads"],
    #     ff_hidden=cfg["ff_hidden"],
    #     dropout=cfg["dropout"],
    # )
    arch = ckpt["model_config"]

    input_dim = int(arch["input_dim"])
    seq_len = int(arch["seq_len"])
    hidden_size = int(arch["hidden_size"])
    num_layers = int(arch["num_lstm_layers"])
    num_heads = int(arch["num_heads"])
    d_model = int(arch["d_model"])
    ff_hidden = int(arch["dim_feedforward"])
    num_transformer_layers = int(arch["num_transformer_layers"])
    mlp_hidden = int(arch["mlp_hidden"])
    threshold = float(ckpt.get("threshold", 0.5))

    # Freeze configuration invariants.
    if input_dim != 25 or seq_len != 21:
        raise CheckpointConfigError(
            f"Checkpoint config must match the frozen 21x25 feature manifest. Got "
            f"input_dim={input_dim}, seq_len={seq_len}"
        )
    if num_layers != 2:
        raise CheckpointConfigError(
            f"Expected a 2-layer LSTM/Transformer, got num_layers={num_layers}"
        )
    if num_heads != 4:
        raise CheckpointConfigError(
            f"Expected 4 attention heads, got num_heads={num_heads}"
        )
    if ff_hidden != 256:
        raise CheckpointConfigError(
            f"Expected transformer FFN dimension 256, got dim_feedforward={ff_hidden}"
        )
    if num_transformer_layers != 2:
        raise CheckpointConfigError(
            f"Expected a 2-layer Transformer encoder, got num_transformer_layers={num_transformer_layers}"
        )
    if d_model != 256:
        raise CheckpointConfigError(
            f"Expected d_model=256, got d_model={d_model}"
        )
    if hidden_size != 128:
        raise CheckpointConfigError(
            f"checkpoint hidden_size must be 128 (frozen), got hidden_size={hidden_size}"
        )
    if mlp_hidden != 128:
        raise CheckpointConfigError(
            f"checkpoint mlp_hidden must be 128 (frozen), got mlp_hidden={mlp_hidden}"
        )

    model = HybridOutbreakModel(
        input_dim=input_dim,
        seq_len=seq_len,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_hidden=ff_hidden,
        dropout=0.1,
    )


    # Explicit frozen-config validation for this Phase 5 load path.
    # if cfg["hidden_size"] != 128:
    #     raise CheckpointConfigError(
    #         f"checkpoint hidden_size must be 128 (frozen), got hidden_size={cfg['hidden_size']}"
    #     )
    # if cfg["mlp_hidden"] != 128:
    #     raise CheckpointConfigError(
    #         f"checkpoint mlp_hidden must be 128 (frozen), got mlp_hidden={cfg['mlp_hidden']}"
    #     )
    # if cfg["dim_feedforward"] != 256:
    #     raise CheckpointConfigError(
    #         f"checkpoint dim_feedforward must be 256 (frozen), got dim_feedforward={cfg['dim_feedforward']}"
    #     )
    if hidden_size != 128:
        raise CheckpointConfigError(
            f"checkpoint hidden_size must be 128 (frozen), got hidden_size={hidden_size}"
        )
    if mlp_hidden != 128:
        raise CheckpointConfigError(
            f"checkpoint mlp_hidden must be 128 (frozen), got mlp_hidden={mlp_hidden}"
        )
    # if ff_hidden != 256:
    #     raise CheckpointConfigError(
    #         f"checkpoint dim_feedforward must be 256 (frozen), got dim_feedforward={ff_hidden}"
    #     )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()

    state = ckpt["model_state_dict"]
    missing_keys, unexpected_keys = model.load_state_dict(state, strict=True)
    if missing_keys or unexpected_keys:
        raise CheckpointConfigError(
            f"strict load produced missing_keys={missing_keys or ''} "
            f"unexpected_keys={unexpected_keys or ''}"
        )

    return model, ckpt


__all__ = [
    "CheckpointConfigError",
    "get_checkpoint",
    "load_best_model",
]
