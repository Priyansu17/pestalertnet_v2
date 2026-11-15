
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from phase5.evaluate import list_test_split
from phase5.checkpoint import load_best_model
from phase5.explain import (
    integrated_gradients_attr,
    aggregate_attribution,
    feature_names,
)

BATCH_SIZE = 16
N_STEPS = 50
SEED = 42


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    checkpoint_path = ROOT / "models" / "best_model.pt"
    output_root = ROOT / "outputs"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = output_root / f"ig_full_test_{run_id}"
    out.mkdir(parents=True, exist_ok=False)

    X, y = list_test_split()
    names = feature_names()

    if len(X) != len(y) or X.shape[1:] != (21, 25):
        raise RuntimeError(f"Unexpected test shapes: X={X.shape}, y={y.shape}")
    if len(names) != 25:
        raise RuntimeError(f"Expected 25 feature names, got {len(names)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_best_model(device=device)
    model.eval()

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(y))
    X, y = X[order], y[order]

    # Accumulate feature-level absolute attributions by class.
    sums = {0: np.zeros(25, dtype=np.float64),
            1: np.zeros(25, dtype=np.float64)}
    counts = {0: 0, 1: 0}
    grand_sum = np.zeros(25, dtype=np.float64)
    completed = 0

    for start in range(0, len(y), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(y))
        xb = X[start:end]
        yb = y[start:end]

        attr = integrated_gradients_attr(
            model,
            xb,
            baseline=None,
            n_steps=N_STEPS,
            device=device,
        )
        if attr.shape != (len(xb), 21, 25):
            raise RuntimeError(f"Unexpected attribution shape: {attr.shape}")
        if not np.isfinite(attr).all():
            raise RuntimeError(f"Non-finite attribution in batch {start}:{end}")

        # Per sample, sum absolute attribution over the 21 input days.
        per_sample = aggregate_attribution(attr)
        grand_sum += per_sample.sum(axis=0)

        for label in (0, 1):
            mask = yb == label
            if mask.any():
                sums[label] += per_sample[mask].sum(axis=0)
                counts[label] += int(mask.sum())

        completed += len(xb)
        if completed % 256 == 0 or completed == len(y):
            print(f"Processed {completed}/{len(y)} samples", flush=True)

        del attr, per_sample
        if device == "cuda":
            torch.cuda.empty_cache()

    overall = grand_sum / len(y)
    rows = []
    for i, name in enumerate(names):
        rows.append({
            "feature": name,
            "mean_abs_attribution_all": float(overall[i]),
            "mean_abs_attribution_negative": (
                float(sums[0][i] / counts[0]) if counts[0] else None
            ),
            "mean_abs_attribution_positive": (
                float(sums[1][i] / counts[1]) if counts[1] else None
            ),
        })

    df = pd.DataFrame(rows).sort_values(
        "mean_abs_attribution_all", ascending=False
    )
    df.to_csv(out / "feature_attributions.csv", index=False)

    plot_df = df.head(15).sort_values("mean_abs_attribution_all")
    plt.figure(figsize=(10, 7))
    plt.barh(
        plot_df["feature"],
        plot_df["mean_abs_attribution_all"],
    )
    plt.xlabel("Mean absolute attribution (sum over 21 input days)")
    plt.ylabel("Feature")
    plt.title("PestAlertNet: Integrated Gradients on frozen test set")
    plt.tight_layout()
    plt.savefig(out / "ig_top15.png", dpi=200)
    plt.close()

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "test_samples": int(len(y)),
        "positive_samples": int((y == 1).sum()),
        "negative_samples": int((y == 0).sum()),
        "input_shape_per_sample": [21, 25],
        "device": device,
        "batch_size": BATCH_SIZE,
        "integrated_gradients_steps": N_STEPS,
        "baseline": "all-zero tensor in frozen feature space",
        "aggregation": "sum absolute attribution over 21 time steps; average over samples",
        "seed": SEED,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_epoch": ckpt.get("epoch"),
        "threshold": float(ckpt.get("threshold", 0.5)),
        "feature_names": names,
        "class_counts": {
            "negative": counts[0],
            "positive": counts[1],
        },
        "interpretation_limit": (
            "Magnitude only; does not indicate direction or causation. "
            "This is a global test-set attribution summary, not a causal analysis."
        ),
    }
    (out / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nAnalysis completed.")
    print("Output directory:", out)
    print("Top 10 features:")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
