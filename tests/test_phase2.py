"""Phase 2 (L2) build verification tests.

Requires: stdlib + numpy + pyyaml.
"""
import csv
import datetime as dt
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

FEATURES = [
    "T2M", "RH2M", "PRECTOTCORR", "WS2M",
    "w_mask_T2M", "w_mask_RH2M", "w_mask_PRECTOTCORR", "w_mask_WS2M",
    "pest_surveyed", "pest_event", "pest_present",
    "T2M_roll7_mean", "T2M_roll21_mean", "RH2M_roll7_mean", "RH2M_roll21_mean",
    "WS2M_roll7_mean", "rain_sum7", "rain_sum21", "rain_days21", "t2m_anomaly",
    "event_count_7d", "event_count_21d", "survey_count_21d",
    "days_since_survey", "days_since_event",
]
BIN_IDX = list(range(4, 11))
CONT_IDX = [i for i in range(25) if i not in BIN_IDX]

FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


def main():
    z = np.load(PROC / "sequences.npz", allow_pickle=False)
    X, y = z["X"], z["y"]
    cells, dates, split = z["cell"], z["date"], z["split"]

    # T1 shapes/dtypes/values
    check("T1a X shape [N,21,25]", X.shape[1:] == (21, 25), str(X.shape))
    check("T1b X float32", X.dtype == np.float32)
    check("T1c y values {0,1}", set(np.unique(y)) <= {0, 1})
    check("T1d no NaN/Inf in X", bool(np.isfinite(X.astype(np.float64)).all()))
    check("T1e y int8", y.dtype == np.int8)

    # T2 split membership vs FIXED calendar
    d = np.array([dt.date.fromisoformat(s) for s in dates])
    tr, va, te = split == 0, split == 1, split == 2
    tr_end = dt.date(2025, 6, 5)
    va_start, va_end = dt.date(2025, 7, 11), dt.date(2026, 1, 5)
    te_start, te_end = dt.date(2026, 2, 10), dt.date(2026, 9, 8)
    check("T2a train dates <= 2025-06-05", bool(d[tr].max() <= tr_end), str(d[tr].max()))
    check("T2b val dates in [07-11,01-05]",
          bool(d[va].min() >= va_start and d[va].max() <= va_end),
          f"{d[va].min()}..{d[va].max()}")
    check("T2c test dates in [02-10,09-08]",
          bool(d[te].min() >= te_start and d[te].max() <= te_end),
          f"{d[te].min()}..{d[te].max()}")

    # T3 embargo gaps + zones empty
    check("T3a train->val gap >= 28d",
          (d[va].min() - d[tr].max()).days >= 28,
          str((d[va].min() - d[tr].max()).days) + "d")
    check("T3b val->test gap >= 28d",
          (d[te].min() - d[va].max()).days >= 28,
          str((d[te].min() - d[va].max()).days) + "d")
    z1 = ((d > tr_end) & (d < va_start)).sum()
    z2 = ((d > va_end) & (d < te_start)).sum()
    check("T3c embargo zones empty", int(z1) == 0 and int(z2) == 0,
          f"z1={int(z1)} z2={int(z2)}")
    check("T3d no unassigned split", set(np.unique(split)) <= {0, 1, 2})

    # T4 chronological order: non-decreasing (ties allowed)
    check("T4 non-decreasing dates", bool((d[1:] >= d[:-1]).all()))

    # T5 guardrail
    for name, m in (("train", tr), ("val", va), ("test", te)):
        p = int(y[m].sum())
        check(f"T5 {name} positives >= 30", p >= 30, str(p))

    # T6 label spot-recompute vs event_days.csv (50 samples, seeded)
    ev = {}
    with open(PROC / "event_days.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            ev.setdefault((float(r["lat"]), float(r["lon"])), set()).add(
                dt.date.fromisoformat(r["date"]))
    rng = np.random.default_rng(42)
    picks = rng.choice(len(y), 50, replace=False)
    bad = 0
    for i in picks:
        la, lo = str(cells[i]).split("_")
        c = (float(la), float(lo))
        t = dt.date.fromisoformat(str(dates[i]))
        exp = int(any(t < e <= t + dt.timedelta(days=7) for e in ev.get(c, set())))
        if exp != int(y[i]):
            bad += 1
    check("T6 label recompute (50 samples)", bad == 0, f"mismatches={bad}")

    # T7 scaler checks
    stats = json.load(open(PROC / "scaler_stats.json"))
    check("T7a manifest feature order", stats["features"] == FEATURES)
    # T7b train-only provenance: the build-side S8 assertion is the enforcement
    # point; these are metadata + sanity checks, NOT mathematical proof.
    check("T7b-1 scaler_stats declares train_only + row accounting",
          stats.get("train_only") is True
          and stats.get("n_train_rows") == int(tr.sum())
          and stats.get("n_nontrain_rows") == int((~tr).sum()))
    check("T7b-2 stored mean/std length == 18",
          len(stats["mean"]) == 18 and len(stats["std"]) == 18)
    ztr = X[tr][:, :, CONT_IDX].astype(np.float64).reshape(-1, len(CONT_IDX))
    check("T7b-3 train z-dist ~ (0,1) (sanity only)",
          bool(np.abs(ztr.mean(0)).max() < 0.05
               and (np.abs(ztr.std(0) - 1) < 0.05).all()))
    check("T7c scaled |z| <= 100", float(np.abs(X[:, :, CONT_IDX]).max()) <= 100,
          f"max={float(np.abs(X[:, :, CONT_IDX]).max()):.2f}")
    binvals = np.unique(X[:, :, BIN_IDX])
    check("T7d binary channels unscaled {0,1}", set(np.unique(binvals)) <= {0.0, 1.0})

    # T8 manifest consistency
    import yaml
    man = yaml.safe_load(open(PROC / "feature_manifest.yaml"))
    check("T8a manifest features", man["features"] == FEATURES)
    check("T8b window/horizon/embargo",
          man["input_days"] == 21 and man["horizon"] == 7 and man["embargo_days"] == 28)
    check("T8c fixed B1/B2", man["split_boundaries_fixed"] ==
          {"B1": "2025-06-12", "B2": "2026-01-12"})
    check("T8d manifest ranges (corrected calendar)",
          man["val_range"] == ["2025-07-11", "2026-01-05"] and
          man["test_range"] == ["2026-02-10", "2026-09-08"] and
          man["train_range"] == ["2023-01-21", "2025-06-05"])

    print(f"\n{len(FAILS)} failure(s)")
    raise SystemExit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
