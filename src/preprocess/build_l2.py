"""Phase 2 (L2): build model-ready sequences per approved Revision 2 plan.

Reads only data/raw inputs; writes only data/processed + outputs/phase2_report.txt.
Label: D1(b) gregarious-form PRESENT events, D2(a) observed cell-days only,
D3(a) any-PRESENT-wins, D4 DUBIOUS/UNCLEAR excluded. 21-day input, 7-day forward
horizon, 28-day embargo, FIXED split boundaries B1/B2, train-only scaling.
"""
import csv
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"

RAMSES = RAW / "ramses_observations_global.csv"
POWER = RAW / "power_cells_global.csv"

START = dt.date(2023, 1, 1)
END = dt.date(2026, 9, 15)
INPUT_DAYS = 21
HORIZON = 7
EMBARGO = 28
B1 = dt.date(2025, 6, 12)   # FIXED (approved) - do not recompute
B2 = dt.date(2026, 1, 12)   # FIXED (approved) - do not recompute
TRIM_FIRST = START + dt.timedelta(days=INPUT_DAYS - 1)     # 2023-01-21
CENSOR_LAST = END - dt.timedelta(days=HORIZON)             # 2026-09-08
GREG_TYPES = {"BAND", "SWARM", "HOPPER GROUPS", "ADULT GROUPS"}
GREG_PHASE = {"GREGARIOUS", "T_CONGREGANS"}
SENTINEL = -999.0

FEATURES = [
    "T2M", "RH2M", "PRECTOTCORR", "WS2M",
    "w_mask_T2M", "w_mask_RH2M", "w_mask_PRECTOTCORR", "w_mask_WS2M",
    "pest_surveyed", "pest_event", "pest_present",
    "T2M_roll7_mean", "T2M_roll21_mean", "RH2M_roll7_mean", "RH2M_roll21_mean",
    "WS2M_roll7_mean", "rain_sum7", "rain_sum21", "rain_days21", "t2m_anomaly",
    "event_count_7d", "event_count_21d", "survey_count_21d",
    "days_since_survey", "days_since_event",
]
BIN_IDX = list(range(4, 11))                           # 7 binary channels
CONT_IDX = [i for i in range(25) if i not in BIN_IDX]  # 18 continuous

NDAYS = (END - START).days + 1               # 1354
DAY = dt.timedelta(days=1)


def snap(lat, lon):
    return (round(round(float(lat) / 0.5) * 0.5, 4),
            round(round(float(lon) / 0.625) * 0.625, 4))


def log(msg):
    with open(OUT / "phase2_report.txt", "a", encoding="utf-8") as fh:
        fh.write(msg + "\n")
    print(msg, flush=True)


def load_obs():
    survey = defaultdict(set)
    event = defaultdict(set)
    present = defaultdict(set)
    kept = excl = 0
    with open(RAMSES, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["Observation_Reliability"] == "DUBIOUS" or \
               r["Observation_Confirmation"] == "UNCLEAR":
                excl += 1
                continue
            kept += 1
            c = snap(r["Latitude"], r["Longitude"])
            d = dt.datetime.utcfromtimestamp(int(r["Obs_Date"]) / 1000).date()
            survey[c].add(d)
            if r["Locust_Presence"] == "PRESENT":
                present[c].add(d)
                greg = (r["Locust_Type"] in GREG_TYPES
                        or r["RAMSES_Locust_Type"] in GREG_TYPES
                        or r["Locust_Phase"] in GREG_PHASE
                        or "GROUPS" in (r["Behaviour"] or ""))
                if greg:
                    event[c].add(d)
    return survey, event, present, kept, excl


def load_weather(cells):
    """Stream POWER CSV once; return W[len(cells), 1354, 4] with NaN sentinels."""
    key2j = {(str(c[0]), str(c[1])): j for j, c in enumerate(cells)}
    W = np.full((len(cells), NDAYS, 4), np.nan, dtype=np.float64)
    seen = set()
    with open(POWER, newline="", encoding="utf-8") as fh:
        rd = csv.reader(fh)
        next(rd)
        for row in rd:
            j = key2j.get((row[0], row[1]))
            if j is None:
                continue
            i = (dt.date.fromisoformat(row[2]) - START).days
            for v in range(4):
                val = float(row[3 + v])
                W[j, i, v] = np.nan if val <= SENTINEL + 0.5 else val
            seen.add(j)
    return W, len(seen)

def build_samples(cells, survey, event, present, W):
    """S4/S5/S6: observed cell-day samples, causal features, 7-day labels."""
    X, ys, cs, ds = [], [], [], []
    ev = {c: {(d - START).days for d in s} for c, s in event.items()}
    sv = {c: {(d - START).days for d in s} for c, s in survey.items()}
    pr = {c: {(d - START).days for d in s} for c, s in present.items()}
    n_trim = n_cens = 0
    for j, c in enumerate(cells):
        evc, svc, prc = ev.get(c, set()), sv.get(c, set()), pr.get(c, set())
        for d in sorted(survey[c]):
            t = (d - START).days
            if d < TRIM_FIRST:
                n_trim += 1
                continue
            if d > CENSOR_LAST:
                n_cens += 1
                continue
            x = np.zeros((INPUT_DAYS, 25), dtype=np.float64)
            for k in range(INPUT_DAYS):
                i = t - 20 + k
                wv = W[j, i]
                x[k, 0:4] = np.nan_to_num(wv, nan=0.0)
                x[k, 4:8] = np.isnan(wv)
                x[k, 8] = 1.0 if i in svc else 0.0
                x[k, 9] = 1.0 if i in evc else 0.0
                x[k, 10] = 1.0 if i in prc else 0.0
                seg7 = W[j, max(0, i - 6):i + 1]
                seg21 = W[j, max(0, i - 20):i + 1]
                f = np.isfinite
                x[k, 11] = np.nanmean(seg7[:, 0]) if f(seg7[:, 0]).sum() >= 4 else np.nan
                x[k, 12] = np.nanmean(seg21[:, 0]) if f(seg21[:, 0]).sum() >= 11 else np.nan
                x[k, 13] = np.nanmean(seg7[:, 1]) if f(seg7[:, 1]).sum() >= 4 else np.nan
                x[k, 14] = np.nanmean(seg21[:, 1]) if f(seg21[:, 1]).sum() >= 11 else np.nan
                x[k, 15] = np.nanmean(seg7[:, 3]) if f(seg7[:, 3]).sum() >= 4 else np.nan
                r7 = np.nansum(seg7[:, 2]) if f(seg7[:, 2]).any() else np.nan
                r21 = np.nansum(seg21[:, 2]) if f(seg21[:, 2]).any() else np.nan
                x[k, 16] = np.log1p(r7) if np.isfinite(r7) else np.nan
                x[k, 17] = np.log1p(r21) if np.isfinite(r21) else np.nan
                x[k, 18] = float(f(seg21[:, 2] > 1.0).sum())
                x[k, 19] = x[k, 0] - x[k, 12] if (np.isfinite(x[k, 12]) and np.isfinite(x[k, 0])) else np.nan
                ev_w = [e for e in evc if t - 20 <= e <= t]
                sv_w = [e for e in svc if t - 20 <= e <= t]
                x[k, 20] = float(sum(1 for e in ev_w if t - 6 <= e <= t))
                x[k, 21] = float(len(ev_w))
                x[k, 22] = float(len(sv_w))
                last_sv = max((e for e in svc if e <= i), default=None)
                last_ev = max((e for e in evc if e <= i), default=None)
                x[k, 23] = 0.0 if last_sv is None else min(90, t - last_sv) / 90.0
                x[k, 24] = 0.0 if last_ev is None else min(90, t - last_ev) / 90.0
            y = 1 if any(t < e <= t + HORIZON for e in evc) else 0
            X.append(x.astype(np.float32))
            ys.append(y)
            cs.append(f"{c[0]}_{c[1]}")
            ds.append(d.isoformat())
    return np.stack(X), np.array(ys, dtype=np.int8), cs, ds, n_trim, n_cens



def main():
    log("=== Phase 2 build start ===")
    survey, event, present, kept, excl = load_obs()
    log(f"S1/S2 kept={kept} excluded={excl} event_cells={len(event)} "
        f"event_days_total={sum(len(s) for s in event.values())}")
    cells = sorted(survey.keys())
    W, n_cells_covered = load_weather(cells)
    log(f"S3 weather grid={W.shape} cells_covered={n_cells_covered} "
        f"sentinel_nans={int(np.isnan(W).sum())}")
    X, y, cs, ds, n_trim, n_cens = build_samples(cells, survey, event, present, W)
    log(f"S4 samples={len(X)} trimmed_left={n_trim} censored_right={n_cens}")
    log(f"S6 positives={int(y.sum())} rate={100 * y.mean():.2f}%")

    # S7: FIXED boundaries + 28-day embargo (Revision 2 calendar)
    date_arr = np.array([dt.date.fromisoformat(s) for s in ds])
    tr_end, va_start = dt.date(2025, 6, 5), dt.date(2025, 7, 11)
    va_end, te_start = dt.date(2026, 1, 5), dt.date(2026, 2, 10)
    is_tr = date_arr <= tr_end
    is_va = (date_arr >= va_start) & (date_arr <= va_end)
    is_te = (date_arr >= te_start) & (date_arr <= CENSOR_LAST)
    drop = ~(is_tr | is_va | is_te)
    split = np.full(len(X), -1, dtype=np.int8)
    split[is_tr], split[is_va], split[is_te] = 0, 1, 2
    keep = ~drop
    X, y = X[keep], y[keep]
    cs = [c for c, k in zip(cs, keep) if k]
    ds = [d for d, k in zip(ds, keep) if k]
    date_arr, split = date_arr[keep], split[keep]
    log(f"S7 embargo_dropped={int(drop.sum())} train={int((split==0).sum())} "
        f"val={int((split==1).sum())} test={int((split==2).sum())}")
    for name, s in (("train", 0), ("val", 1), ("test", 2)):
        m = split == s
        p = int(y[m].sum())
        log(f"  {name}: n={int(m.sum())} pos={p} "
            f"rate={100 * p / max(int(m.sum()), 1):.2f}%")
        if p < 30:
            raise RuntimeError(f"GUARDRAIL FAIL: {name} positives={p} < 30")

    # S8: train-only scaling of the 18 continuous channels (4-10 stay binary)
    # AUTHORITATIVE train-only enforcement: stats computed exactly once from
    # cont_raw[split == 0] BEFORE any standardization; verified below.
    cont_raw = X[:, :, CONT_IDX].astype(np.float64)
    tr_mask = split == 0
    tr_flat = cont_raw[tr_mask].reshape(-1, len(CONT_IDX))
    mean = np.nanmean(tr_flat, axis=0)
    std = np.nanstd(tr_flat, axis=0)
    std = np.where(std == 0, 1.0, std)
    fill = np.where(np.isfinite(mean), mean, 0.0)
    # self-check: recomputation from the same train-only mask reproduces stats
    re_std_raw = np.nanstd(tr_flat, axis=0)
    re_mean = np.nanmean(tr_flat, axis=0)
    re_std = np.where(re_std_raw == 0, 1.0, re_std_raw)
    if not (np.allclose(re_mean, mean, atol=1e-6) and
            np.allclose(re_std, std, atol=1e-6)):
        raise RuntimeError("S8 PROVENANCE: train-only stats not reproducible")
    cont = np.where(np.isnan(cont_raw), fill, cont_raw)
    cont = (cont - mean) / std
    n_big = int((np.abs(cont) > 12).sum())
    log(f"S8 |z|>12 count={n_big} (logged only; no clipping applied)")
    X[:, :, CONT_IDX] = cont.astype(np.float32)
    if not np.isfinite(X.astype(np.float64)).all():
        raise RuntimeError("non-finite values remain in X")
    log("S8 train-only scaling done (18 continuous, 7 binary untouched)")

    # S9: artifacts (chronological order: non-decreasing dates, stable sort)
    order = np.argsort(date_arr, kind="stable")
    X, y, split = X[order], y[order], split[order]
    cs = [cs[i] for i in order]
    ds = [ds[i] for i in order]
    np.savez_compressed(PROC / "sequences.npz", X=X, y=y,
                        cell=np.array(cs), date=np.array(ds), split=split)
    with open(PROC / "event_days.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["lat", "lon", "date"])
        for c in sorted(event):
            for d in sorted(event[c]):
                w.writerow([c[0], c[1], d.isoformat()])
    with open(PROC / "scaler_stats.json", "w", encoding="utf-8") as fh:
        json.dump({"features": FEATURES, "binary_idx": BIN_IDX,
                   "cont_idx": CONT_IDX, "mean": mean.tolist(),
                   "std": std.tolist(), "fill": fill.tolist(),
                   "train_only": True,
                   "n_train_rows": int(tr_mask.sum()),
                   "n_nontrain_rows": int((~tr_mask).sum())}, fh, indent=1)
    import yaml
    yaml.safe_dump({
        "features": FEATURES, "n_features": 25, "input_days": 21,
        "horizon": 7, "embargo_days": 28,
        "split_boundaries_fixed": {"B1": "2025-06-12", "B2": "2026-01-12"},
        "train_range": ["2023-01-21", "2025-06-05"],
        "val_range": ["2025-07-11", "2026-01-05"],
        "test_range": ["2026-02-10", "2026-09-08"],
        "sentinel_policy": "-999 -> NaN + mask channel",
        "snap": {"lat_step": 0.5, "lon_step": 0.625},
        "label": {"event": "gregarious PRESENT (D1b)",
                  "universe": "observed cell-days (D2a)",
                  "conflict": "any-PRESENT-wins (D3a)",
                  "filter": "DUBIOUS/UNCLEAR excluded (D4)"},
    }, open(PROC / "feature_manifest.yaml", "w"), sort_keys=False)
    log(f"S9 artifacts written: sequences.npz X={X.shape} pos={int(y.sum())}")


if __name__ == "__main__":
    main()

