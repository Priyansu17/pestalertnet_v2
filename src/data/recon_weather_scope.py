"""Phase 1b read-only recon: does the POWER weather grid cover the GLOBAL pest scope?

The approved fallback joins FAO RAMSES observations to NASA POWER weather by
location + date. After the approved scope widening (Decision 3 amendment:
global Desert Locust, pooled multi-country), the pest data spans roughly
lat 0..60, lon -17..76, while the POWER download currently covers only
lat 21..31, lon 68..77 (India zone).

This script quantifies the mismatch (read-only, no labelling, no Phase 2 work):
    - how many PRESENT rows / cell-day samples fall inside the current bbox
    - which countries carry the positives and how large a weather grid each
      country needs at 0.5 deg x 0.5 deg resolution
    - the projected POWER regional request cost for candidate scopes

Output: outputs/weather_scope_recon.txt
"""

import csv
import datetime
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = PROJECT_ROOT / "data" / "raw" / "ramses_observations_global.csv"
OUT_PATH = PROJECT_ROOT / "outputs" / "weather_scope_recon.txt"

CURRENT_BBOX = {"lat_min": 21.0, "lat_max": 31.0, "lon_min": 68.0, "lon_max": 77.0}
CELL = 0.5  # POWER regional resolution is ~0.5 deg


def xdate(ms):
    try:
        return datetime.datetime.utcfromtimestamp(int(ms) / 1000).date()
    except (TypeError, ValueError):
        return None


def cell(lat, lon):
    return (int(lat // CELL), int(lon // CELL))


def main() -> int:
    total = 0
    present = 0
    present_in_bbox = 0
    samples = set()
    present_samples = set()
    ctry_rows = defaultdict(int)
    ctry_bbox = {}
    ctry_cells = defaultdict(set)
    ctry_dates = defaultdict(list)

    with open(CSV_PATH, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            total += 1
            lat = row.get("Latitude", "").strip()
            lon = row.get("Longitude", "").strip()
            if not lat or not lon:
                continue
            try:
                la, lo = float(lat), float(lon)
            except ValueError:
                continue
            if la == 0.0 and lo == 0.0:
                continue  # null-island guard
            ctry = (row.get("Country") or "").strip() or "UNKNOWN"
            d = xdate(row.get("Obs_Date"))
            is_present = (row.get("Locust_Presence") or "").strip() == "PRESENT"

            key = (cell(la, lo), d)
            samples.add(key)
            if is_present:
                present += 1
                if (
                    CURRENT_BBOX["lat_min"] <= la <= CURRENT_BBOX["lat_max"]
                    and CURRENT_BBOX["lon_min"] <= lo <= CURRENT_BBOX["lon_max"]
                ):
                    present_in_bbox += 1
                ctry_rows[ctry] += 1
                ctry_cells[ctry].add(cell(la, lo))
                if d:
                    ctry_dates[ctry].append(d)
                bb = ctry_bbox.get(ctry)
                if bb is None:
                    ctry_bbox[ctry] = [la, la, lo, lo]
                else:
                    bb[0] = min(bb[0], la)
                    bb[1] = max(bb[1], la)
                    bb[2] = min(bb[2], lo)
                    bb[3] = max(bb[3], lo)
                present_samples.add(key)

    lines = []
    lines.append("=== Phase 1b recon: POWER weather coverage vs GLOBAL pest scope ===")
    lines.append(f"file: {CSV_PATH}")
    lines.append(f"total observation rows: {total}")
    lines.append(f"PRESENT rows: {present}")
    lines.append(f"distinct (cell x day) samples: {len(samples)}")
    lines.append(f"distinct PRESENT (cell x day) samples: {len(present_samples)}")
    lines.append(
        "PRESENT rows inside CURRENT POWER bbox "
        f"(lat {CURRENT_BBOX['lat_min']}-{CURRENT_BBOX['lat_max']}, "
        f"lon {CURRENT_BBOX['lon_min']}-{CURRENT_BBOX['lon_max']}): "
        f"{present_in_bbox} ({100.0 * present_in_bbox / max(present,1):.2f}% of PRESENT)"
    )
    lines.append(f"distinct countries with PRESENT: {len(ctry_rows)}")
    lines.append("")
    lines.append("--- per-country weather-grid requirement (0.5 deg cells) ---")
    lines.append(
        f"{'country':<18}{'present':>9}{'cells':>7}{'span_days':>10}"
        f"   lat_range            lon_range"
    )
    for ctry, n in sorted(ctry_rows.items(), key=lambda kv: -kv[1]):
        bb = ctry_bbox[ctry]
        dates = ctry_dates[ctry]
        span = (max(dates) - min(dates)).days + 1 if dates else 0
        lines.append(
            f"{ctry:<18}{n:>9}{len(ctry_cells[ctry]):>7}{span:>10}"
            f"   {bb[0]:7.2f}..{bb[1]:7.2f}    {bb[2]:8.2f}..{bb[3]:7.2f}"
        )

    # Cumulative cost if we take the top-K countries by PRESENT
    lines.append("")
    lines.append("--- cumulative scope/cost curve (top-K countries by PRESENT) ---")
    ordered = sorted(ctry_rows.items(), key=lambda kv: -kv[1])
    cum_cells = set()
    cum_present = 0
    for k in range(1, len(ordered) + 1):
        ctry, n = ordered[k - 1]
        cum_cells |= ctry_cells[ctry]
        cum_present += n
        if k in (1, 3, 5, 8, 12, 16, len(ordered)) or k == len(ordered):
            lines.append(
                f"  top-{k:<3} countries: PRESENT={cum_present:>6} "
                f"({100.0 * cum_present / present:5.1f}%)  "
                f"weather cells(0.5deg)={len(cum_cells):>6}"
            )

    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"WROTE {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
