"""Phase 1b read-only recon: exact POWER weather footprint for the GLOBAL scope.

After the approved scope widening (Decision 3 amendment: global Desert Locust),
the existing POWER download (lat 21-31, lon 68-77) covers only 0.31% of PRESENT
rows. This script computes the EXACT set of POWER grid cells required by the
full global RAMSES dataset, so the weather re-acquisition is sized precisely
instead of guessed.

POWER grid geometry (observed empirically from the India download):
    lat step 0.5  (lat cell centres land on .0 / .5)
    lon step 0.625 (lon cell centres land on .125 / .375 / .625 / .875)

Both PRESENT and ABSENT rows are counted, because negative survey samples are
model inputs too -- weather is needed wherever a sample exists, not only where
locoosts were seen.

Output: outputs/weather_footprint.txt  (read-only analysis only)
"""

import csv
import datetime
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = PROJECT_ROOT / "data" / "raw" / "ramses_observations_global.csv"
OUT_PATH = PROJECT_ROOT / "outputs" / "weather_footprint.txt"

LAT_STEP = 0.5
LON_STEP = 0.625
CLUSTER_GAP_DEG = 2.0  # merge tile boxes closer than this


def snap(value, step):
    return round(round(value / step) * step, 4)


def main() -> int:
    total = 0
    skipped = 0
    cells = set()
    sample_days = set()
    date_min = None
    date_max = None
    present_cells = set()

    with open(CSV_PATH, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            total += 1
            lat = (row.get("Latitude") or "").strip()
            lon = (row.get("Longitude") or "").strip()
            if not lat or not lon:
                skipped += 1
                continue
            try:
                la, lo = float(lat), float(lon)
            except ValueError:
                skipped += 1
                continue
            if la == 0.0 and lo == 0.0:
                skipped += 1
                continue

            c = (snap(la, LAT_STEP), snap(lo, LON_STEP))
            cells.add(c)
            if (row.get("Locust_Presence") or "").strip() == "PRESENT":
                present_cells.add(c)

            ms = (row.get("Obs_Date") or "").strip()
            if ms:
                try:
                    d = datetime.datetime.fromtimestamp(
                        int(ms) / 1000, datetime.timezone.utc
                    ).date()
                except (ValueError, OSError):
                    continue
                sample_days.add(d)
                date_min = d if date_min is None else min(date_min, d)
                date_max = d if date_max is None else max(date_max, d)

    lats = sorted({c[0] for c in cells})
    lons = sorted({c[1] for c in cells})

    lines = []
    lines.append("=== Phase 1b recon: exact POWER weather footprint (GLOBAL scope) ===")
    lines.append(f"file: {CSV_PATH}")
    lines.append(f"observation rows: {total} (skipped null/missing coords: {skipped})")
    lines.append(f"required POWER grid cells: {len(cells)}")
    lines.append(f"  of which carry >=1 PRESENT row: {len(present_cells)}")
    lines.append(f"distinct lat values: {len(lats)}  range {min(lats)}..{max(lats)}")
    lines.append(f"distinct lon values: {len(lons)}  range {min(lons)}..{max(lons)}")
    lines.append(f"distinct sample days: {len(sample_days)}  range {date_min}..{date_max}")

    # Latitude bands: group needed cells by lat row to see how compact the grid is
    by_lat = defaultdict(set)
    for la, lo in cells:
        by_lat[la].add(lo)
    lines.append("")
    lines.append("--- needed lon cells per latitude band (compactness check) ---")
    for la in sorted(by_lat):
        lset = sorted(by_lat[la])
        span = max(lset) - min(lset)
        gaps = [
            (lset[i + 1] - lset[i]) for i in range(len(lset) - 1)
        ]
        big_gaps = sum(1 for g in gaps if g > CLUSTER_GAP_DEG)
        lines.append(
            f"lat {la:6.2f}: {len(lset):>4} cells  lon {min(lset):8.3f}..{max(lset):7.3f}"
            f"  span={span:7.3f}  gaps>2deg={big_gaps}  bbox_cells={int(span / LON_STEP) + 1}"
        )

    # Bounding-box cost comparison
    lat_span = max(lats) - min(lats)
    lon_span = max(lons) - min(lons)
    bbox_cells = (int(lat_span / LAT_STEP) + 1) * (int(lon_span / LON_STEP) + 1)
    lines.append("")
    lines.append("--- cost: exact cells vs enclosing bounding box ---")
    lines.append(
        f"exact needed cells: {len(cells)}  "
        f"(point-endpoint requests: {len(cells)})"
    )
    lines.append(
        f"enclosing bbox: lat {min(lats)}..{max(lats)} lon {min(lons)}..{max(lons)}"
        f"  -> ~{bbox_cells} cells  (density {100.0*len(cells)/max(bbox_cells,1):.1f}%)"
    )
    lines.append(
        f"date span: {(date_max - date_min).days + 1} days "
        f"(regional endpoint caps requests at 366 days)"
    )
    n_years = (date_max.year - date_min.year) + 1
    lines.append(
        f"regional-endpoint request cost if one bbox covers all: "
        f"4 params x {n_years} year-chunks = {4 * n_years} requests "
        f"(but downloads ~{bbox_cells} cells, "
        f"{bbox_cells/max(len(cells),1):.1f}x the needed data)"
    )

    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"WROTE {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
