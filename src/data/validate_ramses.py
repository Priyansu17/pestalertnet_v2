"""Phase 1b - L1 validation: schema + quality audit of the India RAMSES CSV.

Reads data/raw/ramses_observations_india.csv and writes a validation
report to outputs/validation_ramses.txt covering:
    - row/column counts
    - Obs_Date / Year ranges
    - geographic coverage (districts, rounded coordinate cells)
    - categorical inventories for labeling-relevant fields
      (Locust_Presence, Locust_Type, RAMSES_Locust_Type, Presence_Type,
       Behaviour, Activity, Density_Description, Observation_Reliability)
    - per-column missing-value counts

Stdlib only. No labeling or feature engineering happens here.
"""

import csv
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = PROJECT_ROOT / "data" / "raw" / "ramses_observations_india.csv"
REPORT_PATH = PROJECT_ROOT / "outputs" / "validation_ramses.txt"

CATEGORICAL_FIELDS = [
    "Locust_Presence",
    "Locust_Type",
    "RAMSES_Locust_Type",
    "Presence_Type",
    "Behaviour",
    "Activity",
    "Density_Description",
    "Density_Unit",
    "Observation_Reliability",
    "Observation_Confirmation",
    "Data_Collection_Method",
]

DATE_FORMATS = ["%m/%d/%Y %I:%M:%S %p", "%Y-%m-%d", "%d/%m/%Y"]


def parse_date(value):
    if not value:
        return None
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(value.strip(), fmt)
        except ValueError:
            continue
    return None


def main() -> int:
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found")
        return 1

    rows = 0
    header = []
    missing = Counter()
    dates = []
    years = Counter()
    districts = Counter()
    coord_cells = set()
    cat_values = {f: Counter() for f in CATEGORICAL_FIELDS}
    bad_dates = 0

    with open(CSV_PATH, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        for row in reader:
            rows += 1
            for col in header:
                v = (row.get(col) or "").strip()
                if v == "":
                    missing[col] += 1
            d = parse_date(row.get("Obs_Date"))
            if d is not None:
                dates.append(d)
            elif (row.get("Obs_Date") or "").strip():
                bad_dates += 1
            y = (row.get("Year") or "").strip()
            if y:
                years[y] += 1
            dist = (row.get("District") or "").strip()
            if dist:
                districts[dist] += 1
            lat = (row.get("Latitude") or "").strip()
            lon = (row.get("Longitude") or "").strip()
            if lat and lon:
                try:
                    coord_cells.add((round(float(lat), 1), round(float(lon), 1)))
                except ValueError:
                    pass
            for f in CATEGORICAL_FIELDS:
                v = (row.get(f) or "").strip()
                if v:
                    cat_values[f][v] += 1

    lines = []
    lines.append("=== PestAlertNet Phase 1b - RAMSES India validation ===")
    lines.append(f"file: {CSV_PATH}")
    lines.append(f"rows: {rows}")
    lines.append(f"columns ({len(header)}): {header}")
    if dates:
        lines.append(f"Obs_Date min: {min(dates).isoformat()}")
        lines.append(f"Obs_Date max: {max(dates).isoformat()}")
        lines.append(f"unparseable non-empty Obs_Date values: {bad_dates}")
    lines.append(f"Year distribution: {dict(sorted(years.items()))}")
    lines.append(f"distinct districts: {len(districts)}")
    top_districts = ", ".join(
        f"{k}({v})" for k, v in districts.most_common(15)
    )
    lines.append(f"top districts: {top_districts}")
    lines.append(f"distinct rounded (lat,lon) cells (0.1 deg): {len(coord_cells)}")
    lats = [c[0] for c in coord_cells]
    lons = [c[1] for c in coord_cells]
    if lats:
        lines.append(f"lat range (rounded cells): {min(lats)} .. {max(lats)}")
        lines.append(f"lon range (rounded cells): {min(lons)} .. {max(lons)}")
    lines.append("")
    lines.append("--- categorical inventories ---")
    for f in CATEGORICAL_FIELDS:
        total_distinct = len(cat_values[f])
        items = ", ".join(f"{k}({v})" for k, v in cat_values[f].most_common(20))
        lines.append(f"{f}: distinct={total_distinct}; {items}")
    lines.append("")
    lines.append("--- missing values (count of empty strings per column) ---")
    for col in header:
        pct = 100.0 * missing[col] / rows if rows else 0.0
        lines.append(f"{col}: {missing[col]} ({pct:.1f}%)")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"WROTE {REPORT_PATH}")
    print(f"rows={rows} cols={len(header)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())