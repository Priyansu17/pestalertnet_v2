"""Phase 1b - L1 validation: schema + quality audit of the GLOBAL RAMSES CSV.

Scope amendment (approved): ingest the full RAMSES Desert Locust observation
dataset (all countries, single species) instead of the India-only subset,
because the India subset had zero positive detections in 2025/2026.

Reads data/raw/ramses_observations_global.csv and writes
outputs/validation_global.txt covering:
    - row/column counts
    - Obs_Date range (parsed as Unix epoch MILLISECONDS - the actual wire
      format; the India validator's string parser silently failed on this)
    - geographic coverage (countries, regions, rounded coordinate cells)
    - categorical inventories for labeling-relevant fields
    - Locust_Presence=PRESENT counts by country and by year
    - per-column missing-value counts

Stdlib only. No labeling or feature engineering happens here.
"""

import csv
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = PROJECT_ROOT / "data" / "raw" / "ramses_observations_global.csv"
REPORT_PATH = PROJECT_ROOT / "outputs" / "validation_global.txt"

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

EPOCH = datetime(1970, 1, 1)
STRING_FORMATS = ["%m/%d/%Y %I:%M:%S %p", "%Y-%m-%d", "%d/%m/%Y"]


def parse_date(value):
    """Parse Obs_Date. Wire format is epoch milliseconds; strings tolerated."""
    if value is None:
        return None
    v = value.strip()
    if not v:
        return None
    try:
        ms = int(float(v))
        if ms > -10_000_000_000:  # treat as epoch millis (allows 1900 sentinel)
            return EPOCH + timedelta(milliseconds=ms)
    except ValueError:
        pass
    for fmt in STRING_FORMATS:
        try:
            return datetime.strptime(v, fmt)
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
    bad_dates = 0
    years = Counter()
    countries = Counter()
    regions = Counter()
    coord_cells = set()
    cat_values = {f: Counter() for f in CATEGORICAL_FIELDS}
    present_by_country = Counter()
    present_by_year = Counter()
    absent_by_year = Counter()
    present_cells = set()
    present_cell_days = set()
    all_cell_days = set()

    with open(CSV_PATH, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        for row in reader:
            rows += 1
            for col in header:
                if not (row.get(col) or "").strip():
                    missing[col] += 1

            d = parse_date(row.get("Obs_Date"))
            if d is not None:
                dates.append(d)
            elif (row.get("Obs_Date") or "").strip():
                bad_dates += 1

            country = (row.get("Country") or "").strip()
            region = (row.get("Region") or "").strip()
            presence = (row.get("Locust_Presence") or "").strip().upper()
            year = (row.get("Year") or "").strip()
            if not year and d is not None:
                year = str(d.year)
            if year:
                years[year] += 1
            if country:
                countries[country] += 1
            if region:
                regions[region] += 1

            lat = (row.get("Latitude") or "").strip()
            lon = (row.get("Longitude") or "").strip()
            cell = None
            if lat and lon:
                try:
                    cell = (round(float(lat), 1), round(float(lon), 1))
                    coord_cells.add(cell)
                except ValueError:
                    cell = None
            if cell is not None and d is not None:
                cday = (cell, d.date())
                all_cell_days.add(cday)
                if presence == "PRESENT":
                    present_cell_days.add(cday)
                    present_cells.add(cell)

            if presence == "PRESENT":
                if country:
                    present_by_country[country] += 1
                if year:
                    present_by_year[year] += 1
            elif presence == "ABSENT":
                if year:
                    absent_by_year[year] += 1

            for f in CATEGORICAL_FIELDS:
                v = (row.get(f) or "").strip()
                if v:
                    cat_values[f][v] += 1

    lines = []
    lines.append("=== PestAlertNet Phase 1b - RAMSES GLOBAL validation ===")
    lines.append(f"file: {CSV_PATH}")
    lines.append(f"rows: {rows}")
    lines.append(f"columns ({len(header)}): {header}")
    if dates:
        lines.append(f"Obs_Date min: {min(dates).isoformat()}")
        lines.append(f"Obs_Date max: {max(dates).isoformat()}")
    lines.append(f"unparseable non-empty Obs_Date values: {bad_dates}")
    lines.append(f"Year distribution: {dict(sorted(years.items()))}")
    lines.append(f"distinct countries: {len(countries)}")
    lines.append(
        "top countries: "
        + ", ".join(f"{k}({v})" for k, v in countries.most_common(20))
    )
    lines.append(f"distinct regions: {len(regions)}")
    lines.append(f"distinct rounded (lat,lon) cells (0.1 deg): {len(coord_cells)}")
    lats = [c[0] for c in coord_cells]
    lons = [c[1] for c in coord_cells]
    if lats:
        lines.append(f"lat range (rounded cells): {min(lats)} .. {max(lats)}")
        lines.append(f"lon range (rounded cells): {min(lons)} .. {max(lons)}")
    lines.append("")
    lines.append("--- PRESENT signal ---")
    total_present = sum(present_by_country.values())
    lines.append(
        f"PRESENT rows: {total_present} "
        f"({100.0 * total_present / rows:.2f}% of rows)" if rows else "n/a"
    )
    lines.append(
        "PRESENT by year: "
        + ", ".join(f"{k}({v})" for k, v in sorted(present_by_year.items()))
    )
    lines.append(
        "ABSENT by year: "
        + ", ".join(f"{k}({v})" for k, v in sorted(absent_by_year.items()))
    )
    lines.append(
        "PRESENT by country: "
        + ", ".join(f"{k}({v})" for k, v in present_by_country.most_common(30))
    )
    lines.append("")
    lines.append("--- survey sample geometry ---")
    lines.append(f"distinct (cell x day) samples: {len(all_cell_days)}")
    lines.append(f"distinct PRESENT (cell x day): {len(present_cell_days)}")
    lines.append(f"distinct cells with any PRESENT: {len(present_cells)}")
    lines.append("")
    lines.append("--- categorical inventories ---")
    for f in CATEGORICAL_FIELDS:
        items = ", ".join(f"{k}({v})" for k, v in cat_values[f].most_common(20))
        lines.append(f"{f}: distinct={len(cat_values[f])}; {items}")
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
