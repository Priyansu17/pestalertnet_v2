"""Phase 1b/1c - READ-ONLY inspection to inform the labeling decision.

This script does NOT create a label and does NOT modify any data. It only
characterises the real RAMSES India observations against the NASA POWER grid
so that the project owner can choose a labeling rule (Decision Sheet #4).

Writes: outputs/label_options_analysis.txt
"""

import csv
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = PROJECT_ROOT / "data" / "raw" / "ramses_observations_india.csv"
POWER_REF = PROJECT_ROOT / "data" / "raw" / "power_T2M_2023.csv"
REPORT_PATH = PROJECT_ROOT / "outputs" / "label_options_analysis.txt"


def epoch_ms_to_date(value):
    try:
        ms = int(float(value))
    except (TypeError, ValueError):
        return None
    try:
        return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).date()
    except (OverflowError, OSError, ValueError):
        return None


def load_power_cells():
    """Return the set of (lat, lon) grid cells published by POWER."""
    cells = set()
    with open(POWER_REF, newline="", encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    start = 0
    for i, line in enumerate(lines):
        if line.startswith("-END HEADER-"):
            start = i + 1
            break
    reader = csv.DictReader(lines[start:])
    for row in reader:
        try:
            cells.add((float(row["LAT"]), float(row["LON"])))
        except (KeyError, TypeError, ValueError):
            continue
    return cells


def nearest_power_cell(lat, lon, cells):
    best = None
    best_d = None
    for clat, clon in cells:
        d = (clat - lat) ** 2 + (clon - lon) ** 2
        if best_d is None or d < best_d:
            best_d = d
            best = (clat, clon)
    return best


def main() -> int:
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found")
        return 1

    cells = load_power_cells()
    rows = list(csv.DictReader(open(CSV_PATH, newline="", encoding="utf-8")))
    total = len(rows)

    all_days = Counter()
    present_by_day = Counter()
    by_year_present = Counter()
    by_district_present = Counter()
    by_district_survey_days = defaultdict(set)
    presence_type_by_day = defaultdict(Counter)
    density_by_day = defaultdict(Counter)
    present_cells = Counter()
    lat_unmatched = 0

    for r in rows:
        d = epoch_ms_to_date(r.get("Obs_Date"))
        if d is None:
            continue
        all_days[d] += 1
        district = (r.get("District") or "").strip()
        by_district_survey_days[district].add(d)
        if (r.get("Locust_Presence") or "").strip() == "PRESENT":
            present_by_day[d] += 1
            by_year_present[(r.get("Year") or "").strip()] += 1
            by_district_present[district] += 1
            presence_type_by_day[d][(r.get("Presence_Type") or "").strip()] += 1
            density_by_day[d][(r.get("Density_Description") or "").strip()] += 1
            try:
                lat = float(r["Latitude"])
                lon = float(r["Longitude"])
            except (TypeError, ValueError):
                lat_unmatched += 1
                continue
            cell = nearest_power_cell(lat, lon, cells)
            if cell:
                present_cells[cell] += 1

    lines = []
    add = lines.append
    add("=== Phase 1b/1c read-only inspection: labeling feasibility ===")
    add(f"file: {CSV_PATH}")
    add(f"total observation rows: {total}")
    add(f"distinct survey days (any row): {len(all_days)}")
    add(f"distinct POWER grid cells referenced: {len(cells)}")
    add("")
    add("--- POSITIVE SIGNAL (Locust_Presence == PRESENT) ---")
    add(f"present rows: {sum(present_by_day.values())}")
    add(f"distinct days with >=1 present: {len(present_by_day)}")
    add(f"by year: {dict(sorted(by_year_present.items()))}")
    add(f"distinct districts with >=1 present: {len(by_district_present)}")
    add(f"distinct POWER cells with >=1 present: {len(present_cells)}")
    add(f"unparseable lat/lon in present rows: {lat_unmatched}")
    add("")
    add("--- PRESENT COUNTS PER DAY (all days) ---")
    for d in sorted(present_by_day):
        ptypes = dict(presence_type_by_day[d])
        dens = dict(density_by_day[d])
        add(f"{d.isoformat()}  present={present_by_day[d]:>3}  "
            f"survey_rows={all_days[d]:>4}  density={dens}  presence_type={ptypes}")
    add("")
    add("--- TOP POWER CELLS BY PRESENT-ROW COUNT ---")
    for (clat, clon), c in present_cells.most_common(12):
        add(f"lat={clat:.3f} lon={clon:.3f} -> {c} present rows")
    add("")
    add("--- SURVEY-ACTIVITY CALENDAR SHAPE ---")
    days_sorted = sorted(all_days)
    if days_sorted:
        add(f"first survey day: {days_sorted[0].isoformat()}")
        add(f"last survey day:  {days_sorted[-1].isoformat()}")
        span = (days_sorted[-1] - days_sorted[0]).days + 1
        add(f"calendar span: {span} days")
        add(f"days with >=1 observation: {len(days_sorted)} "
            f"({100.0 * len(days_sorted) / span:.1f}% of span)")
        gaps = [
            (days_sorted[i + 1] - days_sorted[i]).days
            for i in range(len(days_sorted) - 1)
        ]
        add(f"max gap between consecutive observation days: {max(gaps) if gaps else 0} days")
        add(f"gaps > 7 days: {sum(1 for g in gaps if g > 7)}")
        add(f"gaps > 28 days: {sum(1 for g in gaps if g > 28)}")
        year_days = Counter()
        for d in days_sorted:
            year_days[d.year] += 1
        add(f"distinct survey days per year: {dict(sorted(year_days.items()))}")
    add("")
    add("--- PER-DISTRICT SURVEY COVERAGE (top 20 by survey days) ---")
    add(f"districts surveyed: {len(by_district_survey_days)}")
    for dist, dayset in sorted(
        by_district_survey_days.items(), key=lambda kv: -len(kv[1])
    )[:20]:
        add(f"{dist}: survey_days={len(dayset)} "
            f"present_rows={by_district_present.get(dist, 0)}")
    add("")
    add("--- THRESHOLD SWEEP (illustrative ONLY, no rule chosen) ---")
    add("Days qualifying if a presence-based rule required at least N")
    add("confirmed-present observations on that day:")
    for n in (1, 2, 3, 5, 8):
        cnt = sum(1 for _, c in present_by_day.items() if c >= n)
        add(f"  >= {n} present obs/day -> {cnt} qualifying days")
    add("")
    add("--- LABELLING OPTIONS SURFACED (owner decision; none implemented) ---")
    add("A. FAO official situation field (Recession/Outbreak/Upsurge/Plague):")
    add("   NOT among the 72 columns -> unavailable from this export.")
    add("B. Published ETL vs numeric pest counts:")
    add("   NO numeric abundance/pest_count column -> not directly available.")
    add("C. Presence-based label from confirmed field observations")
    add("   (Locust_Presence / Presence_Type / RAMSES_Locust_Type).")
    add("D. Decision Sheet statistical relative-anomaly rule (1.4x / 30d / floor 8):")
    add("   requires a numeric pest_count series -> unavailable on this export.")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"WROTE {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
