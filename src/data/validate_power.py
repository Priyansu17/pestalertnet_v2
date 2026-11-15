"""Phase 1b - L1 validation: schema + quality audit of NASA POWER regional CSVs.

Reads data/raw/power_{PARAM}_{YYYY}.csv (regional daily endpoint, community=AG)
and writes outputs/validation_power.txt covering, per parameter:
    - chunk files found and their row counts
    - distinct (LAT, LON) grid cells and the grid spacing
    - YEAR/DOY coverage
    - count of the POWER missing-data sentinel (-999)
    - basic min/mean/max of valid values

The regional CSV layout is:
    -BEGIN HEADER-
    ... free text ...
    -END HEADER-
    LAT,LON,YEAR,DOY,<PARAM>
    <rows...>

Stdlib only. No preprocessing, resampling, or joining happens here.
"""

import csv
import sys
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_PATH = PROJECT_ROOT / "outputs" / "validation_power.txt"

PARAMS = ["T2M", "RH2M", "PRECTOTCORR", "WS2M"]
MISSING_SENTINEL = -999.0
EXPECTED_CHUNKS = ["2023", "2024", "2025", "2026"]


def read_regional_csv(path):
    """Yield dict rows from a POWER regional CSV, skipping the header block."""
    with open(path, newline="", encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    start = 0
    for i, line in enumerate(lines):
        if line.strip() == "-END HEADER-":
            start = i + 1
            break
    reader = csv.DictReader(lines[start:])
    return reader.fieldnames, list(reader)


def doy_to_date(year, doy):
    try:
        return date(int(year), 1, 1) + timedelta(days=int(doy) - 1)
    except (ValueError, TypeError):
        return None


def main() -> int:
    lines = []
    lines.append("=== PestAlertNet Phase 1b - NASA POWER regional validation ===")
    lines.append(f"source dir: {RAW_DIR}")
    lines.append("endpoint: power.larc.nasa.gov/api/temporal/daily/regional (community=AG)")
    lines.append("")

    overall_ok = True
    for param in PARAMS:
        lines.append(f"--- {param} ---")
        found = []
        for year in EXPECTED_CHUNKS:
            p = RAW_DIR / f"power_{param}_{year}.csv"
            if p.exists() and p.stat().st_size > 1000:
                found.append((year, p))
            else:
                lines.append(f"  MISSING/EMPTY chunk: {p.name}")
                overall_ok = False

        cells = set()
        missing = 0
        total = 0
        dates = []
        values = []
        per_year = Counter()
        value_col = None

        for year, p in found:
            fieldnames, rows = read_regional_csv(p)
            if value_col is None:
                value_col = fieldnames[-1]
            per_year[year] = len(rows)
            for r in rows:
                total += 1
                raw = (r.get(value_col) or "").strip()
                if raw == "":
                    missing += 1
                    continue
                try:
                    v = float(raw)
                except ValueError:
                    missing += 1
                    continue
                if v == MISSING_SENTINEL:
                    missing += 1
                    continue
                values.append(v)
                try:
                    cells.add((round(float(r["LAT"]), 3), round(float(r["LON"]), 3)))
                except (ValueError, TypeError, KeyError):
                    pass
                d = doy_to_date(r.get("YEAR"), r.get("DOY"))
                if d is not None:
                    dates.append(d)

        lines.append(f"  value column: {value_col}")
        lines.append(f"  chunk row counts: {dict(sorted(per_year.items()))}")
        lines.append(f"  total data rows: {total}")
        lines.append(f"  missing/sentinel(-999) rows: {missing} "
                     f"({100.0 * missing / total:.2f}%)" if total else "  total rows: 0")
        lines.append(f"  distinct grid cells: {len(cells)}")
        if cells:
            lats = sorted({c[0] for c in cells})
            lons = sorted({c[1] for c in cells})
            lat_step = round(lats[1] - lats[0], 4) if len(lats) > 1 else None
            lon_step = round(lons[1] - lons[0], 4) if len(lons) > 1 else None
            lines.append(f"  lat cells: {len(lats)} ({lats[0]} .. {lats[-1]}), "
                         f"step={lat_step}")
            lines.append(f"  lon cells: {len(lons)} ({lons[0]} .. {lons[-1]}), "
                         f"step={lon_step}")
        if dates:
            lines.append(f"  date coverage: {min(dates).isoformat()} .. "
                         f"{max(dates).isoformat()}")
        if values:
            lines.append(f"  value min/mean/max: {min(values):.3f} / "
                         f"{sum(values) / len(values):.3f} / {max(values):.3f}")
        lines.append("")

    lines.append(f"ALL_CHUNKS_PRESENT: {overall_ok}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"WROTE {REPORT_PATH}")
    print(f"ALL_CHUNKS_PRESENT={overall_ok}")
    return 0 if overall_ok else 2


if __name__ == "__main__":
    sys.exit(main())
