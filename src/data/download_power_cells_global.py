"""Phase 1b - L1 ingestion: NASA POWER daily weather for the GLOBAL RAMSES footprint.

Approved by Decision Sheet #2 (FAO Locust Hub + NASA POWER fallback) and the
Decision 3 scope amendment (global Desert Locust pool).

Background
----------
The existing `power_*_2023..2026.csv` files cover ONLY the India bbox
(lat 21-31, lon 68-77), which covers 0.31% of the global PRESENT rows. The
approved global scope therefore requires POWER re-acquisition for the EXACT
2,073 grid cells that appear in ramses_observations_global.csv.

The NASA POWER *point* endpoint (verified 2026-08-22 in probe_power_point.py)
accepts all 4 parameters + the full 2023-01-01..2026-09-15 range in a single
~3 s request. The *regional* endpoint cannot be used here:
  - max 1 parameter/request, >=2 deg bbox, <=366 days/request
  - an enclosing bbox would download ~17,880 cells (8.6x the needed data).

Design: per-cell files + checkpoint-by-existence
-------------------------------------------------
Each grid cell is written to its OWN file:
    data/raw/power_point_cells/{lat}_{lon}.csv
A cell is considered complete iff its file exists and is non-empty, so a
killed/incomplete fetch is simply retried on the next run with ZERO
duplicates (atomic via temp-file + os.replace). This is what makes the
download resumable across terminal cycles.

A `--consolidate` flag merges the per-cell files into one long-format CSV:
    data/raw/power_cells_global.csv   (lat,lon,date,T2M,RH2M,PRECTOTCORR,WS2M)

Stdlib only.
"""

import argparse
import csv
import datetime
import os
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMS = "T2M,RH2M,PRECTOTCORR,WS2M"
START, END = "20230101", "20260915"

LAT_STEP = 0.5
LON_STEP = 0.625

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAMSES_CSV = PROJECT_ROOT / "data" / "raw" / "ramses_observations_global.csv"
CELLS_DIR_DEFAULT = PROJECT_ROOT / "data" / "raw" / "power_point_cells"
CONSOLIDATED_DEFAULT = PROJECT_ROOT / "data" / "raw" / "power_cells_global.csv"
LOG_PATH = PROJECT_ROOT / "outputs" / "download_power_cells_log.txt"

MAX_RETRIES = 4
RETRY_SLEEP_S = 5
MAX_WORKERS = 2
TIME_BUDGET_S = int(os.environ.get("PN_POWER_BUDGET", "1700"))

# These are reassigned in main() if --cells-dir overrides paths.
CELLS_DIR = CELLS_DIR_DEFAULT
CONSOLIDATED = CONSOLIDATED_DEFAULT


def _log(msg: str, also_stdout: bool = True) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    if also_stdout:
        print(line, flush=True)


def snap(value: float, step: float) -> float:
    return round(round(value / step) * step, 4)


def collect_cells() -> list:
    cells = set()
    with open(RAMSES_CSV, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            lat = (row.get("Latitude") or "").strip()
            lon = (row.get("Longitude") or "").strip()
            if not lat or not lon:
                continue
            try:
                la, lo = float(lat), float(lon)
            except ValueError:
                continue
            if la == 0.0 and lo == 0.0:
                continue
            cells.add((snap(la, LAT_STEP), snap(lo, LON_STEP)))
    return sorted(cells)


def cell_path(lat: float, lon: float) -> Path:
    return CELLS_DIR / f"{lat}_{lon}.csv"


def parse_power_csv(text: str) -> list:
    """Return list of [date_str, T2M, RH2M, PRECTOTCORR, WS2M]."""
    data = []
    header_seen = False
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        if not header_seen:
            if s.startswith("YEAR") and "DOY" in s:
                header_seen = True
            continue
        parts = s.split(",")
        if len(parts) < 6:
            continue
        try:
            year, doy = int(parts[0]), int(parts[1])
            d = datetime.date(year, 1, 1) + datetime.timedelta(days=doy - 1)
        except ValueError:
            continue
        data.append([d.isoformat(), parts[2], parts[3], parts[4], parts[5]])
    return data


def fetch_cell(lat: float, lon: float) -> int:
    """Download one cell. Returns data-row count, or -1 if already present."""
    out = cell_path(lat, lon)
    if out.exists() and out.stat().st_size > 500:
        return -1
    q = {
        "parameters": PARAMS,
        "community": "AG",
        "longitude": str(lon),
        "latitude": str(lat),
        "start": START,
        "end": END,
        "format": "CSV",
    }
    url = BASE + "?" + urllib.parse.urlencode(q)
    text = None
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "PestAlertNet-Phase1b/1.0"}
            )
            with urllib.request.urlopen(req, timeout=180) as resp:
                text = resp.read().decode("utf-8", errors="replace")
            if text.lstrip().startswith("{"):
                raise RuntimeError(f"POWER API error: {text[:300]}")
            break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            _log(f"  cell ({lat},{lon}) attempt {attempt} failed: {exc}")
            if "429" in str(exc):
                # Rate-limited by POWER API: exponential backoff (5, 10, 20, 40 s)
                time.sleep(RETRY_SLEEP_S * (2 ** (attempt - 1)))
            else:
                time.sleep(RETRY_SLEEP_S)
    if text is None:
        raise RuntimeError(f"cell ({lat},{lon}) failed after retries: {last_err}")
    rows = parse_power_csv(text)
    last_write_err = None
    for attempt in range(1, 8):
        try:
            with open(out, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["date", "T2M", "RH2M", "PRECTOTCORR", "WS2M"])
                w.writerows(rows)
            return len(rows)
        except OSError as exc:  # noqa: BLE001 - transient external lock (Defender/indexer)
            last_write_err = exc
            time.sleep(1.0)
    raise RuntimeError(
        f"({lat},{lon}) could not write file after retries: {last_write_err}"
    )


def run_download() -> int:
    CELLS_DIR.mkdir(parents=True, exist_ok=True)
    cells = collect_cells()

    remaining = [
        c for c in cells
        if not (cell_path(*c).exists() and cell_path(*c).stat().st_size > 500)
    ]
    total = len(cells)
    done = total - len(remaining)
    _log(f"DOWNLOAD START: total_cells={total} done={done} remaining={len(remaining)} "
         f"workers={MAX_WORKERS} budget={TIME_BUDGET_S}s")
    if not remaining:
        _log("DOWNLOAD COMPLETE: all cells already present")
        return 0

    deadline = time.time() + TIME_BUDGET_S
    errors = []
    completed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        future_map = {ex.submit(fetch_cell, la, lo): (la, lo) for la, lo in remaining}
        for fut in as_completed(future_map):
            la, lo = future_map[fut]
            try:
                n = fut.result()
                if n >= 0:
                    completed += 1
                    done += 1
                    if completed % 50 == 0:
                        _log(f"progress: done={done}/{total}")
            except Exception as exc:  # noqa: BLE001
                errors.append((la, lo, str(exc)[:200]))
                _log(f"  FAILED cell ({la},{lo}): {exc}")
            if time.time() > deadline:
                _log(f"TIME BUDGET reached: done={done}/{total} completed_this_run={completed} "
                     f"errors={len(errors)} - exiting (re-run to resume)")
                break
    _log(f"DOWNLOAD END: done={done}/{total} completed_this_run={completed} errors={len(errors)}")
    if errors:
        for la, lo, msg in errors:
            _log(f"  ERROR cell ({la},{lo}): {msg}")
    return 0


def run_consolidate() -> int:
    """Merge per-cell files into one long-format CSV."""
    if CONSOLIDATED.exists():
        CONSOLIDATED.unlink()
    CELLS_DIR.mkdir(parents=True, exist_ok=True)
    cells = collect_cells()
    written = 0
    skipped_missing = 0
    with open(CONSOLIDATED, "w", newline="", encoding="utf-8") as out:
        w = csv.writer(out)
        w.writerow(["lat", "lon", "date", "T2M", "RH2M", "PRECTOTCORR", "WS2M"])
        for la, lo in cells:
            p = cell_path(la, lo)
            if not p.exists() or p.stat().st_size < 500:
                skipped_missing += 1
                continue
            with open(p, newline="", encoding="utf-8") as fh:
                for row in csv.reader(fh):
                    if not row or row[0] == "date":
                        continue
                    w.writerow([la, lo, row[0], row[1], row[2], row[3], row[4]])
                    written += 1
    _log(f"CONSOLIDATE DONE: rows={written} file={CONSOLIDATED} skipped_missing={skipped_missing}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--consolidate", action="store_true")
    ap.add_argument("--cells-dir", default=None)
    args = ap.parse_args()
    global CELLS_DIR, CONSOLIDATED
    if args.cells_dir:
        CELLS_DIR = Path(args.cells_dir)
        CONSOLIDATED = CELLS_DIR.parent / "power_cells_global.csv"
    if not RAMSES_CSV.exists():
        print(f"ERROR: {RAMSES_CSV} missing - run RAMSES global download first", flush=True)
        return 1
    if args.consolidate:
        return run_consolidate()
    if LOG_PATH.exists():
        LOG_PATH.unlink()
    return run_download()


if __name__ == "__main__":
    sys.exit(main())