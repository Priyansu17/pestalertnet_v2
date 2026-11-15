"""Phase 1b - L1 ingestion: NASA POWER daily regional weather grids.

Official source (verified 2026-08-22):
    https://power.larc.nasa.gov/api/temporal/daily/regional
    Community: AG | Format: CSV

Empirically verified API constraints:
    - max 1 parameter per regional request
    - bbox must span >= 2 degrees in lat and lon
    - time period per request <= 366 days

Strategy: one request per (parameter x calendar-year chunk), bbox covering
the India Desert Locust observation zone (lat 21-31, lon 68-77),
2023-01-01 .. 2026-07-29 (matches the RAMSES India observation window).

Output: data/raw/power_{PARAM}_{YYYY}.csv  (one file per chunk)
"""

import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://power.larc.nasa.gov/api/temporal/daily/regional"
PARAMS = ["T2M", "RH2M", "PRECTOTCORR", "WS2M"]
CHUNKS = [
    ("20230101", "20231231"),
    ("20240101", "20241231"),
    ("20250101", "20251231"),
    ("20260101", "20260729"),
]
LON_MIN, LON_MAX = 68, 77
LAT_MIN, LAT_MAX = 21, 31
MAX_RETRIES = 4
RETRY_SLEEP_S = 8

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "data" / "raw"


def build_url(param: str, start: str, end: str) -> str:
    params = {
        "parameters": param,
        "community": "AG",
        "longitude-min": str(LON_MIN),
        "longitude-max": str(LON_MAX),
        "latitude-min": str(LAT_MIN),
        "latitude-max": str(LAT_MAX),
        "start": start,
        "end": end,
        "format": "CSV",
    }
    return BASE + "?" + urllib.parse.urlencode(params)


def fetch(url: str) -> bytes:
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "PestAlertNet-Phase1b/1.0"}
            )
            with urllib.request.urlopen(req, timeout=300) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001 - retry transient errors
            last_err = exc
            print(f"    attempt {attempt} failed: {exc}", flush=True)
            time.sleep(RETRY_SLEEP_S)
    raise RuntimeError(f"All retries failed: {last_err}")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    failures = []
    for param in PARAMS:
        for start, end in CHUNKS:
            out_path = OUT_DIR / f"power_{param}_{start[:4]}.csv"
            if out_path.exists() and out_path.stat().st_size > 1000:
                print(f"SKIP {out_path.name} (exists)", flush=True)
                continue
            url = build_url(param, start, end)
            print(f"GET {param} {start}-{end} ...", flush=True)
            try:
                data = fetch(url)
            except RuntimeError as exc:
                print(f"FAIL {param} {start}: {exc}", flush=True)
                failures.append((param, start))
                continue
            text = data.decode("utf-8", errors="replace")
            if text.lstrip().startswith("{"):
                print(f"FAIL {param} {start}: API error body: {text[:200]}", flush=True)
                failures.append((param, start))
                continue
            out_path.write_bytes(data)
            print(f"OK   {out_path.name} size={len(data)}", flush=True)
    print(f"DONE failures={failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())