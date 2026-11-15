"""Phase 1b - L1 ingestion: download FAO RAMSES Desert Locust observations (India).

Official source (verified 2026-08-22):
    ArcGIS Feature Service: Ramses_Observations_Dashboard_View
    Org: FAO Locust Hub (sjP4Ugu5s0dZWLjd)
    Layer 0: initial_data_publication (point geometry, global count 174,664;
             India subset count 12,785)

Method notes (empirically verified against the live service):
    - The service REJECTS f=csv queries (HTTP 400 "Bad Request").
    - f=json works; offset pagination REQUIRES orderByFields=OBJECTID.
    So we page through f=json (1000/page) and serialize the feature
    attributes to CSV locally. Stdlib only (urllib) - no installs needed.

Output: data/raw/ramses_observations_india.csv
"""

import csv
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

BASE = (
    "https://services5.arcgis.com/sjP4Ugu5s0dZWLjd/arcgis/rest/services/"
    "Ramses_Observations_Dashboard_View/FeatureServer/0/query"
)
PAGE_SIZE = 1000
MAX_RETRIES = 4
RETRY_SLEEP_S = 5

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = PROJECT_ROOT / "data" / "raw" / "ramses_observations_india.csv"


def build_url(offset: int) -> str:
    params = {
        "where": "Country='India'",
        "outFields": "*",
        "returnGeometry": "false",
        "orderByFields": "OBJECTID",
        "resultOffset": str(offset),
        "resultRecordCount": str(PAGE_SIZE),
        "f": "json",
    }
    return BASE + "?" + urllib.parse.urlencode(params)


def fetch_page(offset: int) -> dict:
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(
                build_url(offset),
                headers={"User-Agent": "PestAlertNet-Phase1b/1.0"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            if "error" in payload:
                raise RuntimeError(f"service error: {payload['error']}")
            return payload
        except Exception as exc:  # noqa: BLE001 - retry any transient error
            last_err = exc
            print(f"  offset={offset} attempt {attempt} failed: {exc}", flush=True)
            time.sleep(RETRY_SLEEP_S)
    raise RuntimeError(f"All retries failed at offset {offset}: {last_err}")


def main() -> int:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = None
    rows_written = 0
    offset = 0

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as fh:
        writer = None
        while True:
            payload = fetch_page(offset)
            feats = payload.get("features", [])
            if not feats:
                break
            if fieldnames is None:
                fieldnames = list(feats[0]["attributes"].keys())
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                print(f"Columns ({len(fieldnames)}): {fieldnames}", flush=True)
            for feat in feats:
                writer.writerow(feat["attributes"])
            rows_written += len(feats)
            print(
                f"offset={offset:>6}  got={len(feats):>5}  "
                f"total={rows_written}  exceeded={payload.get('exceededTransferLimit')}",
                flush=True,
            )
            if len(feats) < PAGE_SIZE or not payload.get("exceededTransferLimit"):
                break
            offset += PAGE_SIZE

    print(f"DONE rows_written={rows_written} file={OUT_PATH}", flush=True)
    if rows_written == 0:
        print("ERROR: zero rows downloaded", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())