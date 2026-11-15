"""Phase 1b (scope widened to GLOBAL, owner Option B) - full RAMSES download.

Downloads ALL countries' Desert Locust observations from the official FAO
RAMSES feature service (NO Country filter), preserving full 72-column
fidelity. Paginated f=json (1000 rows/page, orderByFields=OBJECTID) because
this service rejects f=csv with HTTP 400.

RESUMABLE BY DESIGN
    The downloader appends page-by-page and records the next offset in a
    sidecar state file (data/raw/ramses_observations_global.state) ONLY after
    the page is safely flushed to disk. Rows are written with QUOTE_ALL so
    every record occupies exactly one physical line, which makes the CSV
    trivially verifiable on resume: any trailing line without a terminating
    newline is discarded and the offset is recomputed from the record count.

    A time budget (default 240 s) makes each invocation exit cleanly BEFORE
    the terminal can kill it. Re-run repeatedly until it reports COMPLETE.

No labeling, no feature engineering, no preprocessing.

Output:        data/raw/ramses_observations_global.csv
State:         data/raw/ramses_observations_global.state
Progress log:  outputs/download_ramses_global_log.txt
"""

import csv
import io
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
MAX_RETRIES = 5
RETRY_SLEEP_S = 6
DEFAULT_TIME_BUDGET_S = 240
EXPECTED_ROWS_HINT = 176081  # from recon; dataset is live so drift is allowed

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = PROJECT_ROOT / "data" / "raw" / "ramses_observations_global.csv"
STATE_PATH = PROJECT_ROOT / "data" / "raw" / "ramses_observations_global.state"
LOG_PATH = PROJECT_ROOT / "outputs" / "download_ramses_global_log.txt"

LOG_LINES = []


def log(msg: str) -> None:
    print(msg, flush=True)
    LOG_LINES.append(msg)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("\n".join(LOG_LINES), encoding="utf-8")


def build_url(offset: int) -> str:
    params = {
        "where": "1=1",
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
            with urllib.request.urlopen(req, timeout=180) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            if "error" in payload:
                raise RuntimeError(f"service error: {payload['error']}")
            return payload
        except Exception as exc:  # noqa: BLE001 - retry transient errors
            last_err = exc
            log(f"  offset={offset} attempt {attempt} failed: {exc}")
            time.sleep(RETRY_SLEEP_S)
    raise RuntimeError(f"All retries failed at offset {offset}: {last_err}")


def read_state() -> int | None:
    """Return the next offset to fetch, or None if no state exists."""
    if not STATE_PATH.exists():
        return None
    try:
        txt = STATE_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not txt:
        return None
    if txt == "COMPLETE":
        return -1
    try:
        return int(txt)
    except ValueError:
        return None


def write_state(value) -> None:
    STATE_PATH.write_text(str(value), encoding="utf-8")


def repair_csv() -> tuple[int, list]:
    """Trim any partially-written trailing line and count good records.

    Returns (n_records, header_fields).
    """
    if not OUT_PATH.exists():
        return 0, []

    raw = OUT_PATH.read_bytes()
    # Drop a trailing partial record (no terminating newline).
    if raw and not raw.endswith(b"\n"):
        cut = raw.rfind(b"\n")
        raw = raw[: cut + 1] if cut != -1 else b""
        OUT_PATH.write_bytes(raw)
        log("REPAIR: discarded a partially-written trailing row")

    if not raw:
        return 0, []

    text = raw.decode("utf-8")
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        return 0, []
    header = rows[0]
    records = rows[1:]
    # Sanity: every record must match the header width.
    bad = [i for i, r in enumerate(records) if len(r) != len(header)]
    if bad:
        keep = bad[0]
        log(f"REPAIR: truncating at record {keep} (malformed width)")
        records = records[:keep]
        buf = io.StringIO()
        w = csv.writer(buf, quoting=csv.QUOTE_ALL)
        w.writerow(header)
        w.writerows(records)
        OUT_PATH.write_text(buf.getvalue(), encoding="utf-8")
    return len(records), header


def main() -> int:
    budget_s = DEFAULT_TIME_BUDGET_S
    if len(sys.argv) > 1:
        budget_s = int(sys.argv[1])
    started = time.time()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    state = read_state()
    n_records, header = repair_csv()

    if state == -1:
        log(f"ALREADY COMPLETE: {n_records} records in {OUT_PATH}")
        return 0

    if state is None:
        # No state file: derive offset from whatever survived on disk.
        offset = n_records
        log(f"FRESH/RECOVERED start: {n_records} records on disk -> offset={offset}")
    else:
        offset = state
        if offset != n_records:
            log(
                f"NOTE: state offset {offset} != on-disk records {n_records}; "
                f"trusting on-disk records"
            )
            offset = n_records

    mode = "a" if n_records > 0 else "w"
    rows_written = n_records
    pages_this_run = 0

    with open(OUT_PATH, mode, newline="", encoding="utf-8") as fh:
        writer = None
        if n_records > 0 and header:
            writer = csv.writer(fh, quoting=csv.QUOTE_ALL)

        while True:
            if time.time() - started > budget_s:
                log(
                    f"TIME BUDGET reached ({budget_s}s). "
                    f"rows={rows_written} next_offset={offset} - re-run to continue."
                )
                write_state(offset)
                return 0

            payload = fetch_page(offset)
            feats = payload.get("features", [])
            if not feats:
                log(f"NO MORE FEATURES at offset={offset}")
                break

            if writer is None:
                header = list(feats[0]["attributes"].keys())
                writer = csv.writer(fh, quoting=csv.QUOTE_ALL)
                writer.writerow(header)
                log(f"Columns ({len(header)}): {header}")

            for feat in feats:
                attrs = feat["attributes"]
                writer.writerow([attrs.get(col) for col in header])
            fh.flush()

            rows_written += len(feats)
            pages_this_run += 1
            exceeded = payload.get("exceededTransferLimit")
            log(
                f"offset={offset:>7}  got={len(feats):>5}  "
                f"total={rows_written:>7}  exceeded={exceeded}"
            )

            offset += len(feats)
            write_state(offset)

            if len(feats) < PAGE_SIZE or not exceeded:
                log(f"LAST PAGE reached at offset={offset}")
                break

    write_state("COMPLETE")
    log(f"DONE rows_written={rows_written} pages_this_run={pages_this_run} file={OUT_PATH}")
    if rows_written == 0:
        log("ERROR: zero rows downloaded")
        return 1
    if rows_written != EXPECTED_ROWS_HINT:
        log(
            f"NOTE: row count {rows_written} != recon hint {EXPECTED_ROWS_HINT} "
            "(live dataset may have changed; treated as informational)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())