"""Phase 1b read-only probe: POWER *point* endpoint capabilities.

The *regional* endpoint (already used for the India download) is expensive for
a global scope: max 1 parameter/request, >=2 deg bbox, <=366 days/request.
The *point* endpoint may accept multiple parameters and long date ranges,
which would make per-cell acquisition cheap.

This probe measures, for a single representative locust-zone cell
(Saudi Arabia interior, lat 24 / lon 46):
    - does the point endpoint accept 4 parameters in one request?
    - does it accept the full 2023-01-01..2026-07-29 range in one request?
    - response size and shape (CSV columns)
    - wall-clock latency (to extrapolate cost over N cells)

Read-only. Writes outputs/power_point_probe.txt only.
"""

import time
import urllib.parse
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = PROJECT_ROOT / "outputs" / "power_point_probe.txt"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "probe"

BASE = "https://power.larc.nasa.gov/api/temporal/daily/point"


def build_url(params, lat, lon, start, end):
    q = {
        "parameters": params,
        "community": "AG",
        "longitude": str(lon),
        "latitude": str(lat),
        "start": start,
        "end": end,
        "format": "CSV",
    }
    return BASE + "?" + urllib.parse.urlencode(q)


def fetch(url, timeout=180):
    req = urllib.request.Request(url, headers={"User-Agent": "PestAlertNet-Phase1b/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("=== Phase 1b probe: NASA POWER /point endpoint capability ===")
    lines.append("")

    cases = [
        ("4 params, full range (2023-01-01..2026-07-29)",
         "T2M,RH2M,PRECTOTCORR,WS2M", 24.0, 46.0, "20230101", "20260729"),
        ("4 params, single year",
         "T2M,RH2M,PRECTOTCORR,WS2M", 24.0, 46.0, "20230101", "20231231"),
        ("8 params, full range",
         "T2M,T2M_MAX,T2M_MIN,RH2M,PRECTOTCORR,WS2M,WS2M_MAX,EVPTRNS",
         24.0, 46.0, "20230101", "20260729"),
    ]

    for label, params, lat, lon, start, end in cases:
        url = build_url(params, lat, lon, start, end)
        t0 = time.time()
        try:
            text = fetch(url)
            dt = time.time() - t0
        except Exception as exc:  # noqa: BLE001
            lines.append(f"CASE: {label}")
            lines.append(f"  RESULT: FAILED after {time.time()-t0:.1f}s -> {exc}")
            lines.append("")
            continue

        lines.append(f"CASE: {label}")
        lines.append(f"  URL: {url}")
        lines.append(f"  status: OK  latency={dt:.2f}s  bytes={len(text)}")
        if text.lstrip().startswith("{"):
            lines.append(f"  API-ERROR-BODY: {text[:300]}")
            lines.append("")
            continue
        body = [ln for ln in text.splitlines()]
        header_rows = [ln for ln in body if ln.startswith("-") or "PARAMETER" in ln.upper()]
        lines.append(f"  total lines: {len(body)}")
        for ln in body[:12]:
            lines.append(f"    {ln}")
        lines.append("")
        fname = RAW_DIR / (label.replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")[:40] + ".csv")
        fname.write_text(text, encoding="utf-8")

    lines.append("POINT-PROBE-DONE")
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"WROTE {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
