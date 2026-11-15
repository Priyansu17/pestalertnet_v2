"""Read-only throughput probe: time N paginated RAMSES page fetches."""
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

BASE = (
    "https://services5.arcgis.com/sjP4Ugu5s0dZWLjd/arcgis/rest/services/"
    "Ramses_Observations_Dashboard_View/FeatureServer/0/query"
)
OUT = Path(__file__).resolve().parents[2] / "outputs" / "perf_probe.txt"


def fetch(offset):
    params = {
        "where": "1=1",
        "outFields": "*",
        "returnGeometry": "false",
        "orderByFields": "OBJECTID",
        "resultOffset": str(offset),
        "resultRecordCount": "1000",
        "f": "json",
    }
    url = BASE + "?" + urllib.parse.urlencode(params)
    t0 = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": "PestAlertNet-Perf/1.0"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        raw = resp.read()
    dt = time.time() - t0
    payload = json.loads(raw.decode("utf-8"))
    return dt, len(raw), len(payload.get("features", []))


lines = []
for off in (0, 50000, 120000):
    dt, nbytes, nfeat = fetch(off)
    lines.append(f"offset={off} seconds={dt:.1f} bytes={nbytes} features={nfeat}")
    print(lines[-1], flush=True)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(lines), encoding="utf-8")