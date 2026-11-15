"""Phase 1b - READ-ONLY metadata inspection of the FAO Locust Hub ArcGIS org.

Lists every service published by org sjP4Ugu5s0dZWLjd and flags the ones whose
name relates to locust / RAMSES / swarms / hexbins, so the labeling options
presented at the Phase 1c gate are complete.

No data downloaded here beyond service metadata. Writes
outputs/fao_org_services.txt
"""

import json
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_PATH = PROJECT_ROOT / "outputs" / "fao_org_services.txt"

SERVICES_URL = (
    "https://services5.arcgis.com/sjP4Ugu5s0dZWLjd/arcgis/rest/services?f=json"
)
KEYWORDS = ("locust", "ramses", "swarm", "hexbin", "hex_", "dlis", "survey")


def fetch_json(url: str) -> dict:
    req = urllib.request.Request(
        url, headers={"User-Agent": "PestAlertNet-Phase1b/1.0"}
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    payload = fetch_json(SERVICES_URL)
    (RAW_DIR / "fao_org_services.json").write_text(
        json.dumps(payload, indent=1), encoding="utf-8"
    )

    services = payload.get("services", [])
    lines = []
    add = lines.append
    add("=== FAO Locust Hub org (sjP4Ugu5s0dZWLjd) published services ===")
    add(f"total services: {len(services)}")
    add("")
    add("--- KEYWORD-MATCHED SERVICES (locust/ramses/swarm/hexbin/dlis/survey) ---")
    matched = []
    for s in services:
        name = s.get("name", "")
        if any(k in name.lower() for k in KEYWORDS):
            matched.append(s)
            add(f"{s.get('type')} | {name}")
    if not matched:
        add("(none)")
    add("")
    add("--- ALL SERVICES ---")
    for s in services:
        add(f"{s.get('type')} | {s.get('name')}")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"WROTE {REPORT_PATH}")
    print(f"total services={len(services)} matched={len(matched)}")
    for s in matched:
        print(f"  MATCH {s.get('name')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
