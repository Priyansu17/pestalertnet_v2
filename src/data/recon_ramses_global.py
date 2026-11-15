"""Phase 1b (scope widened to GLOBAL, owner Option B) - read-only reconnaissance.

Quantifies where the positive signal (Locust_Presence == PRESENT) lives in the
GLOBAL FAO RAMSES Desert Locust dataset, by country and by year, BEFORE any
bulk download. Uses ArcGIS statistics queries (cheap: counts only, no geometry,
no attribute payloads).

This script performs NO download, NO labeling, NO feature engineering.
Output: outputs/recon_ramses_global.txt
"""

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
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT = PROJECT_ROOT / "outputs" / "recon_ramses_global.txt"
MAX_RETRIES = 4
RETRY_SLEEP_S = 5


def query(params: dict) -> dict:
    url = BASE + "?" + urllib.parse.urlencode(params)
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "PestAlertNet-Phase1b/1.0"}
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            if "error" in payload:
                raise RuntimeError(f"service error: {payload['error']}")
            return payload
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(f"  attempt {attempt} failed: {exc}", flush=True)
            time.sleep(RETRY_SLEEP_S)
    raise RuntimeError(f"query failed: {last_err}")


def count(where: str) -> int:
    p = query({"where": where, "returnCountOnly": "true", "f": "json"})
    return int(p.get("count", 0))


def group_count(where: str, group_field: str) -> dict:
    p = query(
        {
            "where": where,
            "groupByFieldsForStatistics": group_field,
            "outStatistics": json.dumps(
                [
                    {
                        "statisticType": "count",
                        "onStatisticField": "OBJECTID",
                        "outStatisticFieldName": "cnt",
                    }
                ]
            ),
            "orderByFields": "cnt DESC",
            "returnGeometry": "false",
            "f": "json",
        }
    )
    out = {}
    for f in p.get("features", []):
        attrs = f.get("attributes", {})
        key = attrs.get(group_field)
        out[str(key)] = int(attrs.get("cnt", 0))
    return out


def main() -> int:
    lines = []
    lines.append("=== Phase 1b recon: GLOBAL RAMSES Desert Locust dataset ===")
    lines.append("endpoint: " + BASE)
    lines.append("")

    total = count("1=1")
    lines.append(f"GLOBAL total observations: {total}")

    present = count("Locust_Presence='PRESENT'")
    lines.append(
        f"GLOBAL Locust_Presence='PRESENT': {present}"
        f" ({100.0 * present / total:.2f}%)"
    )
    lines.append("")

    lines.append("--- PRESENT counts by YEAR (global) ---")
    by_year = group_count("Locust_Presence='PRESENT'", "Year")
    for k, v in sorted(by_year.items(), key=lambda kv: kv[0]):
        lines.append(f"  {k}: {v}")
    lines.append("")

    lines.append("--- ALL observations by YEAR (survey effort, global) ---")
    all_year = group_count("1=1", "Year")
    for k, v in sorted(all_year.items(), key=lambda kv: kv[0]):
        lines.append(f"  {k}: {v}")
    lines.append("")

    lines.append("--- PRESENT counts by COUNTRY (global, top 40) ---")
    by_country = group_count("Locust_Presence='PRESENT'", "Country")
    for k, v in list(by_country.items())[:40]:
        lines.append(f"  {k}: {v}")
    lines.append(f"  (distinct countries with >=1 PRESENT: {len(by_country)})")
    lines.append("")

    lines.append("--- countries with PRESENT in 2025 or 2026 ---")
    for yr in ("2025", "2026"):
        sub = group_count(f"Locust_Presence='PRESENT' AND Year={yr}", "Country")
        lines.append(f"  {yr}: {len(sub)} countries, total {sum(sub.values())}")
        for k, v in list(sub.items())[:20]:
            lines.append(f"      {k}: {v}")
    lines.append("ALL-RECON-DONE")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWROTE {REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
