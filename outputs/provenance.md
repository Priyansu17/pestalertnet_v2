# PestAlertNet — Data Provenance Record

**Project phase:** Phase 1b (L1 Data Layer)
**Record created:** 2026-08-22
**Status:** Raw ingestion + schema validation complete for both approved fallback sources.
**Scope at this phase:** ingestion / validation / provenance only. No labeling,
no feature engineering, no preprocessing, no synthetic data.

---

## 0. Primary source attempt (CropPest DSS) — FAILED, fallback triggered

Per Decision Sheet §2, scripted access to CropPest DSS (ICAR-CRIDA) was attempted
first, inside the 20-minute time box, on 2026-08-22.

- 7 candidate hosts probed. 5 were unreachable (no connection), 2 returned an
  identical interception fingerprint (`403 Forbidden`, `Server: Caddy`,
  `Alt-Svc h3=":8443"`) on unrelated government domains — consistent with a local
  network interception layer rather than a genuine service response.
- Control probes to neutral hosts succeeded (`example.com`,
  `power.larc.nasa.gov`, `www.fao.org` → all HTTP 200), so the block is
  selective, not a total loss of connectivity.
- Evidence logs: `outputs/probe_hosts.txt`, `outputs/probe_bodies.txt`,
  `outputs/probe_net.txt`.
- Consequence: Step 2 (fallback) triggered automatically. Per Decision Sheet §2,
  no further CropPest DSS attempts are made for the remainder of the project.

---

## 1. Pest observations — FAO Desert Locust (RAMSES / Locust Hub)

| Field | Value |
|---|---|
| Source name | FAO Locust Hub — RAMSES Observations |
| ArcGIS item title | "Ramses Observations Dashboard View" (item `ed8228a11a2547178cdf4ab002464efb`) |
| Publisher | FAO — Desert Locust Information Service (DLIS) |
| Official portal | https://locust-hub-hqfao.hub.arcgis.com/ |
| Discovery path | https://www.fao.org/ag/locusts/en/info/info/index.html → official Hub site → ArcGIS org item search |
| Endpoint | `https://services5.arcgis.com/sjP4Ugu5s0dZWLjd/arcgis/rest/services/Ramses_Observations_Dashboard_View/FeatureServer/0` |
| Org ID | `sjP4Ugu5s0dZWLjd` |
| Layer | `0 — initial_data_publication` (point geometry) |
| Global record count | 174,664 (verified by `returnCountOnly` query) |
| India record count | 12,785 (verified by `returnCountOnly` with `Country='India'`) |
| Access method | ArcGIS REST `query`: `where=Country='India'`, `outFields=*`, `returnGeometry=false`, `orderByFields=OBJECTID`, offset pagination at 1000 records/page, `f=json` |
| Downloader | `src/data/download_ramses_india.py` (stdlib `urllib` only, ×4 retries/page) |
| Retrieval log | `outputs/download_ramses_log.txt` |
| File | `data/raw/ramses_observations_india.csv` (6,181,485 bytes) |
| Retrieved | 2026-08-22 |
| Rows retrieved | 12,785 — exact match to the service count for India |
| Columns | 72 |
| Validation report | `outputs/validation_ramses.txt` |

### 1.1 Access-method notes (empirically verified)

- The service **rejects `f=csv`** with HTTP 400 `Bad Request`. JSON is the only
  working output format here.
- Offset pagination **requires `orderByFields=OBJECTID`**; without it the paging
  does not return a stable order.
- Verified working variants are recorded in `outputs/variant_test.txt`.
- The ArcGIS Hub *search* API on the portal returns 0 datasets, so the data is
  reachable only via the underlying feature service (org-ID discovery via the
  portal homepage). This path is reproducible from the recorded scripts.

### 1.2 Temporal and geographic coverage

| Dimension | Value |
|---|---|
| `Obs_Date` range | 2023-01-10 .. 2026-07-29 |
| Row counts by `Year` | 2023: 3,845 · 2024: 3,694 · 2025: 3,170 · 2026: 2,076 |
| Geography | India — Rajasthan / Gujarat desert locust zone |
| Latitude range | 21.6 .. 30.1 |
| Longitude range | 68.5 .. 75.9 |
| Distinct districts | 32 |
| Top districts | Barmer (1,660), Jodhpur (1,532), Jaisalmer (1,467), Bikaner (1,313), Kachchh (1,201), Nagaur (943), Ganganagar (693), Churu (669), Jalor (615), Banas Kantha (585) |
| Distinct 0.1° coordinate cells | 1,947 |

### 1.3 Schema inventory — labeling-relevant categorical fields

All 72 columns and the full missing-value audit are in
`outputs/validation_ramses.txt`. The fields that matter for labelling:

| Field | Distinct | Counts |
|---|---|---|
| `Locust_Presence` | 2 | ABSENT 12,709 · PRESENT 76 |
| `Locust_Type` | 4 | NO LOCUST 12,709 · ADULT 43 · HOPPER 27 · HOPPER GROUPS 6 |
| `RAMSES_Locust_Type` | 3 | NO LOCUST 12,709 · ADULT 43 · HOPPER 33 |
| `Presence_Type` | 5 | N/A 12,709 · Unknown Adult Low Density 28 · Hoppers Unknown 27 · Immature Adult Low Density 15 · Hopper Group 6 |
| `Behaviour` | 7 | N/A 12,709 · ISOLATED 38 · SCATTERED 16 · SCATTERED \| ISOLATED 11 · GROUPS 6 · GROUPS \| SCATTERED 4 · GROUPS \| SCATTERED \| ISOLATED 1 |
| `Density_Description` | 3 | N/A 12,742 · LOW 28 · MEDIUM 15 |
| `Density_Unit` | 3 | N/A 12,752 · ha 31 · m² 2 |
| `Activity` | 10 | N/A 12,759 · FEEDING \| MARCHING 7 · FEEDING \| MARCHING \| MOULTING 6 · FEEDING \| MARCHING \| MOULTING \| ROOSTING 4 · + 6 rarer combos |
| `Observation_Reliability` | 1 | RELIABLE 12,785 (all rows) |
| `Observation_Confirmation` | 1 | CONFIRMED 12,785 (all rows) |
| `Data_Collection_Method` | 5 | eL3m Pro Android 5,936 · eL3K 4,904 · eL3 1,798 · Manual Entry 142 · eL3m basic Android 5 |

**Critical schema findings relevant to Decision Sheet §4:**

1. **There is no numeric `pest_count` / abundance column.** The dataset is
   survey-based *presence/absence* plus qualitative density classes (`LOW`,
   `MEDIUM`) — no population counts. §4 option 2 (a published ETL expressed as a
   pest count) has no numeric column to be compared against.
2. **There is no FAO situation-classification field** (Recession / Outbreak /
   Upsurge / Plague) among these 72 columns. §4 option 1 is therefore not
   directly satisfiable from this export.
3. The closest existing positive signal (`Locust_Presence = PRESENT`) is
   **76 of 12,785 rows (0.6%)** at row level.
4. Per the approved instruction, the labelling rule is **not** invented or
   substituted here; Phase 1c presents the options derived from the real columns
   and waits for explicit approval.

---

## 2. Weather — NASA POWER (daily, regional)

| Field | Value |
|---|---|
| Source name | NASA POWER (Prediction Of Worldwide Energy Resources) |
| Publisher | NASA Langley Research Center / ASDC |
| Endpoint | https://power.larc.nasa.gov/api/temporal/daily/regional |
| Community | `AG` (Agroclimatology) |
| Authentication | None required (keyless, free, public) |
| Parameters | `T2M` (temp @2 m, °C), `RH2M` (relative humidity @2 m, %), `PRECTOTCORR` (corrected precipitation, mm/day), `WS2M` (wind speed @2 m, m/s) |
| Bounding box | lat 21–31, lon 68–77 (covers the India Desert Locust observation zone) |
| Chunking | 1 parameter per request; ≤366 days per request (API limits) |
| Chunks | 2023, 2024, 2025, 2026-01-01..2026-07-29 |
| Downloader | `src/data/download_power_regional.py` (stdlib `urllib`, ×4 retries/chunk) |
| Retrieval log | `outputs/download_power_log.txt` (first pass, 4 transient failures), `outputs/download_power_log2.txt` (recovery pass, `failures=[]`) |
| Files | 16 files, `data/raw/power_{PARAM}_{YYYY}.csv` |
| Date coverage | 2023-01-01 .. 2026-07-29 |
| Grid | 315 cells — 21 lat cells (21.0..31.0, step 0.5°) × 15 lon cells (68.125..76.875, step 0.625°) |
| Rows per parameter | 411,390 |
| Missing-data sentinel | `-999` (documented by the API); **0 sentinel rows observed** in any of the 16 files |
| Value ranges | T2M −7.65..43.41 (mean 26.51) · RH2M 4.08..97.16 (mean 48.65) · PRECTOTCORR 0.00..817.64 (mean 1.90) · WS2M 0.34..16.18 (mean 2.56) |
| Validation report | `outputs/validation_power.txt` |

### 2.1 Access-method notes (empirically verified)

- The regional endpoint rejects **more than 1 parameter per request**.
- The bbox must span **at least 2 degrees** in both latitude and longitude
  (otherwise the point endpoint must be used).
- The time period per request must be **≤ 366 days**.
- The network was intermittently flaky during the first pass (DNS `getaddrinfo`
  failures and `WinError 10054` connection resets). The downloader's idempotent
  SKIP logic allowed a clean recovery pass with zero remaining failures. No data
  was synthesised or interpolated to cover the gap.

---

## 3. How the two sources join

- **Spatial key:** nearest NASA POWER grid cell to each RAMSES observation's
  `Latitude`/`Longitude`.
- **Temporal key:** RAMSES `Obs_Date` is Unix epoch **milliseconds** (converted to
  a UTC date during ingestion); POWER uses `YEAR` + `DOY` (day-of-year), converted
  to a calendar date. Both are daily-granularity, so the join is exact at day level.
- **Aggregation requirement:** RAMSES is *point-level* (many observations per
  day/cell) and POWER is *cell-level* (one row per cell per day). Before any model
  input is built, RAMSES must be aggregated to a daily (cell × day) table — that
  belongs to Phase 2, not Phase 1b.
- **Spatial support mismatch (limitation):** RAMSES coordinates are precise
  (1,947 distinct 0.1° cells) whereas POWER is a coarse 0.5°×0.625° grid
  (315 cells). Several observation points therefore map to the same weather cell.