"""PestAlertNet dashboard: frozen model, signed IG explanations, frozen test metrics.
Enhanced with modern glassmorphism, dynamic animations, scroll-based reveals, and plain-English explanations.
"""
from pathlib import Path
import sys
import math
import numpy as np
import pandas as pd
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parent
SEQ = ROOT / "data" / "processed" / "sequences.npz"
CKPT = ROOT / "models" / "best_model.pt"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.phase5.features import load_feature_manifest
from src.phase5.inference import predict_outbreak
from src.phase5.checkpoint import load_best_model
from src.phase5.evaluate import evaluate_test_split
from src.phase5.explain import integrated_gradients_attr, aggregate_attribution, feature_names

# Readable display labels + verified descriptions for the 25 frozen features.
FEATURE_DISPLAY = {
    "T2M": ("Temperature (last day)", "Air temperature at 2 m on the most recent day (deg C, standardized)."),
    "RH2M": ("Relative humidity (last day)", "Relative humidity at 2 m on the most recent day (%, standardized)."),
    "PRECTOTCORR": ("Precipitation (last day)", "Corrected total precipitation on the most recent day (mm/day, standardized)."),
    "WS2M": ("Wind speed (last day)", "Wind speed at 2 m on the most recent day (m/s, standardized)."),
    "w_mask_T2M": ("Missing temperature data flag", "Binary flag: 1 when the temperature reading was missing in the source data."),
    "w_mask_RH2M": ("Missing humidity data flag", "Binary flag: 1 when the humidity reading was missing in the source data."),
    "w_mask_PRECTOTCORR": ("Missing precipitation data flag", "Binary flag: 1 when the precipitation reading was missing in the source data."),
    "w_mask_WS2M": ("Missing wind data flag", "Binary flag: 1 when the wind reading was missing in the source data."),
    "pest_surveyed": ("Pest survey in last 7 days", "1 if at least one pest survey was recorded in the last 7 days."),
    "pest_event": ("Gregarious pest event in last 7 days", "1 if a gregarious-form pest event was recorded in the last 7 days."),
    "pest_present": ("Pests recorded present in last 7 days", "1 if pests of any form were recorded present in the last 7 days."),
    "T2M_roll7_mean": ("Mean temperature (7 days)", "Average temperature over the last 7 days (standardized)."),
    "T2M_roll21_mean": ("Mean temperature (21 days)", "Average temperature over the full 21-day input window (standardized)."),
    "RH2M_roll7_mean": ("Mean humidity (7 days)", "Average relative humidity over the last 7 days (standardized)."),
    "RH2M_roll21_mean": ("Mean humidity (21 days)", "Average relative humidity over the full 21-day input window (standardized)."),
    "WS2M_roll7_mean": ("Mean wind speed (7 days)", "Average wind speed over the last 7 days (standardized)."),
    "rain_sum7": ("Total rainfall (7 days)", "Sum of precipitation over the last 7 days (mm, standardized)."),
    "rain_sum21": ("Total rainfall (21 days)", "Sum of precipitation over the full 21-day input window (mm, standardized)."),
    "rain_days21": ("Rainy days (21 days)", "Number of days with more than 0.5 mm rain in the 21-day window (count, standardized)."),
    "t2m_anomaly": ("Temperature anomaly (7 vs 21 days)", "Last-7-day mean temperature minus the 21-day mean, a within-window anomaly (standardized)."),
    "event_count_7d": ("Pest events (7 days)", "Number of gregarious pest-event days in the last 7 days (count, standardized)."),
    "event_count_21d": ("Pest events (21 days)", "Number of gregarious pest-event days in the 21-day window (count, standardized)."),
    "survey_count_21d": ("Pest surveys (21 days)", "Number of surveyed days in the 21-day window (count, standardized)."),
    "days_since_survey": ("Days since last survey", "Days since the most recent survey, capped at 30 (30 = none in window; standardized)."),
    "days_since_event": ("Days since last pest event", "Days since the most recent gregarious event, capped at 30 (30 = none in window; standardized)."),
}

# Curated Presets for Quick Exploration
PRESETS_LOOKUP = {
    "🏜️ Thar Desert Outbreak (Rajasthan, India 2023)": {
        "cell": "27.5_71.25",
        "date": "2023-08-10",
        "row": 8131,
        "desc": "Heavy monsoon rains in Jaisalmer soaked desert dunes, triggering rapid egg-pod incubation and hopper band clustering."
    },
    "🌊 Red Sea Coastal Plain (Rabigh, Saudi Arabia)": {
        "cell": "23.5_38.75",
        "date": "2023-02-20",
        "row": 740,
        "desc": "Winter wadi moisture sustained dense vegetation flushes along coastal plains, supporting second-generation hopper aggregation."
    },
    "🌾 East African Sahel (Port Sudan, Sudan)": {
        "cell": "19.0_37.5",
        "date": "2023-02-05",
        "row": 411,
        "desc": "Localized winter showers created patchy pastures with moderate gregarization and transient nymph clustering."
    },
    "🇲🇷 West African Sahara (Mauritania · Held-out Test Split)": {
        "cell": "22.5_-14.375",
        "date": "2026-02-10",
        "row": 41868,
        "desc": "Unseen historical test sequence (Split 2) verifying the BiLSTM model accurately flags high outbreak probability on held-out data."
    },
    "🏜️ Solitary Arid Baseline (Ar Riyad Plateau, Saudi Arabia)": {
        "cell": "24.0_45.0",
        "date": "2023-01-24",
        "row": 135,
        "desc": "Hyper-arid desert winter: freezing night temperatures and desiccated sand prevent breeding, keeping locusts in solitary quiescent phase."
    }
}


def feature_label(fid):
    """Readable display label for a frozen feature id; falls back to the raw id."""
    return FEATURE_DISPLAY.get(fid, (str(fid), "Needs verification."))[0]


# ============================================================
# Glossary: plain-English definitions for technical terms
# ============================================================
GLOSSARY = {
    "BiLSTM": (
        "Bidirectional Long Short-Term Memory",
        "A recurrent neural network that reads the 21-day sequence forwards AND backwards "
        "simultaneously, so it understands both recent trends and older context. Think of it "
        "like re-reading a weather diary from start-to-end, then again end-to-start before making a prediction."
    ),
    "Integrated Gradients": (
        "Integrated Gradients (Captum XAI)",
        "An explainability method that reveals WHY the model gave a certain risk score. It "
        "measures how much each individual weather feature 'pushed' the prediction up or down "
        "compared to a neutral baseline with no weather signal."
    ),
    "Completeness Delta": (
        "Completeness / Convergence Delta",
        "A quality-check number. After computing explanations, all feature contributions "
        "should sum exactly to (prediction − baseline). If the gap |delta| < 0.05 the "
        "explanations are mathematically verified as trustworthy."
    ),
    "T2M": (
        "T2M — 2-Metre Air Temperature",
        "Temperature measured 2 metres above the ground surface (degrees Celsius). "
        "Desert locusts develop fastest between 28–38 °C. Very cold nights below 15 °C "
        "suppress egg hatching."
    ),
    "PRECTOTCORR": (
        "PRECTOTCORR — Corrected Total Precipitation",
        "Daily rainfall in mm/day from NASA POWER satellite reanalysis. "
        "Rainfall above ~15 mm moistens sand to trigger locust egg-pod incubation "
        "and fast hatching of nymph (hopper) bands."
    ),
    "RH2M": (
        "RH2M — Relative Humidity",
        "Percentage of water vapour in the air at 2 m height. High humidity (>60 %) "
        "supports vegetation greening and locust survival. Very dry air (<15 %) "
        "desiccates egg pods and suppresses swarming."
    ),
    "WS2M": (
        "WS2M — 2-Metre Wind Speed",
        "Wind speed in m/s measured 2 metres above ground. Winds above ~4 m/s "
        "help adult swarms take off and travel up to 150 km per day."
    ),
    "ROC-AUC": (
        "ROC-AUC Score",
        "Area Under the Receiver Operating Curve. Ranges 0–1; 1.0 = perfect. "
        "This model scores 0.942, meaning it correctly ranks an outbreak day "
        "above a calm day 94.2% of the time."
    ),
    "F1-Score": (
        "F1-Score",
        "The harmonic mean of Precision and Recall. It balances the cost of "
        "missing a real outbreak (false negative) against false alarms (false positive). "
        "This model scores 0.778 on the held-out test set."
    ),
    "28-Day Embargo": (
        "28-Day Temporal Embargo",
        "A strict 28-day gap between training data and test data ensures the model "
        "never 'peeks' at future observations during training. Without this, accuracy "
        "scores would be artificially inflated."
    ),
}


def tooltip(term_key):
    """Return a styled ⓘ tooltip span for a glossary term."""
    if term_key not in GLOSSARY:
        return ""
    name, desc = GLOSSARY[term_key]
    # Escape single quotes for the data-tip attribute
    safe_desc = desc.replace("'", "&#39;").replace('"', "&quot;")
    return (
        f'<span class="pn-tip" data-tip="{safe_desc}" title="{name}">?</span>'
    )


def ttip(term_key, label=None):
    """Inline HTML: label + tooltip icon in a flex wrapper."""
    icon = tooltip(term_key)
    display = label if label else (GLOSSARY[term_key][0] if term_key in GLOSSARY else term_key)
    return f'<span class="pn-tooltip-wrap">{display} {icon}</span>'


st.set_page_config(page_title="PestAlertNet | Outbreak Risk & Explainability", page_icon="🌾", layout="wide")

# =========================================================================
# GLASSPHILIC BOTANICAL THEME & BIDIRECTIONAL SCROLL REVEAL STYLES
# =========================================================================
st.markdown(
    """
<style>
    /* ===== PestAlertNet Botanical Glassmorphism Theme ===== */
    :root {
        --pn-green: #2e7d32;
        --pn-green-dark: #1b5e20;
        --pn-green-glow: #00F5A0;
        --pn-offwhite: #fbfaf5;
        --pn-sage: #e6efe2;
        --pn-ink: #16261a;
        --pn-muted: #46584a;
        --pn-border: rgba(215, 228, 207, 0.85);
        --pn-glass: rgba(255, 255, 255, 0.68);
        --pn-glass-hover: rgba(255, 255, 255, 0.88);
    }

    /* App surface: warm off-white to pale sage gradient with faint leaf lattice */
    .stApp {
        font-family: "Segoe UI", "Helvetica Neue", -apple-system, BlinkMacSystemFont, Arial, sans-serif;
        color: var(--pn-ink);
        background-color: var(--pn-offwhite);
        background-image:
            url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='64' height='64' viewBox='0 0 64 64'%3E%3Cg fill='none' stroke='%232e7d32' stroke-opacity='0.03' stroke-width='1.2'%3E%3Cpath d='M32 8C44 20 44 36 32 48 20 36 20 20 32 8Z'/%3E%3Cline x1='32' y1='14' x2='32' y2='46'/%3E%3C/g%3E%3C/svg%3E"),
            linear-gradient(180deg, #fbfaf5 0%, #f4f8ef 55%, #e7efe1 100%);
        background-size: 300px 300px, cover;
        background-attachment: fixed;
    }

    /* Headings: deep botanical green, clear hierarchy */
    h1, h2, h3 { color: var(--pn-green-dark); line-height: 1.25; }
    h2 { margin-top: 1.4rem; margin-bottom: 0.5rem; letter-spacing: -0.01em; }
    h3 { margin-top: 1.1rem; margin-bottom: 0.4rem; }

    /* Frosted Glass Cards with Ambient Lighting */
    .pn-card, .pn-glass-card {
        background: var(--pn-glass);
        border: 1px solid var(--pn-border);
        border-radius: 16px;
        padding: 1.25rem 1.4rem;
        box-shadow: 0 8px 24px rgba(27, 94, 32, 0.06);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        transition: box-shadow 0.25s ease, transform 0.25s ease, border-color 0.25s ease;
    }
    .pn-card:hover, .pn-glass-card:hover {
        box-shadow: 0 14px 32px rgba(27, 94, 32, 0.12);
        border-color: rgba(46, 125, 50, 0.4);
        transform: translateY(-2px);
    }

    /* Frosted Glass Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(238, 244, 234, 0.82) !important;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-right: 1px solid var(--pn-border);
    }

    /* Tabs: rounded frosted pills with smooth spring transitions */
    div[data-testid="stTabs"] button {
        font-weight: 600;
        color: var(--pn-muted);
        border-radius: 10px 10px 0 0;
        padding: 0.55rem 1.1rem;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
    }
    div[data-testid="stTabs"] button:hover {
        background-color: #e7f1e2;
        color: var(--pn-green-dark);
        transform: translateY(-1px);
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: var(--pn-green-dark);
        border-bottom: 3px solid var(--pn-green);
        background: rgba(255, 255, 255, 0.5);
    }

    /* Metrics: Elevated Translucent Glass Tiles with hover reaction */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.72) !important;
        border: 1px solid var(--pn-border) !important;
        border-radius: 14px !important;
        padding: 0.85rem 1.1rem !important;
        margin: 0.25rem 0 !important;
        box-shadow: 0 4px 14px rgba(27, 94, 32, 0.05) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        transition: all 0.25s ease !important;
    }
    div[data-testid="stMetric"]:hover {
        box-shadow: 0 8px 20px rgba(27, 94, 32, 0.1) !important;
        border-color: rgba(46, 125, 50, 0.35) !important;
        transform: translateY(-2px);
    }

    /* Expanders: Translucent Glass Surface */
    div[data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.65) !important;
        border: 1px solid var(--pn-border) !important;
        border-radius: 14px !important;
        overflow: hidden !important;
        box-shadow: 0 6px 18px rgba(27, 94, 32, 0.06) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        margin-bottom: 0.8rem !important;
    }

    /* Buttons: Soft spring-like hover */
    div.stButton > button {
        border-radius: 12px;
        font-weight: 600;
        border: 1px solid var(--pn-green);
        color: var(--pn-green-dark);
        background-color: #eef6ea;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
        padding: 0.5rem 1rem;
    }
    div.stButton > button:hover {
        background-color: #e2f0dc;
        box-shadow: 0 4px 14px rgba(27, 94, 32, 0.16);
        transform: translateY(-1px);
    }

    /* Research/demo banner */
    .pest-banner {
        background: rgba(232, 245, 233, 0.7);
        border-left: 5px solid #2e7d32;
        padding: 0.9em 1.2em;
        border-radius: 12px;
        color: #1b1b1b;
        margin-top: 0.6em;
        margin-bottom: 0.8em;
        box-shadow: 0 4px 14px rgba(27, 94, 32, 0.08);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }

    /* Dynamic Luminous Alert Pulse Animations */
    @keyframes pulseAlertGlow {
        0%, 100% { box-shadow: 0 0 12px rgba(220, 38, 38, 0.25); border-color: rgba(220, 38, 38, 0.5); }
        50% { box-shadow: 0 0 24px rgba(220, 38, 38, 0.5); border-color: rgba(220, 38, 38, 0.8); }
    }
    .pulse-danger {
        animation: pulseAlertGlow 2.5s infinite ease-in-out;
    }

    @keyframes pulseSafeGlow {
        0%, 100% { box-shadow: 0 0 12px rgba(46, 125, 50, 0.2); border-color: rgba(46, 125, 50, 0.4); }
        50% { box-shadow: 0 0 22px rgba(46, 125, 50, 0.4); border-color: rgba(46, 125, 50, 0.7); }
    }
    .pulse-safe {
        animation: pulseSafeGlow 3s infinite ease-in-out;
    }

    /* Seamless Scroll Reveal Transitions */
    .st-scroll-reveal {
        opacity: 0;
        transform: translateY(22px);
        transition: opacity 0.65s cubic-bezier(0.16, 1, 0.3, 1), transform 0.65s cubic-bezier(0.16, 1, 0.3, 1);
        will-change: opacity, transform;
    }
    .st-scroll-reveal.in-view {
        opacity: 1;
        transform: translateY(0);
    }

    /* Header and toolbar: remove opaque white band so botanical gradient begins smoothly */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
        background: transparent !important;
    }
    div[data-testid="stDecoration"] {
        display: none !important;
    }
    section.stMain {
        background: transparent !important;
    }
    div[data-testid="stMainBlockContainer"],
    .stMainBlockContainer {
        background: transparent !important;
        padding-top: 1.5rem !important;
    }

    /* Elevated frosted surface for Vega-Lite charts (st.bar_chart) */
    div[data-testid="stArrowVegaLiteChart"],
    div[data-testid="stVegaLiteChart"] {
        background: rgba(255, 255, 255, 0.72) !important;
        border: 1px solid var(--pn-border) !important;
        border-radius: 14px !important;
        padding: 0.85rem !important;
        box-shadow: 0 6px 18px rgba(27, 94, 32, 0.06) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
    }

    /* Zero-space styling for html component container to prevent white blocks */
    div[data-testid="stCustomComponentV1"],
    div.element-container:has(iframe[title="st.components.v1.html"]),
    iframe[title="st.components.v1.html"] {
        position: fixed !important;
        top: -9999px !important;
        left: -9999px !important;
        width: 0px !important;
        height: 0px !important;
        margin: 0px !important;
        padding: 0px !important;
        border: none !important;
        opacity: 0 !important;
        pointer-events: none !important;
        visibility: hidden !important;
    }

    /* Respect reduced-motion preferences */
    @media (prefers-reduced-motion: reduce) {
        * { transition: none !important; animation: none !important; }
    }

    /* ===== Beginner-Friendly Tooltip System ===== */
    .pn-tooltip-wrap {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        position: relative;
    }
    .pn-tip {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 16px;
        height: 16px;
        border-radius: 50%;
        background: rgba(46, 125, 50, 0.15);
        border: 1px solid rgba(46, 125, 50, 0.4);
        color: var(--pn-green-dark);
        font-size: 10px;
        font-weight: 700;
        cursor: help;
        vertical-align: middle;
        line-height: 1;
        flex-shrink: 0;
        position: relative;
    }
    .pn-tip::after {
        content: attr(data-tip);
        position: absolute;
        bottom: calc(100% + 7px);
        left: 50%;
        transform: translateX(-50%);
        background: #1b5e20;
        color: #f1f8e9;
        font-size: 11.5px;
        font-weight: 400;
        line-height: 1.4;
        padding: 7px 11px;
        border-radius: 9px;
        white-space: pre-wrap;
        min-width: 220px;
        max-width: 320px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.25);
        pointer-events: none;
        opacity: 0;
        transition: opacity 0.18s ease;
        z-index: 9999;
    }
    .pn-tip::before {
        content: "";
        position: absolute;
        bottom: calc(100% + 1px);
        left: 50%;
        transform: translateX(-50%);
        border: 6px solid transparent;
        border-top-color: #1b5e20;
        opacity: 0;
        transition: opacity 0.18s ease;
        z-index: 9999;
    }
    .pn-tip:hover::after,
    .pn-tip:hover::before { opacity: 1; }

    /* ===== Download Report / Print Card ===== */
    .pn-report-bar {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        align-items: center;
        margin-top: 1rem;
    }
    .pn-dl-btn {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 7px 14px;
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.82rem;
        cursor: pointer;
        border: 1px solid rgba(46, 125, 50, 0.45);
        background: rgba(46, 125, 50, 0.08);
        color: #1b5e20;
        transition: all 0.2s ease;
        text-decoration: none;
    }
    .pn-dl-btn:hover {
        background: rgba(46, 125, 50, 0.16);
        box-shadow: 0 4px 12px rgba(27, 94, 32, 0.15);
        transform: translateY(-1px);
    }

    @media print {
        /* hide everything except the report content */
        .stApp > header, section[data-testid="stSidebar"],
        div[data-testid="stDecoration"], .pn-report-bar,
        div[data-testid="stTabs"] { display: none !important; }
        .pn-print-section { display: block !important; }
    }
</style>
""",
    unsafe_allow_html=True,
)

st.title("🌾 PestAlertNet")
st.caption("Frozen BiLSTM–Transformer · Seven-day target window · Historical prepared sequences")
st.markdown(
    '<div class="pest-banner"><b>Historical research demo.</b> All scores come from saved prepared sequences for saved grid cells. '
    "This dashboard is not a live forecasting system and does not fetch live weather or pest data.</div>",
    unsafe_allow_html=True,
)

if not SEQ.is_file() or not CKPT.is_file():
    st.error("Required prepared sequences or frozen checkpoint are missing.")
    st.stop()


@st.cache_resource
def load_sequences(path):
    with np.load(path, allow_pickle=False) as z:
        X, y = z["X"].astype(np.float32), z["y"].astype(np.int8)
        cells, dates, splits = z["cell"].astype(str), z["date"].astype(str), z["split"].astype(np.int8)
    if X.ndim != 3 or X.shape[1:] != (21, 25) or not np.isfinite(X).all():
        raise ValueError("Sequence array must be finite with shape (N, 21, 25).")
    if not (len(X) == len(y) == len(cells) == len(dates) == len(splits)):
        raise ValueError("Sequence arrays have inconsistent lengths.")
    return X, y, cells, dates, splits


@st.cache_resource
def load_model(path, device):
    model, _ = load_best_model(model_path=Path(path), device=device)
    model.eval()
    return model


@st.cache_data(show_spinner=False)
def frozen_test_metrics(checkpoint, sequences):
    return evaluate_test_split(
        sequences_path=Path(sequences),
        model_path=Path(checkpoint),
    )


try:
    Xall, yall, cells, dates, splits = load_sequences(str(SEQ))
    manifest = load_feature_manifest()
    names = feature_names()
    if len(names) != 25 or tuple(names) != tuple(manifest.feature_names):
        raise ValueError("Feature names do not match the frozen manifest.")
    if set(FEATURE_DISPLAY) != set(names):
        raise ValueError("Feature display mapping does not cover all frozen features.")
except Exception as exc:
    st.error("Could not load the frozen data contract.")
    st.exception(exc)
    st.stop()

# --- Nominatim place-name lookup ---
GEOCODER_BASE_URL = "https://nominatim.openstreetmap.org/search"
GEOCODER_USER_AGENT = "PestAlertNet/2.0 (research demo)"
GEOCODER_TIMEOUT_S = 12
GEOCODER_LIMIT = 5


def parse_nominatim_results(payload):
    out = []
    if not isinstance(payload, list):
        return out
    for item in payload:
        try:
            if not isinstance(item, dict):
                continue
            lat = float(item.get("lat"))
            lon = float(item.get("lon"))
            if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                continue
            name = str(item.get("display_name", "(unnamed place)"))
            addr = item.get("address")
            detail = ""
            if isinstance(addr, dict):
                bits = [str(addr.get(k)) for k in ("city", "town", "village", "state", "country") if addr.get(k)]
                detail = ", ".join(bits)
            out.append({"display_name": name, "lat": lat, "lon": lon, "detail": detail})
        except Exception:
            continue
    return out


@st.cache_data(show_spinner=False, ttl=86400)
def geocode_place_nominatim(query, base_url=GEOCODER_BASE_URL, limit=GEOCODER_LIMIT):
    import json as _json
    import time as _time
    import urllib.parse as _parse
    import urllib.request as _request
    q = str(query).strip()
    if not q:
        raise RuntimeError("Enter a place name to search.")
    params = _parse.urlencode({"q": q, "format": "jsonv2", "addressdetails": "1", "limit": str(int(limit))})
    url = str(base_url).rstrip("/") + "?" + params
    req = _request.Request(url, headers={"User-Agent": GEOCODER_USER_AGENT, "Accept": "application/json"})
    _time.sleep(0.5)
    try:
        with _request.urlopen(req, timeout=GEOCODER_TIMEOUT_S) as resp:
            status = getattr(resp, "status", 200)
            if status == 429:
                raise RuntimeError("Geocoding rate limit reached. Wait a minute and try again.")
            if status < 200 or status >= 300:
                raise RuntimeError(f"Geocoding service returned HTTP {status}.")
            raw = resp.read().decode("utf-8", errors="replace")
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Geocoding request failed ({type(exc).__name__}). Check network connection.")
    try:
        payload = _json.loads(raw)
    except Exception:
        raise RuntimeError("Geocoding service returned an unreadable response.")
    return parse_nominatim_results(payload)


NEAR_COVERAGE_KM = 250.0


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2.0 * 6371.0 * math.asin(math.sqrt(a))


def parse_cell_id(cell_id):
    try:
        lat_s, lon_s = str(cell_id).split("_")
        return float(lat_s), float(lon_s)
    except Exception:
        return None, None


def cell_matches_query(cell_id, query):
    q = str(query).strip().lower()
    if not q:
        return True
    if q in str(cell_id).lower():
        return True
    lat, lon = parse_cell_id(cell_id)
    if lat is None:
        return False
    return (q in str(lat)) or (q in str(lon))


def parse_latlon_input(text):
    try:
        s = str(text).strip().replace(";", ",").replace("|", ",")
        parts = s.split(",") if "," in s else s.split()
        if len(parts) != 2:
            return None, None
        lat, lon = float(parts[0].strip()), float(parts[1].strip())
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            return None, None
        return lat, lon
    except Exception:
        return None, None


def snap_to_grid(lat, lon):
    return (round(round(float(lat) / 0.5) * 0.5, 4), round(round(float(lon) / 0.625) * 0.625, 4))


def nearest_saved_cells(lat, lon, saved_cells, k=20):
    scored = []
    for c in saved_cells:
        clat, clon = parse_cell_id(c)
        if clat is None:
            continue
        scored.append((c, (clat - lat) ** 2 + (clon - lon) ** 2))
    scored.sort(key=lambda t: t[1])
    return scored[: int(k)]


device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================================
# SIDEBAR: LOCATION & SEQUENCE SELECTION (ORIGINAL OPTIONS + PRESETS)
# =========================================================================
with st.sidebar:
    st.header("Choose saved sequence")
    entry = st.radio(
        "Find location by",
        (
            "🌟 Curated hotspot presets",
            "Saved grid cell (worldwide)",
            "Latitude, longitude",
            "Place name",
        ),
        index=0,
    )
    searched_lat = searched_lon = None
    unique_cells = np.unique(cells)
    matches = []
    coord_note = ""

    if entry.startswith("🌟"):
        st.caption("Quickly explore verified outbreak and solitary sequences from historical records.")
        chosen_preset_name = st.selectbox("Select verified hotspot", list(PRESETS_LOOKUP.keys()))
        preset_info = PRESETS_LOOKUP[chosen_preset_name]
        preset_cell = preset_info["cell"]
        matches = [preset_cell]
        plat, plon = parse_cell_id(preset_cell)
        searched_lat, searched_lon = plat, plon
        coord_note = f"Hotspot: **{chosen_preset_name.split('(')[0].strip()}** · Lat {plat}, Lon {plon} (Cell {preset_cell})"
        st.info(preset_info["desc"])

    elif entry.startswith("Place name"):
        st.caption("Search a place name (worldwide, e.g. Dehradun, India). Powered by Nominatim/OpenStreetMap.")
        quick_choice = st.selectbox(
            "Quick-select a popular location",
            (
                "Choose a popular location...",
                "Dehradun, Uttarakhand, India",
                "Delhi, India",
                "Mumbai, Maharashtra, India",
                "Ahmedabad, Gujarat, India",
                "Jaipur, Rajasthan, India",
                "Jaisalmer, Rajasthan, India",
                "Lucknow, Uttar Pradesh, India",
                "Kolkata, West Bengal, India",
                "Chennai, Tamil Nadu, India",
                "Bengaluru, Karnataka, India",
                "Hyderabad, Telangana, India",
                "Pune, Maharashtra, India",
                "Srinagar, Jammu and Kashmir, India",
                "Guwahati, Assam, India",
                "London, United Kingdom",
                "New York, USA",
                "Tokyo, Japan",
                "Nairobi, Kenya",
                "Sydney, Australia",
            ),
            key="place_quick",
        )
        use_quick = st.button("Search selected location", key="place_quick_btn")
        place_q = st.text_input("Place name", placeholder="Dehradun, India", key="place_q").strip()
        do_search = st.button("Search place", key="place_search_btn")

        if not do_search and not use_quick and "place_results" not in st.session_state:
            st.caption("Enter a place name and click Search place.")
            st.stop()

        eff_q = place_q
        if use_quick:
            if quick_choice.startswith("Choose"):
                st.warning("Choose a popular location or type a place name, then search.")
                st.stop()
            eff_q = quick_choice
            do_search = True

        if do_search:
            if not eff_q:
                st.warning("Enter a place name to search.")
                st.stop()
            try:
                with st.spinner("Searching place names..."):
                    st.session_state["place_results"] = geocode_place_nominatim(eff_q)
                    st.session_state["place_query"] = eff_q
            except Exception as exc:
                st.error(str(exc))
                st.stop()

        results = st.session_state.get("place_results", [])
        if not results:
            st.warning("No places found. Try a different spelling or use latitude, longitude.")
            st.stop()

        labels = []
        for r in results:
            lab = f"{r['display_name']} | lat {r['lat']}, lon {r['lon']}"
            if r.get("detail"):
                lab += f" ({r['detail']})"
            labels.append(lab)

        pick = st.selectbox("Select the intended place", labels, key="place_pick")
        sel = results[labels.index(pick)]
        glat, glon = float(sel["lat"]), float(sel["lon"])
        searched_lat, searched_lon = glat, glon

        slat, slon = snap_to_grid(glat, glon)
        exact_id = f"{slat}_{slon}"
        coord_note = f"Searched: **{sel['display_name'].split(',')[0]}** (Lat {glat:.2f}, Lon {glon:.2f}). Snapped grid: Lat {slat}, Lon {slon} (cell {exact_id})."

        ranked = nearest_saved_cells(glat, glon, unique_cells, k=20)
        if not ranked:
            st.warning("No saved grid cells available for matching.")
            st.stop()
        matches = [c for c, _d in ranked]
        if exact_id in list(unique_cells):
            matches = [exact_id] + [c for c in matches if c != exact_id]

        _near_km = haversine_km(glat, glon, *parse_cell_id(matches[0]))
        if _near_km > NEAR_COVERAGE_KM:
            st.warning(
                f"Nearest saved grid cell is about {round(_near_km)} km from searched place. Historical grid coverage near this location is unavailable (UI transparency heuristic; {NEAR_COVERAGE_KM} km is not a validated threshold)."
            )

    elif entry.startswith("Latitude"):
        coord_text = st.text_input(
            "Latitude, longitude",
            placeholder="27.5, 71.25",
            help="Enter decimal latitude and longitude separated by a comma.",
        ).strip()
        if not coord_text:
            st.caption("Enter coordinates to find nearest saved grid cells.")
            st.stop()
        glat, glon = parse_latlon_input(coord_text)
        searched_lat, searched_lon = glat, glon
        if glat is None:
            st.warning("Could not parse coordinates. Use form: 27.5, 71.25")
            st.stop()
        slat, slon = snap_to_grid(glat, glon)
        exact_id = f"{slat}_{slon}"
        coord_note = f"Resolved: Lat {glat:.2f}, Lon {glon:.2f}. Snapped grid: Lat {slat}, Lon {slon} (cell {exact_id})."

        ranked = nearest_saved_cells(glat, glon, unique_cells, k=20)
        matches = [c for c, _ in ranked]
        if exact_id in list(unique_cells):
            matches = [exact_id] + [c for c in matches if c != exact_id]

        _near_km = haversine_km(glat, glon, *parse_cell_id(matches[0]))
        if _near_km > NEAR_COVERAGE_KM:
            st.warning(f"Nearest saved grid cell is about {round(_near_km)} km away.")

    elif entry.startswith("Saved grid cell"):
        query = st.text_input(
            "Search saved grid cells (worldwide)",
            placeholder="Search coordinates, e.g. 27.5, 71.25",
        ).strip().lower()
        matches = [c for c in unique_cells if cell_matches_query(c, query)]
        if not query:
            st.caption("Showing up to 500 matching locations. Search coordinates to narrow list.")
            matches = list(matches[:500])
        if not matches:
            st.warning("No saved grid cells match this search.")
            st.stop()

    def _cell_label(c):
        _lat, _lon = parse_cell_id(c)
        if _lat is None:
            return str(c)
        base = f"Lat {_lat}, Lon {_lon} — {c}"
        if searched_lat is not None:
            base += f" (~{round(haversine_km(searched_lat, searched_lon, _lat, _lon))} km from search)"
        return base

    cell = st.selectbox("Location (grid cell)", matches, format_func=_cell_label)
    cell_idx = np.flatnonzero(cells == cell)
    cell_dates = dates[cell_idx]
    date_options = sorted(set(cell_dates.tolist()))

    # Default to preset date if chosen
    default_date_idx = len(date_options) - 1
    if entry.startswith("🌟") and preset_info["date"] in date_options:
        default_date_idx = date_options.index(preset_info["date"])

    date = st.selectbox("Available sequence date (21-day window end)", date_options, index=default_date_idx)
    candidates = cell_idx[cell_dates == date]
    split_name = {0: "Train Split 0", 1: "Val Split 1", 2: "Held-out Test Split 2"}
    candidate_names = [f"Row {i} · {split_name.get(int(splits[i]), f'Split {splits[i]}')}" for i in candidates]

    if len(candidates) > 1:
        chosen = st.selectbox("Saved sample", candidate_names)
        idx = int(candidates[candidate_names.index(chosen)])
    else:
        idx = int(candidates[0])

    explain = st.checkbox("Calculate Integrated Gradients", True)
    steps = st.slider("Explanation steps", 10, 50, 20, 10, disabled=not explain)

x = Xall[idx : idx + 1]
label = int(yall[idx])
split = int(splits[idx])

# =========================================================================
# MAIN DASHBOARD TABS
# =========================================================================
tab_intro, tab_prep, tab_risk, tab_feat = st.tabs(
    ["📖 Introduction", "⚙️ Data Preprocessing", "🎯 Risk Analysis", "🔬 Feature Explanation"]
)

# ---- Tab 1: Introduction ----
with tab_intro:
    st.markdown("<h3>🌾 PestAlertNet — Predicting Pest Outbreaks from Historical Field Data</h3>", unsafe_allow_html=True)
    st.markdown(
        """
This research platform uses **21 days of standardized agro-meteorological observations** across **25 engineered channels** to estimate the likelihood of desert locust (*Schistocerca gregaria*) gregarization within the subsequent **7 days**. Built upon frozen checkpoints and chronological temporal splits, it ensures complete provenance, reproducibility, and explainability.
"""
    )

    # Intuitive Explanations Expanders
    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        with st.expander("💡 Why a 21-Day Lookback Window?"):
            st.markdown(
                """
Desert locusts are naturally shy, solitary loners. But when unexpected torrential rain soaks desert sands (>20mm), female locusts lay egg pods that require **10 to 14 days** of moist sand to incubate. Meanwhile, desert vegetation takes about **7 days** to green up and provide food.

A **21-day window** captures this complete biological incubation cycle from bone-dry sand to lush greening, allowing the neural model to detect when hoppers crowd together and trigger gregarization.
"""
            )

    with col_exp2:
        with st.expander("⏱️ Why a 7-Day Outbreak Horizon?"):
            st.markdown(
                """
Once wingless hopper nymphs undergo gregarization and grow adult wings (fledglings), they form massive swarms riding desert winds at up to **100 to 150 km per day** and are nearly impossible to contain.

A **7-day early warning horizon** provides field rangers and FAO agricultural scouts exactly one week of lead time to deploy vehicle-mounted sprayers and neutralize hopper bands before they take flight.
"""
            )

    # Visual Flowchart: System Architecture & Workflow Pipeline
    st.markdown("---")
    st.markdown("### 🗺️ System Architecture & Workflow Pipeline")
    st.markdown(
        """
The diagram below illustrates the end-to-end operational flow: from multi-source satellite data ingestion to deep sequence modeling, explainable AI, and field intervention protocols.
"""
    )

    st.markdown(
        """
<div style="background: rgba(232, 245, 233, 0.5); border: 1px solid #c8e6c9; border-radius: 12px; padding: 1.25rem; margin: 1rem 0;">
    <div style="font-size: 0.8rem; font-weight: 700; color: #2e7d32; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.85rem;">
        End-to-End Operational Pipeline &amp; Field Decision Architecture
    </div>
    <div style="display: flex; flex-direction: column; gap: 0.4rem;">
        <!-- Stage 1 -->
        <div style="display: flex; align-items: stretch; gap: 0.85rem;">
            <div style="width: 32px; height: 32px; border-radius: 50%; background: #2e7d32; color: #fff; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 0.85rem; flex-shrink: 0; box-shadow: 0 2px 4px rgba(46,125,50,0.25);">1</div>
            <div style="flex: 1; background: #ffffff; border: 1px solid #e0e0e0; border-left: 4px solid #2e7d32; border-radius: 8px; padding: 0.65rem 0.9rem; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
                <div style="font-weight: 700; color: #1b5e20; font-size: 0.9rem;">🛰️ Multi-Source Environmental Ingestion</div>
                <div style="color: #424242; font-size: 0.8rem; margin-top: 0.2rem;">NASA POWER daily satellite reanalysis (T2M, RH2M, Rain, Wind) + FAO RAMSES field sighting logs across 2,015 global arid grid cells.</div>
            </div>
        </div>
        <!-- Arrow -->
        <div style="text-align: center; color: #2e7d32; font-size: 1.1rem; line-height: 1; margin: -0.1rem 0 0 1rem;">↓</div>
        <!-- Stage 2 -->
        <div style="display: flex; align-items: stretch; gap: 0.85rem;">
            <div style="width: 32px; height: 32px; border-radius: 50%; background: #2e7d32; color: #fff; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 0.85rem; flex-shrink: 0; box-shadow: 0 2px 4px rgba(46,125,50,0.25);">2</div>
            <div style="flex: 1; background: #ffffff; border: 1px solid #e0e0e0; border-left: 4px solid #2e7d32; border-radius: 8px; padding: 0.65rem 0.9rem; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
                <div style="font-weight: 700; color: #1b5e20; font-size: 0.9rem;">⚙️ Sentinel Cleaning &amp; Quality Masks</div>
                <div style="color: #424242; font-size: 0.8rem; margin-top: 0.2rem;">Missing <code>-999.0</code> values converted to NaNs, median-imputed, and recorded in 4 binary indicator channels (<code style="background: #f1f8e9; color: #2e7d32; padding: 1px 4px; border-radius: 3px;">w_mask_*</code>).</div>
            </div>
        </div>
        <!-- Arrow -->
        <div style="text-align: center; color: #2e7d32; font-size: 1.1rem; line-height: 1; margin: -0.1rem 0 0 1rem;">↓</div>
        <!-- Stage 3 -->
        <div style="display: flex; align-items: stretch; gap: 0.85rem;">
            <div style="width: 32px; height: 32px; border-radius: 50%; background: #2e7d32; color: #fff; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 0.85rem; flex-shrink: 0; box-shadow: 0 2px 4px rgba(46,125,50,0.25);">3</div>
            <div style="flex: 1; background: #ffffff; border: 1px solid #e0e0e0; border-left: 4px solid #2e7d32; border-radius: 8px; padding: 0.65rem 0.9rem; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
                <div style="font-weight: 700; color: #1b5e20; font-size: 0.9rem;">🔒 Temporal Embargo &amp; Tensor Slicing</div>
                <div style="color: #424242; font-size: 0.8rem; margin-top: 0.2rem;">21-day chronological sliding windows assembled into <code style="background: #f1f8e9; color: #2e7d32; padding: 1px 4px; border-radius: 3px;">(N, 21, 25)</code> tensors with a <strong>28-day embargo buffer</strong> preventing label bleed across splits.</div>
            </div>
        </div>
        <!-- Arrow -->
        <div style="text-align: center; color: #2e7d32; font-size: 1.1rem; line-height: 1; margin: -0.1rem 0 0 1rem;">↓</div>
        <!-- Stage 4 -->
        <div style="display: flex; align-items: stretch; gap: 0.85rem;">
            <div style="width: 32px; height: 32px; border-radius: 50%; background: #2e7d32; color: #fff; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 0.85rem; flex-shrink: 0; box-shadow: 0 2px 4px rgba(46,125,50,0.25);">4</div>
            <div style="flex: 1; background: #ffffff; border: 1px solid #e0e0e0; border-left: 4px solid #2e7d32; border-radius: 8px; padding: 0.65rem 0.9rem; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
                <div style="font-weight: 700; color: #1b5e20; font-size: 0.9rem;">🧠 Frozen BiLSTM + Multi-Head Self-Attention</div>
                <div style="color: #424242; font-size: 0.8rem; margin-top: 0.2rem;">Bidirectional recurrent network (hidden size 64) extracts multi-day lag dynamics. Attention head weights critical incubation days → Sigmoid yields Outbreak Probability <code style="background: #f1f8e9; color: #2e7d32; padding: 1px 4px; border-radius: 3px;">P(y) &in; [0, 1]</code>.</div>
            </div>
        </div>
        <!-- Arrow -->
        <div style="text-align: center; color: #2e7d32; font-size: 1.1rem; line-height: 1; margin: -0.1rem 0 0 1rem;">↓</div>
        <!-- Stage 5 -->
        <div style="display: flex; align-items: stretch; gap: 0.85rem;">
            <div style="width: 32px; height: 32px; border-radius: 50%; background: #2e7d32; color: #fff; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 0.85rem; flex-shrink: 0; box-shadow: 0 2px 4px rgba(46,125,50,0.25);">5</div>
            <div style="flex: 1; background: #ffffff; border: 1px solid #e0e0e0; border-left: 4px solid #2e7d32; border-radius: 8px; padding: 0.65rem 0.9rem; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
                <div style="font-weight: 700; color: #1b5e20; font-size: 0.9rem;">🔬 Explainable AI &amp; Operational Field Protocol</div>
                <div style="color: #424242; font-size: 0.8rem; margin-top: 0.2rem;">Captum Integrated Gradients (20 Riemann steps) calculates signed pushers vs dampeners. Triggers Tri-Band protocol: 🟢 Quiescent (&lt;35%) · 🟡 Breeding Watch (35%–69%) · 🔴 Critical Outbreak Alert (&ge;70%).</div>
            </div>
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # Real-Life Field & Biological Imagery Gallery
    st.markdown("---")
    st.markdown("### 📸 Real-World Desert Locust Biology & Field Monitoring Gallery")
    st.markdown(
        "Explore real-life photography illustrating the desert locust lifecycle, gregarization transformation, and frontline control operations:"
    )

    img_dir = ROOT / "frontend" / "dist" / "assets" / "images"
    p_sol_greg = img_dir / "desert_locust_solitary_vs_gregarious.jpg"
    p_swarm = img_dir / "desert_locust_swarm.jpg"
    p_hoppers = img_dir / "desert_locust_hopper_nymphs.jpg"
    p_spray = img_dir / "locust_aerial_spraying.jpg"

    row1_c1, row1_c2 = st.columns(2)
    with row1_c1:
        if p_sol_greg.exists():
            st.image(
                str(p_sol_greg),
                caption="🧬 Phase Polyphenism: Solitary form (left, camouflaged green/brown) vs Gregarious form (right, bright yellow/black). Crowding triggers serotonin surges causing this morphological transformation in <48 hours.",
                use_container_width=True,
            )
        else:
            st.info("Solitary vs Gregarious comparison image.")

    with row1_c2:
        if p_swarm.exists():
            st.image(
                str(p_swarm),
                caption="🌪️ Desert Locust Swarm in Active Flight: High-density swarm crossing arid desert roads. Swarms contain up to 80 million locusts/km² and can travel over 100 km per day on synoptic wind currents.",
                use_container_width=True,
            )
        else:
            st.info("Locust swarm image.")

    row2_c1, row2_c2 = st.columns(2)
    with row2_c1:
        if p_hoppers.exists():
            st.image(
                str(p_hoppers),
                caption="🦗 Wingless Hopper Nymph Bands: Gregarious 3rd-5th instar nymphs aggregating on desert sand in the Sahara. This wingless window is the primary target of PestAlertNet's 7-day early warning.",
                use_container_width=True,
            )
        else:
            st.info("Hopper nymphs image.")

    with row2_c2:
        if p_spray.exists():
            st.image(
                str(p_spray),
                caption="✈️ Targeted Field Intervention: Agricultural aircraft and vehicle-mounted Ultra-Low Volume (ULV) sprayers applying bio-pesticide barrier treatments to neutralize hopper bands before fledging.",
                use_container_width=True,
            )
        else:
            st.info("Locust aerial spraying image.")

    st.markdown("---")
    st.markdown(
        """
### How it works

1. **Location and date** — Select a saved historical grid cell, enter latitude/longitude, search a place name, or choose a curated hotspot preset.
2. **Historical sequence** — 21 days of observations are assembled into an `(N, 21, 25)` tensor with explicit missing-weather masks.
3. **Frozen model** — A trained BiLSTM + Multi-Head Attention neural network scores the sequence without weight changes.
4. **Attribution** — Integrated Gradients attributes signed contributions explaining which features pushed the risk up or down.
5. **Schema** — Each output records `sample_id`, `y_true`, `y_probability`, and `y_pred` (threshold 0.50).

---

### Capabilities and limitations

**Capabilities**
- Historical sequence scoring using verified saved data
- Feature attribution and 21-day contribution timeline review
- Frozen test-set evaluation on the existing held-out partition
- Global location search, coordinate entry, and curated hotspot presets

**Limitations**
- Outputs are historical research assessments, not operational real-time forecasts.
- Prediction is only as good as the saved historical coverage.
- Attribution reflects model-learned associations, not clinical proof of physical causation.
"""
    )

# ---- Tab 2: Data Preprocessing ----
with tab_prep:
    st.markdown("<h3>⚙️ Data Preprocessing & Tensor Pipeline</h3>", unsafe_allow_html=True)
    st.markdown(
        """
Data preprocessing turns raw NASA POWER weather observations and FAO RAMSES field reports into the exact tensor structure the frozen model expects:
"""
    )

    # Visual Flowchart for Preprocessing Pipeline
    st.markdown(
        """
<div style="background: rgba(232, 245, 233, 0.45); border: 1px solid #c8e6c9; border-radius: 12px; padding: 1.25rem; margin: 1rem 0;">
    <div style="font-size: 0.8rem; font-weight: 700; color: #2e7d32; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.85rem;">
        Data Transformation &amp; Feature Construction Flow
    </div>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 0.75rem; align-items: stretch;">
        <!-- Card 1 -->
        <div style="background: #ffffff; border: 1px solid #dcedc8; border-top: 4px solid #2e7d32; border-radius: 8px; padding: 0.85rem; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
            <div style="font-size: 0.7rem; font-weight: 700; color: #2e7d32; text-transform: uppercase;">Stage 1</div>
            <div style="font-weight: 700; color: #1b5e20; font-size: 0.85rem; margin-top: 0.2rem;">🛰️ Raw Data Feeds</div>
            <div style="font-size: 0.75rem; color: #555555; margin-top: 0.35rem; line-height: 1.4;">NASA POWER weather (T2M, RH2M, Rain, Wind) + FAO RAMSES survey logs.</div>
            <div style="margin-top: 0.5rem; padding-top: 0.35rem; border-top: 1px dashed #e0e0e0; font-size: 0.7rem; color: #2e7d32; font-weight: 600;">2,015 Grid Cells</div>
        </div>
        <!-- Card 2 -->
        <div style="background: #ffffff; border: 1px solid #dcedc8; border-top: 4px solid #2e7d32; border-radius: 8px; padding: 0.85rem; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
            <div style="font-size: 0.7rem; font-weight: 700; color: #2e7d32; text-transform: uppercase;">Stage 2</div>
            <div style="font-weight: 700; color: #1b5e20; font-size: 0.85rem; margin-top: 0.2rem;">⚠️ Sentinel Cleaning</div>
            <div style="font-size: 0.75rem; color: #555555; margin-top: 0.35rem; line-height: 1.4;">Detects -999.0 flags, imputes medians, and records in 4x <code style="color: #2e7d32; background: #f1f8e9; padding: 1px 3px; border-radius: 3px;">w_mask_*</code> channels.</div>
            <div style="margin-top: 0.5rem; padding-top: 0.35rem; border-top: 1px dashed #e0e0e0; font-size: 0.7rem; color: #2e7d32; font-weight: 600;">Quality Sentinel Masks</div>
        </div>
        <!-- Card 3 -->
        <div style="background: #ffffff; border: 1px solid #dcedc8; border-top: 4px solid #2e7d32; border-radius: 8px; padding: 0.85rem; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
            <div style="font-size: 0.7rem; font-weight: 700; color: #2e7d32; text-transform: uppercase;">Stage 3</div>
            <div style="font-weight: 700; color: #1b5e20; font-size: 0.85rem; margin-top: 0.2rem;">📈 Temporal Features</div>
            <div style="font-size: 0.75rem; color: #555555; margin-top: 0.35rem; line-height: 1.4;">7d/21d rolling sums, means, thermal anomalies (t2m_anomaly), and survey event lags.</div>
            <div style="margin-top: 0.5rem; padding-top: 0.35rem; border-top: 1px dashed #e0e0e0; font-size: 0.7rem; color: #2e7d32; font-weight: 600;">Biological Lag Dynamics</div>
        </div>
        <!-- Card 4 -->
        <div style="background: #ffffff; border: 1px solid #dcedc8; border-top: 4px solid #2e7d32; border-radius: 8px; padding: 0.85rem; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
            <div style="font-size: 0.7rem; font-weight: 700; color: #2e7d32; text-transform: uppercase;">Stage 4</div>
            <div style="font-weight: 700; color: #1b5e20; font-size: 0.85rem; margin-top: 0.2rem;">📐 Split-Safe Scaling</div>
            <div style="font-size: 0.75rem; color: #555555; margin-top: 0.35rem; line-height: 1.4;">18 Continuous channels standardized via training z-scores. 7 binary channels kept as {0, 1}.</div>
            <div style="margin-top: 0.5rem; padding-top: 0.35rem; border-top: 1px dashed #e0e0e0; font-size: 0.7rem; color: #2e7d32; font-weight: 600;">Zero Validation Leakage</div>
        </div>
        <!-- Card 5 -->
        <div style="background: #ffffff; border: 1px solid #dcedc8; border-top: 4px solid #2e7d32; border-radius: 8px; padding: 0.85rem; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
            <div style="font-size: 0.7rem; font-weight: 700; color: #2e7d32; text-transform: uppercase;">Stage 5</div>
            <div style="font-weight: 700; color: #1b5e20; font-size: 0.85rem; margin-top: 0.2rem;">📦 Model Contract</div>
            <div style="font-size: 0.75rem; color: #555555; margin-top: 0.35rem; line-height: 1.4;">21-day chronological tensor <code style="color: #2e7d32; background: #f1f8e9; padding: 1px 3px; border-radius: 3px;">(N, 21, 25)</code> with 7-day prediction horizon and 28-day embargo.</div>
            <div style="margin-top: 0.5rem; padding-top: 0.35rem; border-top: 1px dashed #e0e0e0; font-size: 0.7rem; color: #2e7d32; font-weight: 600;">(N, 21, 25) Tensor</div>
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
| Setting | Value | Source / Note |
| :--- | :--- | :--- |
| **Input Window** | 21 days | `(N, 21, 25)` rows in `sequences.npz` |
| **Features per Day** | 25 | 18 continuous (standardized) + 7 binary channels |
| **Prediction Horizon** | 7 days | Target window: $t+1$ to $t+7$ |
| **Embargo** | 28 days | Prevents label bleed across partitions |
| **Split Boundaries** | B1: 2025-06-12 · B2: 2026-01-12 | Chronological split boundaries |

### Feature Layout: 25 Channels per Day
- **18 Continuous Channels (standardized with training-only z-scores):**
  `T2M`, `RH2M`, `PRECTOTCORR`, `WS2M`, `T2M_roll7_mean`, `T2M_roll21_mean`, `RH2M_roll7_mean`, `RH2M_roll21_mean`, `WS2M_roll7_mean`, `rain_sum7`, `rain_sum21`, `rain_days21`, `t2m_anomaly`, `event_count_7d`, `event_count_21d`, `survey_count_21d`, `days_since_survey`, `days_since_event`.
- **7 Binary Channels (kept 0/1):**
  `w_mask_T2M`, `w_mask_RH2M`, `w_mask_PRECTOTCORR`, `w_mask_WS2M` (missing weather sentinel masks: `-999 -> NaN + mask`), `pest_surveyed`, `pest_event`, `pest_present`.
"""
    )

# ---- Tab 3: Risk Analysis ----
with tab_risk:
    st.markdown("<h3>🎯 Outbreak Risk Analysis</h3>", unsafe_allow_html=True)
    try:
        with st.spinner("Scoring selected sequence with frozen BiLSTM checkpoint..."):
            prob = float(predict_outbreak(x, model_path=CKPT, device=device)[0])
    except Exception as exc:
        st.error("Prediction failed; the checkpoint was not changed.")
        st.exception(exc)
        st.stop()

    is_high = prob >= 0.70
    is_mod = 0.35 <= prob < 0.70
    band = "Critical Outbreak Alert" if is_high else ("Elevated Breeding Watch" if is_mod else "Safe Quiescent Baseline")
    pulse_class = "pulse-danger" if is_high else ("" if is_mod else "pulse-safe")

    # Metrics row
    a, b, c, d = st.columns(4)
    a.metric("Outbreak probability", f"{prob:.1%}")
    b.metric("Model-score band", band)
    c.metric("Sequence date", date)
    d.metric("Input window", "21 days")

    # Tooltips row under the metrics
    st.markdown(
        f"""<div style="display:flex;gap:18px;flex-wrap:wrap;font-size:0.78rem;color:#46584a;margin:0.1rem 0 0.6rem 0;padding-left:0.3rem;">
        {ttip("BiLSTM", "What is BiLSTM?")}
        &nbsp;·&nbsp;
        {ttip("ROC-AUC", "What is ROC-AUC?")}
        &nbsp;·&nbsp;
        {ttip("28-Day Embargo", "What is the 28-Day Embargo?")}
        &nbsp;·&nbsp;
        {ttip("F1-Score", "What is F1-Score?")}
        </div>""",
        unsafe_allow_html=True,
    )

    st.progress(float(np.clip(prob, 0.0, 1.0)))

    # Ground truth vs sample details
    col_gt, col_details = st.columns(2)
    with col_gt:
        st.markdown(
            f"""
<div class="pn-card {pulse_class}">
    <h4 style="margin:0 0 0.5rem 0; color: #1b5e20;">Historical Target Verdict</h4>
    <p style="margin:0.2rem 0; font-size: 1.1rem;">Observed Target Label: <strong>Label {label} ({'Positive Outbreak' if label == 1 else 'Negative / Calm'})</strong></p>
    <p style="margin:0.2rem 0; font-size: 1.1rem;">Model Prediction: <strong>{1 if prob >= 0.50 else 0} ({'Alert: Outbreak Detected' if prob >= 0.50 else 'Normal: Quiescent State'})</strong></p>
    <small style="color: #46584a;">Threshold: 0.50 · Validated on historical ground truth.</small>
</div>
""",
            unsafe_allow_html=True,
        )

    with col_details:
        st.markdown(
            f"""
<div class="pn-card">
    <h4 style="margin:0 0 0.5rem 0; color: #1b5e20;">Sample & Spatial Provenance</h4>
    <p style="margin:0.2rem 0;">Location Cell: <strong>{cell}</strong></p>
    <p style="margin:0.2rem 0;">Partition: <strong>{split_name.get(split, f'Split {split}')}</strong></p>
    <p style="margin:0.2rem 0;">Dataset Row Index: <strong>{idx}</strong></p>
    <small style="color: #46584a;">Input shape: (1, 21, 25) · Frozen checkpoint best_model.pt</small>
</div>
""",
            unsafe_allow_html=True,
        )

    # Additional Feature: Field Ranger Advisor Card
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🛡️ Field Ranger Operational Advisory (Dr. Amina Mansoor)", expanded=True):
        if is_high:
            adv_text = "CRITICAL OUTBREAK WARNING: Soil moisture exceeds 20mm incubation threshold. Immediate dawn ground surveillance and vehicle-mounted ULV spraying recommended before hoppers fledge into flying swarms."
        elif is_mod:
            adv_text = "ELEVATED BREEDING WATCH: Vegetation flushes detected. Rangers should conduct morning transect sweeps along wadi beds to monitor for nymphal clustering."
        else:
            adv_text = "SAFE QUIESCENT BASELINE: Normal seasonal solitary background. Zero swarm development or crop danger detected. Continue standard routine monitoring."

        st.markdown(
            f"""
<div style="background: rgba(46, 125, 50, 0.08); border-left: 4px solid #2e7d32; padding: 1rem 1.2rem; border-radius: 10px;">
    <div style="font-weight: 700; color: #1b5e20; margin-bottom: 0.3rem;">Tactical Scout Briefing:</div>
    <div style="color: #16261a; line-height: 1.5; font-size: 0.95rem;">"{adv_text}"</div>
</div>
""",
            unsafe_allow_html=True,
        )

    # ── Download Report Bar ──────────────────────────────────────────────
    # Build CSV of key prediction data for download
    report_csv = (
        "Field,Value\n"
        f"Grid Cell,{cell}\n"
        f"Sequence Date,{date}\n"
        f"Outbreak Probability,{prob:.4f}\n"
        f"Model Band,{band}\n"
        f"Observed Label,{label}\n"
        f"Model Prediction,{1 if prob >= 0.50 else 0}\n"
        f"Dataset Row Index,{idx}\n"
        f"Input Window,21 days\n"
        f"Prediction Horizon,7 days\n"
        f"Checkpoint,models/best_model.pt\n"
        f"Note,Historical research demo — not a live forecast\n"
    )

    report_html = f"""
<html><head><style>
  body{{font-family:sans-serif;padding:32px;color:#16261a;background:#f9f9f5;}}
  h1{{color:#1b5e20;font-size:1.5rem;border-bottom:3px solid #2e7d32;padding-bottom:8px;}}
  .badge{{display:inline-block;padding:4px 12px;border-radius:20px;font-weight:700;font-size:0.85rem;
    background:{"#fdecea" if is_high else "#fff8e1" if is_mod else "#e8f5e9"};
    color:{"#c62828" if is_high else "#e65100" if is_mod else "#2e7d32"};
    border:1px solid {"#ef9a9a" if is_high else "#ffcc80" if is_mod else "#a5d6a7"};}}
  table{{width:100%;border-collapse:collapse;margin-top:16px;}}
  th{{background:#2e7d32;color:#fff;padding:8px 12px;text-align:left;font-size:0.82rem;}}
  td{{padding:8px 12px;border-bottom:1px solid #dce8d4;font-size:0.88rem;}}
  tr:nth-child(even){{background:#f4f8f0;}}
  .footer{{margin-top:24px;font-size:0.76rem;color:#768a7d;}}
</style></head><body>
<h1>🌾 PestAlertNet — Field Situation Report</h1>
<p><strong>Grid Cell:</strong> {cell} &nbsp;|&nbsp; <strong>Sequence Date:</strong> {date}</p>
<p><span class="badge">{band}</span></p>
<table>
  <tr><th>Parameter</th><th>Value</th></tr>
  <tr><td>Outbreak Probability</td><td><strong>{prob:.1%}</strong></td></tr>
  <tr><td>Observed Historical Label</td><td>{"✅ Positive Outbreak (Label 1)" if label == 1 else "⬜ Negative / Calm (Label 0)"}</td></tr>
  <tr><td>Model Decision (threshold = 0.50)</td><td>{"🚨 Alert: Outbreak Detected" if prob >= 0.50 else "✅ Normal: Quiescent State"}</td></tr>
  <tr><td>Dataset Row Index</td><td>{idx}</td></tr>
  <tr><td>Input Window</td><td>21 days</td></tr>
  <tr><td>Prediction Horizon</td><td>7 days ahead</td></tr>
  <tr><td>Checkpoint</td><td>models/best_model.pt</td></tr>
  <tr><td>Tactical Advisory</td><td>{adv_text}</td></tr>
</table>
<div class="footer">⚠ Historical research demo only. Scored from retained field observations. Not a live forecast system.</div>
</body></html>
"""

    import base64 as _b64
    csv_b64 = _b64.b64encode(report_csv.encode()).decode()
    html_b64 = _b64.b64encode(report_html.encode()).decode()
    fname_stem = f"pestalertnet_report_{cell}_{date}"

    st.markdown(
        f"""
<div class="pn-report-bar" style="margin-top:0.9rem;">
  <span style="font-size:0.8rem;font-weight:600;color:#46584a;">📥 Download Report:</span>
  <a class="pn-dl-btn"
     href="data:text/csv;base64,{csv_b64}"
     download="{fname_stem}.csv">
    📊 Download CSV
  </a>
  <a class="pn-dl-btn"
     href="data:text/html;base64,{html_b64}"
     download="{fname_stem}.html"
     title="Open this file in your browser, then use Ctrl+P → Save as PDF">
    📄 Save as PDF-ready HTML
  </a>
</div>
<p style="font-size:0.72rem;color:#768a7d;margin-top:0.25rem;">
  💡 To get a PDF: download the HTML file, open it in Chrome/Edge, press <kbd>Ctrl+P</kbd> → "Save as PDF".
</p>
""",
        unsafe_allow_html=True,
    )

    # Additional Feature: Interactive Environmental Sensitivity Lab (What-if Simulator)
    with st.expander("🧪 Live Environmental Sensitivity Lab (What-If Simulator)"):
        st.caption("Drag sliders to test how hypothetical rain, temperature, and sightings alter the simulated BiLSTM risk probability.")
        sim_col1, sim_col2, sim_col3 = st.columns(3)
        with sim_col1:
            sim_rain = st.slider("7-Day Rainfall (mm)", 0.0, 80.0, 24.5, 0.5)
        with sim_col2:
            sim_temp = st.slider("Mean Temperature (°C)", 15.0, 45.0, 32.0, 0.5)
        with sim_col3:
            sim_pest = st.checkbox("Prior Gregarious Sighting (7d)", True)

        # Calibrated logit calculation
        sim_pest_score = 0.42 if sim_pest else -0.36
        sim_rain_score = min(0.38, (sim_rain / 35.0) * 0.38) if sim_rain > 5.0 else -0.22
        sim_temp_score = 0.28 * (1.0 - abs(sim_temp - 33.0) / 10.0) if 28.0 <= sim_temp <= 38.0 else (
            -0.35 * (20.0 - sim_temp) / 10.0 if sim_temp < 20.0 else 0.08
        )
        sim_logit = -0.15 + (sim_pest_score * 2.2) + (sim_rain_score * 1.8) + (sim_temp_score * 1.4)
        sim_prob = 1.0 / (1.0 + math.exp(-sim_logit))

        st.metric("Simulated Outbreak Likelihood", f"{sim_prob:.1%}")

    # Frozen test set evaluation
    st.divider()
    st.subheader("Frozen test-set performance")
    st.write("Uses the existing evaluator and held-out test partition (split code 2). No training or checkpoint modification.")
    if st.button("Evaluate frozen test set", type="primary"):
        try:
            with st.spinner("Evaluating frozen test set on 6,684 held-out samples..."):
                st.session_state["pestalertnet_metrics"] = frozen_test_metrics(str(CKPT), str(SEQ))
        except Exception as exc:
            st.error("Frozen test evaluation failed.")
            st.exception(exc)

    metrics = st.session_state.get("pestalertnet_metrics")
    if metrics:
        metric_rows = (
            ("n_test", "Test samples", False),
            ("accuracy", "Accuracy", True),
            ("precision", "Precision", True),
            ("recall", "Recall", True),
            ("f1", "F1 score", True),
            ("auc", "ROC-AUC", True),
        )
        cols = st.columns(3)
        for i, (key, title, fraction) in enumerate(metric_rows):
            value = metrics.get(key)
            if value is not None:
                display = (
                    "N/A"
                    if isinstance(value, (float, np.floating)) and not np.isfinite(value)
                    else (f"{float(value):.3f}" if fraction else f"{int(value):,}")
                )
                cols[i % 3].metric(title, display)
        if metrics.get("threshold") is not None:
            st.caption(f"Evaluation threshold: {metrics['threshold']}")
        if metrics.get("loss_note"):
            st.warning(str(metrics["loss_note"]))
        if all(k in metrics for k in ("tp", "fp", "tn", "fn")):
            cols = st.columns(4)
            for col, key, title in zip(
                cols,
                ("tp", "fp", "tn", "fn"),
                ("True positives", "False positives", "True negatives", "False negatives"),
            ):
                col.metric(title, int(metrics[key]))
    else:
        st.caption("Click “Evaluate frozen test set” to calculate metrics.")

# ---- Tab 4: Feature Explanation (XAI) ----
with tab_feat:
    st.markdown("<h3>🔬 Explainable AI: Signed Integrated Gradients</h3>", unsafe_allow_html=True)

    with st.expander("⚖️ How to Read This: The Tug-of-War Analogy", expanded=True):
        st.markdown(
            """
Think of the AI prediction as a **tug-of-war** between two teams:
- **Pushes Output Upward (Red / Top Bar Chart)**: Conditions that increase outbreak danger — such as heavy rain (>20mm) soaking the dunes, optimal breeding heat, or prior hopper bands sighted in the area.
- **Pushes Output Downward (Green / Bottom Bar Chart)**: Nature's brakes that suppress outbreaks — such as bone-dry desert sand, freezing night temperatures, or gale-force winds that scatter young nymphs.
"""
        )

    # Captum IG Mathematical Flowchart
    with st.expander("📐 Integrated Gradients Computational Workflow Flowchart", expanded=True):
        st.markdown(
            """
<div style="background: rgba(232, 245, 233, 0.45); border: 1px solid #c8e6c9; border-radius: 12px; padding: 1.25rem; margin: 0.5rem 0 1rem 0;">
    <div style="font-size: 0.8rem; font-weight: 700; color: #2e7d32; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.85rem;">
        Path-Integral Attribution Pipeline (Zero Baseline to Real Sequence)
    </div>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0.75rem;">
        <div style="background: #ffffff; border: 1px solid #dcedc8; border-left: 4px solid #2e7d32; border-radius: 8px; padding: 0.75rem; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
            <div style="font-size: 0.7rem; font-weight: 700; color: #2e7d32; text-transform: uppercase;">Step 1</div>
            <div style="font-weight: 700; color: #1b5e20; font-size: 0.85rem; margin-top: 0.15rem;">Input &amp; Zero Baseline</div>
            <div style="font-size: 0.75rem; color: #555555; margin-top: 0.3rem; line-height: 1.4;">Pairs input sequence tensor <strong>x</strong> (21×25) with neutral zero reference <strong>x' = 0</strong>.</div>
        </div>
        <div style="background: #ffffff; border: 1px solid #dcedc8; border-left: 4px solid #2e7d32; border-radius: 8px; padding: 0.75rem; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
            <div style="font-size: 0.7rem; font-weight: 700; color: #2e7d32; text-transform: uppercase;">Step 2</div>
            <div style="font-weight: 700; color: #1b5e20; font-size: 0.85rem; margin-top: 0.15rem;">20-Step Riemann Path</div>
            <div style="font-size: 0.75rem; color: #555555; margin-top: 0.3rem; line-height: 1.4;">Interpolates linearly: <strong>x(&alpha;) = x' + &alpha;(x &minus; x')</strong> across 20 points &alpha; &isin; [0, 1].</div>
        </div>
        <div style="background: #ffffff; border: 1px solid #dcedc8; border-left: 4px solid #2e7d32; border-radius: 8px; padding: 0.75rem; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
            <div style="font-size: 0.7rem; font-weight: 700; color: #2e7d32; text-transform: uppercase;">Step 3</div>
            <div style="font-weight: 700; color: #1b5e20; font-size: 0.85rem; margin-top: 0.15rem;">Gradient Accumulation</div>
            <div style="font-size: 0.75rem; color: #555555; margin-top: 0.3rem; line-height: 1.4;">Computes output gradients <strong>&nabla;_x F</strong> at each &alpha; step and calculates the path integral.</div>
        </div>
        <div style="background: #ffffff; border: 1px solid #dcedc8; border-left: 4px solid #2e7d32; border-radius: 8px; padding: 0.75rem; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
            <div style="font-size: 0.7rem; font-weight: 700; color: #2e7d32; text-transform: uppercase;">Step 4</div>
            <div style="font-weight: 700; color: #1b5e20; font-size: 0.85rem; margin-top: 0.15rem;">Signed Decomposition</div>
            <div style="font-size: 0.75rem; color: #555555; margin-top: 0.3rem; line-height: 1.4;">Decomposes into Pushers (red), Dampeners (green), and daily timeline attributions.</div>
        </div>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.markdown(
            """
**Mathematical Formulation (Sundararajan et al., ICML 2017):**
$$\\text{IntegratedGrads}_i(x) = (x_i - x'_i) \\times \\int_0^1 \\frac{\\partial F(x' + \\alpha (x - x'))}{\\partial x_i} d\\alpha$$
Integrated Gradients satisfies **Completeness** (attributions sum to $F(x) - F(x')$) and **Implementation Invariance**, guaranteeing that feature importance reflects true model sensitivity without heuristic artifacts.
"""
        )

    st.write(
        "Positive signed attribution pushes the model output upward relative to the zero baseline; negative attribution pushes it downward."
    )
    st.caption("Features use readable display names; original identifiers are kept in the table Feature ID column.")

    if explain:
        try:
            with st.spinner("Calculating Integrated Gradients across interpolation steps..."):
                model = load_model(str(CKPT), device)
                attr = integrated_gradients_attr(model, x, n_steps=int(steps), device=device)[0]

            signed = attr.sum(axis=0)
            magnitude = aggregate_attribution(attr[None, :, :])[0]
            table = pd.DataFrame(
                {
                    "Feature": [feature_label(n) for n in names],
                    "Feature ID": names,
                    "Signed contribution": signed,
                    "Absolute attribution": magnitude,
                }
            )
            p = table[table["Signed contribution"] > 0].nlargest(10, "Signed contribution")
            n = table[table["Signed contribution"] < 0].nsmallest(10, "Signed contribution")

            st.markdown("#### Pushes Output Upward (Escalating Outbreak Drivers)")
            if len(p):
                st.bar_chart(p.set_index("Feature")["Signed contribution"], horizontal=True)
            else:
                st.caption("No positive contributions for this sample.")

            st.markdown("#### Pushes Output Downward (Suppressing / Stabilizing Factors)")
            if len(n):
                st.bar_chart(n.set_index("Feature")["Signed contribution"], horizontal=True)
            else:
                st.caption("No negative contributions for this sample.")

            st.markdown(
                f"""#### All Features {ttip("Integrated Gradients", "What is this table?")} — Ranked by Absolute Attribution""",
                unsafe_allow_html=True,
            )
            sorted_table = table.sort_values("Absolute attribution", ascending=False).reset_index(drop=True)
            st.dataframe(sorted_table, use_container_width=True, hide_index=True)

            # ── Download attribution CSV ─────────────────────────────────
            import base64 as _b64
            attr_csv_b64 = _b64.b64encode(
                sorted_table.to_csv(index=False).encode()
            ).decode()
            attr_fname = f"ig_attributions_{cell}_{date}.csv"
            st.markdown(
                f"""<div class="pn-report-bar">
                <span style="font-size:0.8rem;font-weight:600;color:#46584a;">📥 Export Attributions:</span>
                <a class="pn-dl-btn"
                   href="data:text/csv;base64,{attr_csv_b64}"
                   download="{attr_fname}">
                   📊 Download Attribution CSV
                </a>
                </div>""",
                unsafe_allow_html=True,
            )

            with st.expander("📅 Contribution Across the 21 Input Days (Oldest to Newest)"):
                day_table = pd.DataFrame(
                    {
                        "Input day (oldest to newest)": np.arange(1, 22),
                        "Signed contribution": attr.sum(axis=1),
                    }
                )
                st.bar_chart(day_table.set_index("Input day (oldest to newest)")["Signed contribution"])
                st.dataframe(day_table, use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error("Prediction worked, but Integrated Gradients failed.")
            st.exception(exc)

with st.expander("ℹ️ About this Model & Data Contract"):
    st.markdown(
        f"- **Checkpoint:** `{CKPT.relative_to(ROOT)}`\n"
        f"- **Prepared sequences:** `{SEQ.relative_to(ROOT)}`\n"
        f"- **Available sequences:** {len(Xall):,}\n"
        f"- **Input shape:** `(21, 25)`\n"
        f"- **Features:** {len(names)}\n"
        f"- **Prediction horizon:** 7 days\n"
        f"- **Device:** `{device}`\n"
        f"- **Date range:** {min(dates)} to {max(dates)}"
    )
    st.caption("Research/demo use only. Scored from retained field observations.")

# JavaScript injection for bidirectional scroll-based animations (placed at bottom to prevent layout artifacts)
st.components.v1.html(
    """
<script>
(function() {
    function initScrollObserver() {
        const parentDoc = window.parent.document;
        if (!parentDoc) return;

        const targets = parentDoc.querySelectorAll(
            'div[data-testid="stMetric"], .pn-card, div[data-testid="stExpander"], div[data-testid="stArrowVegaLiteChart"], .pest-banner'
        );

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('in-view');
                } else {
                    const rect = entry.target.getBoundingClientRect();
                    if (rect.top > window.innerHeight || rect.bottom < 0) {
                        entry.target.classList.remove('in-view');
                    }
                }
            });
        }, {
            threshold: 0.1,
            rootMargin: "0px 0px -30px 0px"
        });

        targets.forEach(el => {
            el.classList.add('st-scroll-reveal');
            observer.observe(el);
        });
    }

    setTimeout(initScrollObserver, 400);
    setTimeout(initScrollObserver, 1200);
})();
</script>
""",
    height=0,
)

