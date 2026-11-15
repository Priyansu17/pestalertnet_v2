# 🌾 PestAlertNet — 2-Minute Live Showcase Walkthrough

> **Time Target**: Exactly 2 Minutes · **Primary Display**: `http://localhost:8080` (or phone) · **Secondary**: `http://localhost:8501`

---

## ⏱️ Step-by-Step Story Arc

```
[0:00 - 0:25] ── The Problem & Geospatial Reality
[0:25 - 0:50] ── Outbreak Detection & The Biological Incubation Window
[0:50 - 1:15] ── Transparent AI: Integrated Gradients (Why the model scored 65.2%)
[1:15 - 1:40] ── Operational Decision Support: Dynamic τ Slider & Side-by-Side Compare
[1:40 - 2:00] ── 1-Click Field Artifacts & The Multi-Device Edge
```

---

### Phase 1: The Problem & The Satellite View (0:00 – 0:25)
* **Screen**: Open `http://localhost:8080` (Risk Analysis tab).
* **Action**:
  1. Toggle the **Satellite map layer** (top-right layer control on the Leaflet map).
  2. Point to the arid recession belt spanning Mauritania to India.
* **Speaking Script**:
  > *"Desert locusts are humanity’s oldest agricultural plague. A single 1-square-kilometer swarm eats the same food in one day as 35,000 people. Our mission with **PestAlertNet** is to move from reactive emergency firefighting to a strict 7-day predictive early-warning system powered by NASA satellite agro-meteorology and FAO field records."*

---

### Phase 2: Detecting an Outbreak & 21-Day Incubation (0:25 – 0:50)
* **Screen**: Leaflet map pulsating red marker at Lat 27.5, Lon 71.25.
* **Action**:
  1. Click the red marker → Click **"Inspect Hotspot →"** (or select *Thar Desert Outbreak* from the preset dropdown).
  2. Listen for the **tactical audio chime 🔔** as the risk dial sweeps to **65.2%**.
  3. Scroll down to the **21-Day Time-Lapse Player** and click **▶ Play**.
* **Speaking Script**:
  > *"When we inspect the 2023 Thar Desert outbreak, our BiLSTM model instantly sounds a critical alert at 65.2% risk. But watch what happens over the 21-day timeline: locusts don't just appear out of nowhere. Notice the massive rainfall spike on Day 7 exceeding 20 millimeters. That moisture initiates egg-pod incubation, followed by vegetation flush on Day 14, culminating in gregarious hopper bands by Day 21."*

---

### Phase 3: Explainable AI — Opening the Black Box (0:50 – 1:15)
* **Screen**: Switch to **🔬 Feature Attribution (Captum)** tab (or scroll down to Integrated Gradients).
* **Action**:
  1. Point to the top positive attribution bars: **PRECTOTCORR** (Precipitation) and **RH2M** (Relative Humidity).
  2. Highlight the **Completeness Delta ($|\delta| < 0.05$)**.
* **Speaking Script**:
  > *"In high-stakes agriculture, a black-box probability is unacceptable. Using Captum Integrated Gradients, PestAlertNet attributes signed contributions to all 25 features. We can mathematically prove to field scouts that rainfall and humidity were the causal drivers pushing this prediction upward, verified by an axiomatic Completeness Delta of 0.003."*

---

### Phase 4: Operational Decision Support ($\tau$ Slider & Comparison) (1:15 – 1:40)
* **Screen**: Return to **🎯 Outbreak Risk Analysis** tab.
* **Action**:
  1. Drag the **Decision Threshold ($\tau$) Slider**:
     - Drag left to **0.30**: Watch the badge switch to **🔍 Preventive Scouting Mode**.
     - Drag right to **0.70**: Watch the badge switch to **✈️ Targeted Spraying Mode**.
  2. Scroll to **⚖️ Compare 2 Locations**: Select *Thar Desert* on Zone A vs *Solitary Arid Baseline* on Zone B.
* **Speaking Script**:
  > *"Different operations require different risk appetites. When scouting teams head out at dawn, we lower tau to 0.30 to maximize recall so no nymph clusters are missed. When deploying expensive aerial spray aircraft, commanders slide tau to 0.70 to eliminate false alarms. Furthermore, our side-by-side comparison engine immediately contrasts Thar Desert's saturated wadis against a freezing, bone-dry solitary baseline showing a 64.8% risk delta."*

---

### Phase 5: Field Artifacts & Conclusion (1:40 – 2:00)
* **Screen**: Click **📄 Save PDF Report** or camera icon **📷 Export PNG**.
* **Action**:
  1. Show the instant, single-page official FAO field report ready to sign.
  2. Point out that the entire dashboard is responsive on smartphones for rangers in desert 4x4 vehicles.
* **Speaking Script**:
  > *"Finally, rangers in the field can export an official 1-page situational report with one tap or download raw telemetry as CSV. PestAlertNet connects deep sequence learning directly to frontline tactical decisions. Thank you!"*

---

## 💡 Quick Q&A Defense Sheet for Judges

| Anticipated Question | 10-Second Winning Answer |
|---|---|
| **"Is this a live weather forecast?"** | *"No, this is an empirical retrospective validation system. It proves that given 21 days of standardized observations, the BiLSTM architecture reliably predicts gregarization 7 days in advance without temporal leakage (strict 28-day embargo)."* |
| **"Why BiLSTM over a Transformer?"** | *"A 21-day time series is relatively short. BiLSTMs achieve higher data efficiency with fewer parameters, preventing overfitting on rare historical outbreak events while maintaining sub-10ms inference latency."* |
| **"How is this used in the real world?"** | *"FAO Desert Locust Information Service (DLIS) in Rome and national locust control centres can ingest morning satellite feeds and deploy localized ground teams days before swarms take wing."* |
