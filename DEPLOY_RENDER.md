# 🚀 How to Deploy PestAlertNet on Render (For Phone & Public Access)

This project is pre-configured with a **`render.yaml` Blueprint** and a **lightweight CPU dependency file (`requirements-render.txt`)** tailored for Render's free tier (512MB RAM).

---

## 📱 Quick 3-Click Deployment via Render Dashboard

1. **Log in to Render**:
   - Go to [render.com](https://render.com) and sign in with your GitHub account (`Priyansu17`).

2. **Create New Web Service**:
   - Click the blue **"New +"** button in the top right → Select **"Web Service"**.
   - Under *Connect a repository*, choose `pestalertnet_v2` (or search for `pestalertnet_v2`).
   - Click **"Connect"**.

3. **Configure Settings**:
   Fill in these exact values:
   - **Name**: `pestalertnet` (or any name you prefer)
   - **Region**: Any (e.g., Oregon or Frankfurt)
   - **Branch**: `main`
   - **Language / Runtime**: `Python 3`
   - **Build Command**:
     ```bash
     pip install -r requirements-render.txt
     ```
   - **Start Command**:
     ```bash
     python serve_frontend.py
     ```
   - **Instance Type**: Select **Free** (0.5 CPU, 512 MB RAM)

4. **Click "Deploy Web Service"**:
   - Render will build and deploy the app in ~2 minutes.
   - Once the status turns green (**Live**), copy your public URL:
     `https://pestalertnet-xxxx.onrender.com`

---

## 📱 Opening on Your Phone
1. Open Chrome / Safari on your smartphone.
2. Navigate to your Render URL: `https://pestalertnet-xxxx.onrender.com`
3. The dashboard automatically adapts to your phone screen:
   - **Top tabs**: Swipe horizontally across the tab bar.
   - **Satellite Map**: Touch and pinch-to-zoom across the recession belt.
   - **Threshold Slider ($\tau$)**: Drag with your finger to toggle between Scouting and Spraying modes.
   - **Tactical Audio**: Tap the chime icon 🔔 to hear alerts when inspecting high-risk zones.

---

## 💡 Alternative: Deploying Streamlit on Render
If you want to deploy the Streamlit version instead of the Web Console:
- Change the **Start Command** to:
  ```bash
  streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
  ```
