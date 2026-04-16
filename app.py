"""
app.py — TorNet Tornado Alert Dashboard
Streamlit-based Real-Time Tornado Monitoring System.
"""

import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
import pandera as pa
from pandera.typing import Series
import httpx


# --- DATA VALIDATION SCHEMAS ---
PredictionSchema = pa.DataFrameSchema({
    "scan_id": pa.Column(str),
    "timestamp": pa.Column(str),
    "latitude": pa.Column(float, checks=pa.Check.in_range(-90, 90)),
    "longitude": pa.Column(float, checks=pa.Check.in_range(-180, 180)),
    "probability": pa.Column(float, checks=pa.Check.in_range(0, 1)),
    "tornado_detected": pa.Column(int, checks=pa.Check.isin([0, 1])),
    "sensor": pa.Column(str),
})


# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="TorNet — Tornado Alert Dashboard",
    page_icon="🌪️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- CUSTOM CSS INJECTION ---
def load_css(file_name):
    """Loads custom CSS file and injects it into streamlit."""
    if Path(file_name).exists():
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("css/style.css")


# --- GLOBAL CONSTANTS ---
PREDICTIONS_PATH = Path(os.getenv("PREDICTIONS_CSV", "/app/data/offline_data_fallback.csv")) 
API_URL = os.getenv("API_URL", "http://fastapi-service:80")
REFRESH_INTERVAL_SECONDS = 30

ALERT_COLORS = {
    "CRITICAL": "#ef4444",
    "HIGH":     "#fb923c",
    "MODERATE": "#eab308",
    "NONE":     "#3b82f6", # Blue for normal
}


# --- HIGH-PERFORMANCE CALCULATIONS ---
def haversine_vectorized(lat1: float, lon1: float, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Fast vectorized distance calculation in km using NumPy."""
    R = 6371.0
    lat1, lon1, lats, lons = map(np.radians, [lat1, lon1, lats, lons])
    dlat = lats - lat1
    dlon = lons - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lats) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


# --- DATA FETCHING & PROCESSING ---

@st.cache_data(ttl=REFRESH_INTERVAL_SECONDS)
def load_predictions_csv(path: str) -> pd.DataFrame:
    """Loads and validates the predictions CSV from local storage."""
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(p)
        df.columns = [c.lower().strip() for c in df.columns]
        return PredictionSchema.validate(df)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=600)
def fetch_api_inventory() -> list:
    """Fetches list of available dates from the inference API."""
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{API_URL}/api/v1/inventory")
            if resp.status_code == 200:
                return resp.json().get("dates", [])
    except Exception:
        pass
    return []

@st.cache_data(ttl=3600)
def fetch_api_forecast(target_date_iso: str) -> pd.DataFrame:
    """Fetches real-time model predictions for a specific date."""
    try:
        with httpx.Client(timeout=300.0) as client:
            resp = client.post(f"{API_URL}/api/v1/forecast", json={"date_": target_date_iso})
            if resp.status_code == 200:
                data = resp.json()
                predictions = data.get("predictions", [])
                if not predictions:
                    return pd.DataFrame()
                
                df = pd.DataFrame(predictions)
                # Adapter: Map API names to Schema names
                df["probability"] = df["tornado_probability"]
                df["latitude"] = df["lat"]
                df["longitude"] = df["lon"]
                df["tornado_detected"] = (df["probability"] > 0.5).astype(int)
                df["scan_id"] = [f"api_{i:04d}" for i in range(len(df))]
                
                return PredictionSchema.validate(df)
    except Exception as e:
        st.sidebar.warning(f"API Connection Error: {e}")
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def enrich_with_distance(df: pd.DataFrame, user_lat: float, user_lon: float) -> pd.DataFrame:
    """Calculates distances using high-speed vectorized NumPy operations."""
    if df.empty:
        return df
    
    df = df.copy()
    df["distance_km"] = haversine_vectorized(
        user_lat, user_lon, 
        df["latitude"].values.astype(float), 
        df["longitude"].values.astype(float)
    ).round(1)
    
    return df.sort_values("distance_km")


# --- GEOSPATIAL VISUALIZATION ---

def build_map(df: pd.DataFrame, user_lat: float, user_lon: float,
              threshold_km: float, show_all: bool) -> folium.Map:
    """
    Constructs an interactive Folium map.
    Preserves normal scans as discrete blue points as per user request.
    """
    m = folium.Map(
        location=[user_lat, user_lon],
        zoom_start=5,
        tiles=None,
    )

    # Dark-themed Tile Layer
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; '
             '<a href="https://carto.com/attributions">CARTO</a>',
        name="Dark Matter",
        max_zoom=19,
    ).add_to(m)

    # Dynamic Alert Radius
    folium.Circle(
        location=[user_lat, user_lon],
        radius=threshold_km * 1000, 
        color="#3b82f6",
        weight=1.5,
        fill=True,
        fill_color="#3b82f6",
        fill_opacity=0.05,
        tooltip=f"Alert Radius: {threshold_km:.0f} km",
    ).add_to(m)

    # Home/User Marker
    folium.Marker(
        location=[user_lat, user_lon],
        tooltip="📍 Your Location",
        popup=folium.Popup(
            f"<b>Your Location</b><br>Lat: {user_lat:.4f}<br>Lon: {user_lon:.4f}",
            max_width=200
        ),
        icon=folium.Icon(color="blue", icon="home", prefix="fa"),
    ).add_to(m)

    if not df.empty:
        # Separate detected tornadoes from background scans
        tornados = df[df["tornado_detected"] == 1]
        tornado_sensors = tornados["sensor"].unique()

        # Render background scans as discrete blue dots (Normal Scans)
        if show_all:
            normal_scans = df[
                (df["tornado_detected"] == 0) & 
                (~df["sensor"].isin(tornado_sensors))
            ]

            for _, row in normal_scans.iterrows():
                lat, lon = float(row["latitude"]), float(row["longitude"])
                ses = str(row.get("sensor", "NONE"))

                folium.CircleMarker(
                    location=[lat, lon],
                    radius=4,
                    color="#3b82f6",
                    weight=1,
                    fill=True,
                    fill_color="#3b82f6",
                    fill_opacity=0.4,
                    tooltip=f"ℹ️ Clear Scan | Sensor: {ses}"
                ).add_to(m)

        # Render Tornado Detections
        for _, row in tornados.iterrows():
            lat  = float(row["latitude"])
            lon  = float(row["longitude"])
            prob = float(row.get("probability", 0.5))
            dist = float(row.get("distance_km", 0))
            ses  = str(row.get("sensor", "NONE"))
            sid  = str(row.get("scan_id", "?"))

            # Determine icon color based on threat level
            icon_color = ("red" if ses == "CRITICAL" else
                         "orange" if ses == "HIGH" else 
                         "cadetblue" if ses == "MODERATE" else "blue")

            # Impact Halo
            folium.CircleMarker(
                location=[lat, lon],
                radius=12 + prob * 10,
                color=ALERT_COLORS.get(ses, "#ef4444"),
                weight=2,
                fill=True,
                fill_color=ALERT_COLORS.get(ses, "#ef4444"),
                fill_opacity=0.35,
            ).add_to(m)
            
            # Tornado Bolt Icon
            folium.Marker(
                location=[lat, lon],
                tooltip=f"🌪️ Tornado | {dist:.0f} km | p={prob:.2f}",
                popup=folium.Popup(
                    f"""<div style='font-family:sans-serif;font-size:13px'>
                    <b>🌪️ Tornado Detected</b><br>
                    <b>Scan ID:</b> {sid}<br>
                    <b>Probability:</b> {prob:.1%}<br>
                    <b>Sensor:</b> {ses}<br>
                    <b>Distance:</b> {dist:.1f} km<br>
                    <b>Coords:</b> {lat:.4f}, {lon:.4f}
                    </div>""",
                    max_width=250
                ),
                icon=folium.Icon(color=icon_color, icon="bolt", prefix="fa"),
            ).add_to(m)

    folium.LayerControl().add_to(m)
    return m


# --- INITIALIZATION LOGIC ---

# 1. Check API Status
api_online = False
api_dates = []
api_metadata = {}
try:
    with httpx.Client(timeout=3.0) as client:
        resp = client.get(f"{API_URL}/health")
        if resp.status_code == 200:
            api_online = True
            api_metadata = resp.json()
    if api_online:
        api_dates = fetch_api_inventory()
except Exception:
    api_online = False

# 2. Load Fallback CSV Data
df_csv = load_predictions_csv(str(PREDICTIONS_PATH))
csv_dates = []
if not df_csv.empty:
    df_csv["timestamp_dt"] = pd.to_datetime(df_csv["timestamp"])
    csv_dates = df_csv["timestamp_dt"].dt.strftime("%d - %m - %Y").unique().tolist()

# 3. Consolidate Data Sources
api_dates_friendly = []
for d in api_dates:
    try:
        api_dates_friendly.append(datetime.strptime(d, "%Y-%m-%d").strftime("%d - %m - %Y"))
    except:
        pass

timestamp_prod = sorted(list(set(api_dates_friendly + csv_dates)), reverse=True)
if not timestamp_prod:
    timestamp_prod = ["No Data Available"]


# --- SIDEBAR UI ---

with st.sidebar:
    st.markdown("## 🌪️ TorNet")
    st.markdown('<p style="color:#64748b;font-size:0.85rem;margin-top:-8px">Real-Time Tornado Monitoring</p>',
                unsafe_allow_html=True)

    st.divider()

    st.markdown('<p class="section-header">📍 Target Location</p>', unsafe_allow_html=True)

    cities = {
        "— Custom —":           (None, None),
        "Oklahoma City, OK":       (35.4676, -97.5164),
        "Wichita, KS":             (37.6872, -97.3301),
        "Dallas, TX":              (32.7767, -96.7970),
        "Nashville, TN":           (36.1627, -86.7816),
        "Chicago, IL":             (41.8781, -87.6298),
    }

    preset = st.selectbox("Preset City", list(cities.keys()), key="preset_city")
    prelat, prelon = cities[preset]

    default_lat = prelat if prelat is not None else 37.6872
    default_lon = prelon if prelon is not None else -97.3301

    col_a, col_b = st.columns(2)
    with col_a:
        user_lat = st.number_input("Lat", value=default_lat, step=0.01, format="%.4f")
    with col_b:
        user_lon = st.number_input("Lon", value=default_lon, step=0.01, format="%.4f")

    st.divider()
    st.markdown('<p class="section-header">🛡️ Alert Thresholds</p>', unsafe_allow_html=True)

    threshold_km = st.slider("Alert Radius (km)", 50, 1000, 250, 25)

    st.markdown('<p class="section-header">📅 Dataset Control</p>', unsafe_allow_html=True)

    if "data_source_select" not in st.session_state:
        st.session_state["data_source_select"] = "Live API" if api_online else "Local CSV"

    src_val = st.session_state["data_source_select"]
    ts_options = (api_dates_friendly if src_val == "Live API" else csv_dates) or ["No Data Available"]

    timestamp = st.selectbox("Date / Timestamp", options=ts_options, key="timestamp_select")

    data_source = st.radio(
        "Source Mode",
        options=["Live API", "Local CSV"],
        key="data_source_select"
    )

    st.divider()
    st.markdown('<p class="section-header">⚙️ Operations</p>', unsafe_allow_html=True)

    show_all_scans = st.toggle("Show background scans (Blue Dots)", value=True, key="show_all_toggle")
    auto_refresh = st.toggle(f"Auto‑refresh ({REFRESH_INTERVAL_SECONDS}s)", value=True)
    
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    if api_online:
        mn = api_metadata.get("model", "Model")
        mv = api_metadata.get("version", "1.0")
        st.markdown(f'<span style="color:#10b981; font-weight:700">● API ONLINE | {mn} v{mv}</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span style="color:#ef4444; font-weight:700">○ API OFFLINE</span>', unsafe_allow_html=True)


# --- MAIN EXECUTION LOGIC ---

df = pd.DataFrame()
if timestamp and timestamp != "No Data Available":
    try:
        target_iso = datetime.strptime(timestamp, "%d - %m - %Y").strftime("%Y-%m-%d")
        
        if data_source == "Live API":
            if api_online and target_iso in api_dates:
                df = fetch_api_forecast(target_iso)
                if df.empty and not df_csv.empty:
                    st.warning("API Response Empty. Falling back to local cache.")
                    df = df_csv[df_csv["timestamp_dt"].dt.strftime("%d - %m - %Y") == timestamp].copy()
            else:
                if not df_csv.empty:
                    df = df_csv[df_csv["timestamp_dt"].dt.strftime("%d - %m - %Y") == timestamp].copy()
        else:
            if not df_csv.empty:
                df = df_csv[df_csv["timestamp_dt"].dt.strftime("%d - %m - %Y") == timestamp].copy()

    except Exception as e:
        st.error(f"Error loading stream: {e}")

# Process and filter data
if not df.empty:
    df = enrich_with_distance(df, user_lat, user_lon)
    df_tornados = df[df["tornado_detected"] == 1].copy()
    df_in_range = df_tornados[df_tornados["distance_km"] <= threshold_km]
    closest_dist = df_in_range["distance_km"].min() if not df_in_range.empty else None
    alert_status = not df_in_range.empty
else:
    df_tornados = df_in_range = pd.DataFrame()
    closest_dist = None
    alert_status = False


# --- DASHBOARD UI LAYOUT ---

st.markdown("# 🌪️ TorNet Tornado Alert Dashboard")
st.markdown('<p style="color:#64748b;margin-top:-12px;margin-bottom:4px">Unified Real-Time Monitoring & Prediction System</p>', unsafe_allow_html=True)

# Ticker Status
now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
st.markdown(
    f'<div class="ticker-wrap">'
    f'<span class="{"ticker-alert" if alert_status else "ticker-ok"}">{"● ACTIVE" if not df.empty else "◌ IDLE"}</span>'
    f'&nbsp;&nbsp;|&nbsp;&nbsp;Update: {now_str}'
    f'&nbsp;&nbsp;|&nbsp;&nbsp;Radius: {threshold_km} km'
    f'&nbsp;&nbsp;|&nbsp;&nbsp;Threats Detected: {len(df_in_range)}'
    f'</div>',
    unsafe_allow_html=True
)

if df.empty:
    st.info("📂 **Monitoring Active.** Waiting for inference packets from the pipeline...")
else:
    # Alert Banner
    if alert_status:
        st.markdown(
            f'<div class="status-danger">'
            f'<p class="status-text-danger">TORNADO ALERT!</p>'
            f'<p class="status-sub">Detected within <strong>{closest_dist:.1f} km</strong>. Review coordinates immediately.</p>'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="status-safe">'
            f'<p class="status-text-safe">✅ CLEAR SKIES</p>'
            f'<p class="status-sub">No threats detected within {threshold_km} km perimeter.</p>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Metrics
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1:
        st.metric("📡 Scans", f"{len(df['sensor'].unique()):,}")
    with mcol2:
        st.metric("🌪️ Tornadoes", len(df_tornados["sensor"].unique()))
    with mcol3:
        p_rate = (len(df_tornados) / len(df) * 100) if not df.empty else 0
        st.metric("📊 Positive Rate", f"{p_rate:.1f}%")
    with mcol4:
        st.metric("⚠️ In Range", len(df_in_range), delta=f"<{threshold_km}km")

    st.markdown("<br>", unsafe_allow_html=True)

    # Main Visuals
    map_col, data_col = st.columns([3, 2], gap="large")

    with map_col:
        st.markdown('<p class="section-header">🗺️ Interactive Threat Map</p>', unsafe_allow_html=True)
        t_map = build_map(df, user_lat, user_lon, threshold_km, show_all=show_all_scans)
        st_folium(t_map, width=None, height=520, key="tornet_map", returned_objects=[])

    with data_col:
        st.markdown('<p class="section-header">🚨 Active Perimeter Alerts</p>', unsafe_allow_html=True)
        if df_in_range.empty:
            st.markdown(
                '<div style="background:#052e16;border:1px solid #10b981;border-radius:12px;'
                'padding:40px;text-align:center;color:#34d399;font-weight:600">'
                '✅ AREA SECURE<br><span style="font-weight:400;font-size:0.8rem">All monitored sensors report clear.</span></div>',
                unsafe_allow_html=True
            )
        else:
            df_disp = df_in_range[["scan_id", "distance_km", "probability", "sensor"]].copy()
            df_disp["probability"] = df_disp["probability"].apply(lambda x: f"{x:.1%}")
            st.dataframe(df_disp, use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-header">📍 Closest Detections</p>', unsafe_allow_html=True)
        if not df_tornados.empty:
            top_detects = df_tornados.nsmallest(10, "distance_km")[["scan_id", "distance_km", "probability", "sensor"]]
            st.table(top_detects)
        else:
            st.caption("No detections recorded in current time window.")

    # Data Inspector
    with st.expander("🔬 Deep Packet Inspection (Raw Predictions)", expanded=False):
        st.dataframe(df, use_container_width=True)


# --- FOOTER ---
st.divider()
st.markdown(
    '<p style="text-align:center;color:#64748b;font-size:0.8rem">'
    'TorNet Dashboard &nbsp;·&nbsp; v2.1.0 &nbsp;·&nbsp; '
    'Academic Demonstration Dataset'
    '</p>',
    unsafe_allow_html=True
)

if auto_refresh:
    time.sleep(REFRESH_INTERVAL_SECONDS)
    st.rerun()