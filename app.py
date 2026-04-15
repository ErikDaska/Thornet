"""
app.py — TorNet Tornado Alert Dashboard
Real-Time Tornado Alert Dashboard
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


# Define the expected structure of our data
PredictionSchema = pa.DataFrameSchema({
    "scan_id": pa.Column(str),
    "timestamp": pa.Column(str),
    "latitude": pa.Column(float, checks=pa.Check.in_range(-90, 90)),
    "longitude": pa.Column(float, checks=pa.Check.in_range(-180, 180)),
    "probability": pa.Column(float, checks=pa.Check.in_range(0, 1)),
    "tornado_detected": pa.Column(int, checks=pa.Check.isin([0, 1])),
    "sensor": pa.Column(str),
})


# PAGE CONFIGURATION

st.set_page_config(
    page_title="TorNet — Tornado Alert Dashboard",
    page_icon="🌪️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# LOAD CUSTOM CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("css/style.css")


# CONSTANTS
PREDICTIONS_PATH = Path(os.getenv("PREDICTIONS_CSV", "/app/data/offline_data_fallback.csv")) 
API_URL = os.getenv("API_URL", "http://fastapi-service:80")
REFRESH_INTERVAL_SECONDS = 30

ALERT_COLORS = {
    "CRITICAL": "#ef4444",
    "HIGH":     "#fb923c",
    "MODERATE": "#eab308",
    "NONE":     "#6b7280",
}


# HELPER FUNCTIONS
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in km between two points using the Haversine formula."""
    R = 6371.0  # Earth's mean radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi       = math.radians(lat2 - lat1)
    dlambda    = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


@st.cache_data(ttl=REFRESH_INTERVAL_SECONDS)
def load_predictions_csv(path: str) -> pd.DataFrame:
    """Loads and validates the predictions CSV."""
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(p)
        df.columns = [c.lower().strip() for c in df.columns]
        return PredictionSchema.validate(df)
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=600)
def fetch_api_inventory() -> list:
    """Fetches list of available dates from the API."""
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
    """Fetches forecast for a specific date from the API."""
    try:
        # Note: st.spinner is handled in the main loop to avoid cache issues
        with httpx.Client(timeout=300.0) as client:
            resp = client.post(f"{API_URL}/api/v1/forecast", json={"date_": target_date_iso})
            if resp.status_code == 200:
                data = resp.json()
                predictions = data.get("predictions", [])
                if not predictions:
                    return pd.DataFrame()
                
                df = pd.DataFrame(predictions)
                # Schema Adaptation
                df["probability"] = df["tornado_probability"]
                df["latitude"] = df["lat"]
                df["longitude"] = df["lon"]
                df["tornado_detected"] = (df["probability"] > 0.5).astype(int)
                df["scan_id"] = [f"api_{i:04d}" for i in range(len(df))]
                # Sensor is already in the API response now
                
                return PredictionSchema.validate(df)
    except Exception as e:
        st.sidebar.warning(f"API Fetch Error: {e}")
    return pd.DataFrame()


def enrich_with_distance(df: pd.DataFrame, user_lat: float, user_lon: float) -> pd.DataFrame:
    """Adds 'distance_km' column with Haversine distance to the user."""
    if df.empty:
        return df
    df = df.copy()
    df["distance_km"] = df.apply(
        lambda r: haversine_km(user_lat, user_lon, float(r["latitude"]), float(r["longitude"])),
        axis=1
    ).round(1)
    return df.sort_values("distance_km")


def build_map(df: pd.DataFrame, user_lat: float, user_lon: float,
              threshold_km: float, active_only: bool) -> folium.Map:
    """Builds the Folium map with user, tornado markers, and normal scans."""
    m = folium.Map(
        location=[user_lat, user_lon],
        zoom_start=5,
        tiles=None,
    )

    # Tile layer 
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; '
             '<a href="https://carto.com/attributions">CARTO</a>',
        name="Dark Matter",
        max_zoom=19,
    ).add_to(m)

    # Alert radius circle
    folium.Circle(
        location=[user_lat, user_lon],
        radius=threshold_km * 1000,  # meters
        color="#3b82f6",
        weight=1.5,
        fill=True,
        fill_color="#3b82f6",
        fill_opacity=0.05,
        tooltip=f"Alert Radius: {threshold_km:.0f} km",
    ).add_to(m)

    # User location
    folium.Marker(
        location=[user_lat, user_lon],
        tooltip="📍 Your Location",
        popup=folium.Popup(
            f"<b>Your Location</b><br>Lat: {user_lat:.4f}<br>Lon: {user_lon:.4f}",
            max_width=200
        ),
        icon=folium.Icon(color="blue", icon="home", prefix="fa"),
    ).add_to(m)

    # Markers (Tornados & Normal Scans)
    if not df.empty:
        # 1. Isolar os tornados reais
        tornados = df[df["tornado_detected"] == 1]
        sensores_com_tornado = tornados["sensor"].unique()

        # 2. Desenhar scans normais APENAS se "active_only" for False
        # E garantindo que não desenhamos por cima de um sensor que já tem tornado
        if not active_only:
            scans_normais = df[
                (df["tornado_detected"] == 0) & 
                (~df["sensor"].isin(sensores_com_tornado))
            ]

            for _, row in scans_normais.iterrows():
                lat = float(row["latitude"])
                lon = float(row["longitude"])
                ses = str(row.get("sensor", "NONE"))

                # Círculo azul pequeno para scans sem alerta
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=5,  # Mais pequeno que o tornado para não distrair
                    color="#3b82f6", # Azul Folium/Tailwind
                    weight=1,
                    fill=True,
                    fill_color="#3b82f6",
                    fill_opacity=0.4,
                    tooltip=f"ℹ️ Clear Scan | Sensor: {ses}"
                ).add_to(m)


        # 3. Desenhar os Tornados (Sempre visíveis se existirem)
        for _, row in tornados.iterrows():
            lat  = float(row["latitude"])
            lon  = float(row["longitude"])
            prob = float(row.get("probability", 0.5))
            dist = float(row.get("distance_km", 0))
            ses  = str(row.get("sensor", "NONE"))
            sid  = str(row.get("scan_id", "?"))

            color = ("red" if ses == "CRITICAL" else
                     "orange" if ses == "HIGH" else "beige")

            # Círculo vermelho/laranja de impacto
            folium.CircleMarker(
                location=[lat, lon],
                radius=12 + prob * 10,
                color=ALERT_COLORS.get(ses, "#ef4444"),
                weight=2,
                fill=True,
                fill_color=ALERT_COLORS.get(ses, "#ef4444"),
                fill_opacity=0.35,
            ).add_to(m)
            
            # Ícone central de alerta
            folium.Marker(
                location=[lat, lon],
                tooltip=f"🌪️ {ses} | {dist:.0f} km | p={prob:.2f}",
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
                icon=folium.Icon(color=color, icon="bolt", prefix="fa"),
            ).add_to(m)

    folium.LayerControl().add_to(m)
    return m

# --- DATA SOURCE INITIALIZATION ---
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

# Load Fallback CSV
df_csv = load_predictions_csv(str(PREDICTIONS_PATH))
csv_dates = []
if not df_csv.empty:
    df_csv["timestamp_dt"] = pd.to_datetime(df_csv["timestamp"])
    csv_dates = df_csv["timestamp_dt"].dt.strftime("%d - %m - %Y").unique().tolist()

# Normalize API dates to friendly format for the UI
api_dates_friendly = []
for d in api_dates:
    try:
        api_dates_friendly.append(datetime.strptime(d, "%Y-%m-%d").strftime("%d - %m - %Y"))
    except:
        pass

# Consolidate all available dates for the selector
timestamp_prod = sorted(list(set(api_dates_friendly + csv_dates)), reverse=True)
if not timestamp_prod:
    timestamp_prod = ["No Data Available"]

# SIDEBAR
with st.sidebar:
    st.markdown("## 🌪️ TorNet")
    st.markdown('<p style="color:#64748b;font-size:0.85rem;margin-top:-8px">Tornado Alert Dashboard</p>',
                unsafe_allow_html=True)

    st.divider()

    st.markdown('<p class="section-header">📍 Location</p>', unsafe_allow_html=True)

    # City Presets
    presets = {
        "— Custom —":           (None, None),
        "Oklahoma City, OK 🌪️":       (35.4676, -97.5164),
        "Wichita, KS 🌪️":            (37.6872, -97.3301),
        "Dallas, TX":                 (32.7767, -96.7970),
        "Kansas City, MO":            (39.0997, -94.5786),
        "Nashville, TN":              (36.1627, -86.7816),
        "Chicago, IL":                (41.8781, -87.6298),
        "Denver, CO":                 (39.7392, -104.9903),
        "New York, NY":               (40.7128, -74.0060),
    }

    preset = st.selectbox("🏙️ Preset City", list(presets.keys()), key="preset_city")
    prelat, prelon = presets[preset]

    default_lat = prelat if prelat is not None else 37.6872
    default_lon = prelon if prelon is not None else -97.3301

    col_a, col_b = st.columns(2)
    with col_a:
        user_lat = st.number_input("Latitude", value=default_lat,
                                   min_value=24.0, max_value=50.0,
                                   step=0.01, format="%.4f", key="lat_input")
    with col_b:
        user_lon = st.number_input("Longitude", value=default_lon,
                                   min_value=-125.0, max_value=-67.0,
                                   step=0.01, format="%.4f", key="lon_input")

    st.divider()
    st.markdown('<p class="section-header">⚙️ Alert Parameters</p>', unsafe_allow_html=True)

    threshold_km = st.slider(
        "Alert Radius (km)",
        min_value=50,
        max_value=1000,
        value=200,
        step=25,
        help="Maximum distance to issue a tornado alert",
        key="threshold_slider"
    )

    st.markdown('<p class="section-header"> TimeStamp with Data</p>', unsafe_allow_html=True)

    timestamp = st.selectbox(
        "Select Timestamp",
        options=timestamp_prod,
        key="timestamp_select"
    )

    show_all_scans = st.toggle("Show all scans on map", value=False, key="show_all_toggle")

    st.divider()
    st.markdown('<p class="section-header">🔄 Data</p>', unsafe_allow_html=True)

    auto_refresh = st.toggle(f"Auto‑refresh ({REFRESH_INTERVAL_SECONDS}s)", value=True, key="auto_refresh")
    
    data_source = st.radio(
        "Select Data Source",
        options=["Live API (Real-time)", "Local CSV (Offline Fallback)"],
        index=0 if api_online else 1,
        help="Choose between real-time model inference or cached offline data.",
        key="data_source_select"
    )
    if st.button("🔄 Refresh Now", use_container_width=True, key="refresh_btn"):
        st.cache_data.clear()
        st.rerun()

    st.caption(f"📂 `{PREDICTIONS_PATH}`")

    if api_online:
        mv = api_metadata.get("version", "unknown")
        dv = api_metadata.get("device", "cpu")
        st.markdown(f'<span style="color:#10b981; font-weight:700">● ONLINE | v{mv} ({dv.upper()})</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span style="color:#ef4444; font-weight:700">○ OFFLINE</span>', unsafe_allow_html=True)
        st.caption("Using CSV fallback.")

# --- DATA SELECTION LOGIC ---
df = pd.DataFrame()
if timestamp and timestamp != "No Data Available":
    try:
        target_iso = datetime.strptime(timestamp, "%d - %m - %Y").strftime("%Y-%m-%d")
        
        # Choice 1: Live API
        if data_source == "Live API (Real-time)":
            if api_online and target_iso in api_dates:
                with st.spinner(f"Running Deep Learning Inference for {target_iso}... (First time may take a few minutes)"):
                    df = fetch_api_forecast(target_iso)
                
                # RELIABILITY-FIRST: Automatic failover if API fails
                if df.empty and not df_csv.empty:
                    st.warning("API fetch failed or timed out. Automatically falling back to local CSV records.")
                    df = df_csv[df_csv["timestamp_dt"].dt.strftime("%d - %m - %Y") == timestamp].copy()
            else:
                if not api_online:
                    st.error("API Backend is currently OFFLINE. Automatically attempting to load from Local CSV...")
                else:
                    st.warning(f"No API data for {target_iso}. Attempting CSV fallback...")
                
                if not df_csv.empty:
                    df = df_csv[df_csv["timestamp_dt"].dt.strftime("%d - %m - %Y") == timestamp].copy()
            
        # Choice 2: Local CSV
        else:
            if not df_csv.empty:
                df = df_csv[df_csv["timestamp_dt"].dt.strftime("%d - %m - %Y") == timestamp].copy()
            else:
                st.error("Fallback CSV not found. Please trigger the Airflow pipeline.")

    except Exception as e:
        st.error(f"Error loading data for {timestamp}: {e}")

# Enrich the selected data with distances
if not df.empty:
    df = enrich_with_distance(df, user_lat, user_lon)

# Active tornadoes within radius
if not df.empty:
    df_tornados   = df[df["tornado_detected"] == 1].copy()
    df_in_range   = df_tornados[df_tornados["distance_km"] <= threshold_km]
    closest_dist  = df_in_range["distance_km"].min() if not df_in_range.empty else None
    max_prob      = df_tornados["probability"].max() if not df_tornados.empty else 0.0
    sensor   = df_in_range["sensor"].iloc[0] if not df_in_range.empty else "NONE"
else:
    df_tornados = df_in_range = pd.DataFrame()
    closest_dist = None
    max_prob = 0.0
    sensor = "NONE"

is_alert = not df_in_range.empty

# HEADER
st.markdown("# 🌪️ TorNet Tornado Alert Dashboard")
st.markdown('<p style="color:#64748b;margin-top:-12px;margin-bottom:4px">Real-Time Storm Monitoring System &nbsp;·&nbsp; Powered by MLflow + Apache Airflow</p>',
            unsafe_allow_html=True)

# Status ticker
now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
data_status = "● LIVE" if not df.empty else "◌ NO DATA"
ticker_class = "ticker-alert" if is_alert else "ticker-ok"
st.markdown(
    f'<div class="ticker-wrap">'
    f'<span class="{ticker_class}">{data_status}</span>'
    f'&nbsp;&nbsp;|&nbsp;&nbsp;Last update: {now_str}'
    f'&nbsp;&nbsp;|&nbsp;&nbsp;Location: {user_lat:.4f}°N, {abs(user_lon):.4f}°W'
    f'&nbsp;&nbsp;|&nbsp;&nbsp;Alert radius: {threshold_km} km'
    f'&nbsp;&nbsp;|&nbsp;&nbsp;Tornadoes in range: {"⚠️ " + str(len(df_in_range)) if is_alert else "✓ 0"}'
    f'</div>',
    unsafe_allow_html=True
)


# MAIN STATUS PANEL
if df.empty:
    st.warning("⚠️ No data available. The inference pipeline has not yet generated predictions.")
    st.info(f"📂 Waiting for data in: `{PREDICTIONS_PATH}`\n\nTrigger the DAG `tornado_inference_realtime` on Airflow (`localhost:8080`) to generate the data.")
else:
    if is_alert:
        min_dist_str  = f"{closest_dist:.1f} km" if closest_dist is not None else "?"
        alert_count   = len(df_in_range["sensor"].unique())
        st.markdown(
            f'<div class="status-danger">'
            f'<p class="status-text-danger">TORNADO ALERT!</p>'
            f'<p class="status-sub">Tornado detected within <strong>{min_dist_str}</strong> from your location &nbsp;·&nbsp; '
            f'{alert_count} tornado{"es" if alert_count > 1 else ""} detected; '
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        n_tornados_total = len(df_tornados["sensor"].unique())
        st.markdown(
            f'<div class="status-safe">'
            f'<p class="status-text-safe">✅ SAFE ZONE</p>'
            f'<p class="status-sub">No tornado detected within {threshold_km} km &nbsp;·&nbsp; '
            f'{n_tornados_total} tornado{"es" if n_tornados_total != 1 else ""} monitored</p>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📡 Total Scans",   f"{len(df["sensor"].unique()):,}")
    with col2:
        st.metric("🌪️ Tornadoes Detected", f"{len(df_tornados["sensor"].unique()):,}")
    with col3:
        pct = len(df_tornados["sensor"].unique()) / len(df["sensor"].unique()) * 100 if len(df) > 0 else 0
        st.metric("📊 Positive Rate",    f"{pct:.1f}%")
    with col4:
        st.metric("⚠️ On Alert",        f"{len(df_in_range):,}",
                  delta=f"within {threshold_km} km")

    st.markdown("<br>", unsafe_allow_html=True)

    # Map of Tornadoes + Active Alerts
    col_map, col_alerts = st.columns([3, 2], gap="large")

    with col_map:
        st.markdown('<p class="section-header">🗺️ Map of Tornadoes</p>', unsafe_allow_html=True)
        tornado_map = build_map(df, user_lat, user_lon, threshold_km, active_only=not show_all_scans)
        st_folium(tornado_map, width=None, height=500, key="tornado_map",
                  returned_objects=[])

    with col_alerts:
        st.markdown('<p class="section-header">🚨 Active Alerts</p>', unsafe_allow_html=True)

        if df_in_range.empty:
            st.markdown(
                '<div style="background:#052e16;border:1px solid #10b981;border-radius:12px;'
                'padding:24px;text-align:center;color:#34d399;font-size:1.1rem;font-weight:600">'
                '✅ No active alerts<br>'
                '<span style="font-weight:400;font-size:0.85rem;color:#6ee7b7">'
                f'Area within {threshold_km} km is clear</span></div>',
                unsafe_allow_html=True
            )
        else:
            display_cols = ["scan_id", "distance_km", "probability", "sensor", "latitude", "longitude"]
            display_cols = [c for c in display_cols if c in df_in_range.columns]
            df_display = df_in_range[display_cols].copy()

            # Format columns
            if "probability" in df_display.columns:
                df_display["probability"] = df_display["probability"].apply(lambda x: f"{x:.1%}")
            if "distance_km" in df_display.columns:
                df_display["distance_km"] = df_display["distance_km"].apply(lambda x: f"{x:.1f} km")
            if "latitude" in df_display.columns:
                df_display["latitude"]  = df_display["latitude"].apply(lambda x: f"{x:.4f}°")
                df_display["longitude"] = df_display["longitude"].apply(lambda x: f"{x:.4f}°")

            df_display.columns = ["Scan ID", "Distance", "Prob.", "Sensor", "Lat", "Lon"]
            st.dataframe(df_display, use_container_width=True, hide_index=True,
                         height=min(300, 36 + 35 * len(df_display)))

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-header">📍 Closest Tornadoes (Top 10)</p>', unsafe_allow_html=True)
        df_tornados_sensor = df_tornados.groupby("sensor").first().reset_index()
        if not df_tornados_sensor.empty:
            top10 = df_tornados_sensor.nsmallest(10, "distance_km")[
                ["scan_id", "distance_km", "probability", "sensor"]
            ].copy()
            top10.columns = ["Scan ID", "Dist. (km)", "Prob.", "Sensor"]
            top10["Prob."] = top10["Prob."].apply(lambda x: f"{x:.1%}")
            top10["Dist. (km)"] = top10["Dist. (km)"].apply(lambda x: f"{x:.1f}")
            st.dataframe(top10, use_container_width=True, hide_index=True, height=360)
        else:
            st.info("No tornado data available in the current dataset.")

    # Complete data table (expandable)
    with st.expander("📋 See all prediction data", expanded=False):
        if not df.empty:
            df_show = df.copy()
            show_cols = ["scan_id", "timestamp", "tornado_detected",
                         "probability", "distance_km", "sensor",
                         "latitude", "longitude"]
            show_cols = [c for c in show_cols if c in df_show.columns]
            st.dataframe(
                df_show[show_cols].style.applymap(
                    lambda v: "background-color: rgba(239,68,68,0.15); color: #fca5a5" if v == 1 else "",
                    subset=["tornado_detected"] if "tornado_detected" in show_cols else []
                ),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No tornado data available in the current dataset.")


# FOOTER
st.divider()
st.markdown(
    '<p style="text-align:center;color:#374151;font-size:0.8rem">'
    '🌪️ TorNet Tornado Alert Dashboard &nbsp;·&nbsp; '
    'Powered by <strong style="color:#4f46e5">MLflow</strong>, '
    '<strong style="color:#0ea5e9">Apache Airflow</strong> & '
    '<strong style="color:#e11d48">PyTorch</strong> &nbsp;·&nbsp; '
    'Simulated data for academic demonstration'
    '</p>',
    unsafe_allow_html=True
)

# AUTO REFRESH
if auto_refresh:
    time.sleep(REFRESH_INTERVAL_SECONDS)
    st.cache_data.clear()
    st.rerun()