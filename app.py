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
PREDICTIONS_PATH = Path(os.getenv("PREDICTIONS_CSV", "/app/data/dados_para_teste.csv")) 
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
def load_predictions(path: str) -> pd.DataFrame:
    """Loads and validates the predictions CSV."""
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(p)
        
        # 1. Basic cleaning
        df.columns = [c.lower().strip() for c in df.columns]
        
        # 2. Strict Validation with Pandera
        validated_df = PredictionSchema.validate(df)
        
        return validated_df

    except pa.errors.SchemaError as ve:
        st.error(f"Data Validation Error: The inference output format has changed.")
        st.sidebar.error(f"Validation Detail: {ve}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Unexpected error loading data: {e}")
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
    """Builds the Folium map with user and tornado markers."""
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

    # Tornado markers
    if not df.empty:
        tornados = df[df["tornado_detected"] == 1] if active_only else df
        # Re-verify detection to ensure only relevant icons show if toggle is set
        tornados = tornados[tornados["tornado_detected"] == 1]

        for _, row in tornados.iterrows():
            lat  = float(row["latitude"])
            lon  = float(row["longitude"])
            prob = float(row.get("probability", 0.5))
            dist = float(row.get("distance_km", 0))
            ses  = str(row.get("sensor", "NONE"))
            sid  = str(row.get("scan_id", "?"))

            color = ("red" if ses == "CRITICAL" else
                     "orange" if ses == "HIGH" else "beige")

            # Intensity circle
            folium.CircleMarker(
                location=[lat, lon],
                radius=12 + prob * 10,
                color=ALERT_COLORS.get(ses, "#ef4444"),
                weight=2,
                fill=True,
                fill_color=ALERT_COLORS.get(ses, "#ef4444"),
                fill_opacity=0.35,
            ).add_to(m)
            
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

    st.markdown('<p class="section-header">Day Month and Year</p>', unsafe_allow_html=True)

    date_input = st.date_input(
        "Select Date",
        value=datetime(2014, 1, 1).date(),
        key="date_input"
    )

    st.markdown('<p class="section-header">Time of the Day</p>', unsafe_allow_html=True)

    slider_time = st.slider(
        "Select Time",
        value=datetime.now().time(),
        key="time_slider"
    )


    show_all_scans = st.toggle("Show all scans on map", value=False, key="show_all_toggle")

    st.divider()
    st.markdown('<p class="section-header">🔄 Data</p>', unsafe_allow_html=True)

    auto_refresh = st.toggle(f"Auto‑refresh ({REFRESH_INTERVAL_SECONDS}s)", value=True, key="auto_refresh")
    if st.button("🔄 Refresh Now", use_container_width=True, key="refresh_btn"):
        st.cache_data.clear()
        st.rerun()

    st.caption(f"📂 `{PREDICTIONS_PATH}`")

    # API STATUS INDICATOR (Moved to bottom)
    st.markdown("<br><br>", unsafe_allow_html=True) # Spacer
    st.divider()
    st.markdown('<p class="section-header" style="font-size:0.75rem">🔌 Backend Status</p>', unsafe_allow_html=True)
    
    api_online = False
    try:
        with httpx.Client(timeout=1.0) as client:
            resp = client.get(f"{API_URL}/")
            api_online = resp.status_code == 200
    except Exception:
        api_online = False

    if api_online:
        st.markdown('<span style="color:#10b981; font-weight:700">● ONLINE</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span style="color:#ef4444; font-weight:700">○ OFFLINE</span>', unsafe_allow_html=True)
        st.caption("Using CSV fallback.")


# LOAD DATA
df_raw = load_predictions(str(PREDICTIONS_PATH))
df = enrich_with_distance(df_raw, user_lat, user_lon) if not df_raw.empty else pd.DataFrame()

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
        alert_count   = len(df_in_range)
        icon_alert    = ALERT_ICONS.get(Sensor, "🔴")
        st.markdown(
            f'<div class="status-danger">'
            f'<p class="status-text-danger">{icon_alert} TORNADO ALERT!</p>'
            f'<p class="status-sub">Tornado detected within <strong>{min_dist_str}</strong> from your location &nbsp;·&nbsp; '
            f'{alert_count} tornado{"es" if alert_count > 1 else ""} detected within {threshold_km} km &nbsp;·&nbsp; '
            f'Sensor: <strong>{Sensor}</strong></p>'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        n_tornados_total = len(df_tornados)
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
        st.metric("📡 Total Scans",   f"{len(df):,}")
    with col2:
        st.metric("🌪️ Tornadoes Detected", f"{len(df_tornados):,}")
    with col3:
        pct = len(df_tornados) / len(df) * 100 if len(df) > 0 else 0
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

        if not df_tornados.empty:
            top10 = df_tornados.nsmallest(10, "distance_km")[
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