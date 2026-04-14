from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date, datetime, time, timezone
import mlflow
import torch
import numpy as np
import pandas as pd
import logging
import os
import glob
import xarray as xr
import re

# Configure isolated logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Thornet API", description="MaaS: Model-as-a-Service for PyTorch inference.")

# Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_server:5000")
MODEL_NAME = "ThornetTornadoPrediction"
ALIAS = "production"
DATA_DIR = os.getenv("PROCESSED_DATA_DIR", "/data/processed")
AVAILABLE_DATES = []

VAR_BOUNDS = {
    'DBZ':   {'min': -32.0, 'max': 95.0},
    'VEL':   {'min': -100.0, 'max': 100.0},
    'KDP':   {'min': -10.0, 'max': 10.0},
    'RHOHV': {'min': 0.0, 'max': 1.05},
    'ZDR':   {'min': -8.0, 'max': 15.0},
    'WIDTH': {'min': 0.0, 'max': 30.0}
}

CHANNEL_ORDER = ['DBZ', 'VEL', 'KDP', 'RHOHV', 'ZDR', 'WIDTH', 'MASK']

# Global State for Model
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@app.on_event("startup")
def startup_event():
    """Load model and scan inventory on startup."""
    global model, AVAILABLE_DATES
    
    # 1. Load Model from MLflow
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()
        model_info = client.get_model_version_by_alias(name=MODEL_NAME, alias=ALIAS)
        model_uri = f"models:/{MODEL_NAME}/{model_info.version}"

        model = mlflow.pytorch.load_model(model_uri, map_location=torch.device("cpu"))
        model = model.to(device)
        model.eval()
        logger.info(f"Loaded {MODEL_NAME} | Version: {model_info.version} on {device}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")

    # 2. Build Inventory
    scan_available_data()

# Load Radar Coordinates Database
try:
    RADAR_DF = pd.read_csv("/data/radars/radars.csv", index_col="radar_id")
    logger.info("Successfully loaded radar coordinates database.")
except Exception as e:
    logger.warning(f"Could not load radars.csv: {e}")
    RADAR_DF = pd.DataFrame()

# --- Schemas ---
class PredictionItem(BaseModel):
    radar_id: str
    lat: float
    lon: float
    tornado_probability: float
    timestamp: str  # Required by Streamlit app

class ForecastResponse(BaseModel):
    target_date: date
    predictions: List[PredictionItem]

class ForecastRequest(BaseModel):
    date_: date

# --- Helpers ---
def load_and_preprocess_file(filepath: str) -> torch.Tensor:
    """Reads NetCDF and converts to normalized Tensor."""
    with xr.open_dataset(filepath, engine="h5netcdf") as ds:
        if 'time' in ds.dims:
            ds = ds.isel(time=0)

        if 'DBZ' in ds.data_vars:
            ds['MASK'] = (~ds['DBZ'].isnull()).astype(float)
        else:
            ds['MASK'] = xr.zeros_like(ds[list(ds.data_vars)[0]])

        ds = ds.fillna(0.0)

        for var, bounds in VAR_BOUNDS.items():
            if var in ds.data_vars:
                ds[var] = (ds[var] - bounds['min']) / (bounds['max'] - bounds['min'])
                ds[var] = ds[var].clip(0.0, 1.0)

        channels_data = []
        for var in CHANNEL_ORDER:
            if var in ds.data_vars:
                channels_data.append(ds[var].values.copy())
            else:
                shape = (ds.sizes.get('sweep', 1), ds.sizes.get('azimuth', 1), ds.sizes.get('range', 1))
                channels_data.append(np.zeros(shape, dtype=np.float32))

    stacked_data = np.stack(channels_data, axis=0)
    return torch.tensor(stacked_data, dtype=torch.float32).unsqueeze(0)

# No main.py, confirma que estas funções estão assim:

def scan_available_data():
    global AVAILABLE_DATES
    # Adicionamos um log para debug
    logger.info(f"Scanning for .nc files in: {DATA_DIR}")
    
    # Procura ficheiros .nc em todas as subpastas
    all_nc_files = glob.glob(os.path.join(DATA_DIR, "**", "*.nc"), recursive=True)
    
    unique_dates = set()
    for f in all_nc_files:
        # Extrai YYMMDD (ex: 131222) do nome do ficheiro
        match = re.search(r"_(\d{6})_", os.path.basename(f))
        if match:
            try:
                # Converte para objeto date e depois para ISO string
                dt = datetime.strptime(match.group(1), "%y%m%d").date()
                unique_dates.add(dt.isoformat())
            except Exception:
                continue
    
    # Atualiza a lista global
    AVAILABLE_DATES = sorted(list(unique_dates), reverse=True)
    logger.info(f"Inventory updated: {len(AVAILABLE_DATES)} dates found.")



# --- Endpoints ---

@app.get("/api/v1/inventory")
def get_inventory():
    if not AVAILABLE_DATES:
        scan_available_data()
    return {"dates": AVAILABLE_DATES}


@app.get("/api/v1/radars")
def get_radars():
    if RADAR_DF.empty:
        logger.warning("Data base of radars (RADAR_DF) is empty.")
        return {}
    
    return RADAR_DF.fillna(0.0).to_dict(orient="index")

@app.get("/")
def read_root():
    return {"status": "online", "message": "ThorNet API is running"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/api/v1/forecast", response_model=ForecastResponse)
def generate_forecast(request: ForecastRequest):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        date_str = request.date_.strftime("%y%m%d")
        search_pattern = os.path.join(DATA_DIR, "**", f"*_{date_str}_*.nc")
        matched_files = glob.glob(search_pattern, recursive=True)

        results = []
        with torch.no_grad():
            for filepath in matched_files:
                filename = os.path.basename(filepath)
                parts = filename.split("_")
                
                # Extract metadata from filename: PREFIX_DATE_TIME_RADARID_...
                try:
                    raw_time = parts[2] # HHMMSS
                    radar_id = parts[3]
                    
                    # Construct valid ISO timestamp for the Streamlit filter
                    dt_obj = datetime.combine(
                        request.date_, 
                        time(hour=int(raw_time[:2]), minute=int(raw_time[2:4]), second=int(raw_time[4:6]))
                    ).replace(tzinfo=timezone.utc)
                    ts_str = dt_obj.isoformat()
                except:
                    radar_id = "UNKNOWN"
                    ts_str = datetime.combine(request.date_, time(0)).isoformat()

                # Inference
                input_tensor = load_and_preprocess_file(filepath).to(device)
                prob = torch.sigmoid(model(input_tensor)).item()

                # Geo Lookup
                lat, lon = 0.0, 0.0
                if radar_id in RADAR_DF.index:
                    lat = float(RADAR_DF.loc[radar_id, "lat"])
                    lon = float(RADAR_DF.loc[radar_id, "lon"])

                results.append(PredictionItem(
                    radar_id=radar_id,
                    lat=lat,
                    lon=lon,
                    tornado_probability=prob,
                    timestamp=ts_str
                ))

        return ForecastResponse(target_date=request.date_, predictions=results)

    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")