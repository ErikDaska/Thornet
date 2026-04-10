@echo off
echo =========================================
echo  Starting TorNet MLOps Environment Setup
echo =========================================

echo [1/3] Verifying directory structure...
:: We only need the base folders; Airflow will create the rest
if not exist "data\raw" mkdir "data\raw"
if not exist "data\processed" mkdir "data\processed"
if not exist "config" mkdir "config"

echo [2/3] Spinning up Infrastructure (Airflow/MLflow)...
:: This starts the actual 'workers'
docker compose up -d
echo Waiting 15s for services to initialize...
timeout /t 15 /nobreak >nul

echo [3/3] Configuring local Python environment...
:: We keep the venv only for local development/IDE support
if not exist venv (
    python -m venv venv
)
call venv\Scripts\activate
pip install -r requirements.txt -q

echo ===================================================
echo  Infrastructure Ready! 
echo  1. MLflow: http://127.0.0.1:5000
echo  2. Airflow: http://127.0.0.1:8080
echo  ---------------------------------------------------
echo  Note: Data Ingestion, Processing, and DVC 
echo  tracking are now handled entirely by Airflow.
echo ===================================================
pause