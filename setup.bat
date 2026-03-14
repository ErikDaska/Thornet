@echo off
echo ===================================================
echo  Starting TorNet MLOps Environment Setup (Windows)
echo ===================================================

echo [1/6] Verifying directory structure...
if not exist "..\emi_remote" mkdir "..\emi_remote"
if not exist "data\raw\tornet" mkdir "data\raw\tornet"
if not exist "src" mkdir "src"

echo [2/6] Spinning up MLflow Tracking Server...
docker compose up -d

echo [3/6] Configuring Python environment...
if exist venv (
    echo    Cleaning up old broken virtual environment...
    rmdir /s /q venv
)
python -m venv venv
call venv\Scripts\activate
pip install -r requirements.txt zenodo-get -q

echo [4/6] Checking dataset status...
dir /b /a "data\raw\tornet" | findstr . >nul 2>&1
if errorlevel 1 (
    echo    Dataset not found locally. Initiating automatic download via Zenodo...
    echo    Downloading the 2013 subset - 3GB - Please wait...
    zenodo_get 12636522 -o data\raw\
    echo    Extracting archive...
    tar -xzf data\raw\tornet_2013.tar.gz -C data\raw\tornet\
    echo    Cleaning up zip file...
    del data\raw\tornet_2013.tar.gz
    echo    Download and extraction complete!
) else (
    echo    Dataset found locally.
)

echo [5/6] Syncing with local DVC remote...
dvc remote list | findstr "localremote" >nul 2>&1
if errorlevel 1 (
    dvc remote add -d localremote ..\emi_remote
)
dvc add data\raw\tornet
dvc push

echo [6/6] Executing Data Ingestion Pipeline...
python src\data_ingestion.py

echo ===================================================
echo  Setup Complete! Check MLflow at http://127.0.0.1:5000
echo ===================================================
pause