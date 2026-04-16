#!/bin/bash

echo "Starting TorNet MLOps Environment Setup..."

# 1. Ensure all required directories exist safely
echo "Verifying directory structure..."
mkdir -p ../emi_remote
mkdir -p data/raw/tornet
mkdir -p src

# 2. Setup Python Virtual Environment FIRST
echo "Configuring Python environment..."
python3 -m venv venv
# Activate the environment
source venv/bin/activate
# Install requirements PLUS zenodo-get safely inside the isolated environment
pip install -r requirements.txt zenodo-get -q

# 3. Start the MLflow Server via Docker Compose
echo "Spinning up MLflow Tracking Server..."
docker compose up -d

# 4. Check for Data & Auto-Download if Missing
echo "Checking dataset status..."
if [ -z "$(ls -A data/raw/tornet 2>/dev/null)" ]; then
    echo "Dataset not found locally. Initiating automatic download via Zenodo..."
    
    echo "Downloading the 2013 subset (3GB) from Zenodo..."
    # 12636522 is the official MIT-LL Zenodo record ID for the 2013 TorNet data
    zenodo_get 12636522 -o data/raw/
    
    echo "Extracting the downloaded archive..."
    # Extract the tarball directly into the tornet folder
    tar -xzf data/raw/tornet_2013.tar.gz -C data/raw/tornet/
    
    # Delete the heavy zip file to save hard drive space
    rm data/raw/tornet_2013.tar.gz
    
    echo "Dataset found locally."
fi

# 5. Automate DVC Configuration safely
echo "Syncing with local DVC remote..."
dvc remote list | grep -q "localremote" || dvc remote add -d localremote ../emi_remote

# Re-add the data folder to sync OS differences, then push to the remote
dvc add data/raw/tornet
dvc push

# 6. Run the Pipeline
echo "Executing Data Ingestion Pipeline..."
python src/data_ingestion.py

echo "✅ Setup Complete! Check MLflow at http://127.0.0.1:5000"