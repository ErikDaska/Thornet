#!/bin/bash

echo "🚀 Starting TorNet MLOps Environment Setup..."

# 1. Ensure all required directories exist safely
echo "📁 Verifying directory structure..."
mkdir -p ../emi_remote
mkdir -p data/raw/tornet
mkdir -p src

# 2. Check for Data BEFORE touching DVC
echo "🗄️ Checking dataset status..."
if [ -z "$(ls -A data/raw/tornet)" ]; then
    echo "❌ ERROR: The dataset is missing from data/raw/tornet!"
    echo "Please download the data."
    exit 1 # This stops the script from continuing and ruining your DVC file!
else
    echo "✅ Dataset found locally."
fi

# 3. Start the MLflow Server via Docker Compose
echo "📦 Spinning up MLflow Tracking Server..."
docker compose up -d

# 4. Setup Python Virtual Environment
echo "🐍 Configuring Python environment..."
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt -q

# 5. Automate DVC Configuration safely
echo "🔄 Syncing with local DVC remote..."
dvc remote list | grep -q "localremote" || dvc remote add -d localremote ../emi_remote

# Re-add the data folder to sync OS differences, then push to the remote
dvc add data/raw/tornet
dvc push

# 6. Run the Pipeline
echo "🏃‍♂️ Executing Data Ingestion Pipeline..."
python src/data_ingestion.py

echo "✅ Setup Complete! Check MLflow at http://127.0.0.1:5000"