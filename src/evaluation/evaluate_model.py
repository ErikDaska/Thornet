import logging
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import hydra
from omegaconf import DictConfig
import mlflow
import mlflow.pytorch
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc
import tempfile
import os

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- 1. DATASET & MODEL DEFINITIONS ---
# (Required in the namespace so MLflow can load the PyTorch model properly)
class TornetDataset(Dataset):
    def __init__(self, data_dir: Path):
        self.files = sorted(list(data_dir.rglob("*.pt")))
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx])
        if isinstance(data, torch.Tensor):
            features = data
            label = torch.tensor(0.0) # Dummy fallback
        else:
            features = data['features']
            label = torch.tensor(float(data['label']))
        return features, label

class Tornet3DCNN(torch.nn.Module):
    def __init__(self, in_channels=6):
        super(Tornet3DCNN, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d(kernel_size=2, stride=2),
            torch.nn.Conv3d(16, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d(kernel_size=2, stride=2),
        )
        self.global_pool = torch.nn.AdaptiveAvgPool3d(1)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(16, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# --- MAIN EVALUATION PIPELINE ---
@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def evaluate(cfg: DictConfig):
    logger.info("--- Starting Model Evaluation ---")

    # Set up MLflow
    mlflow.set_tracking_uri(cfg.tracking.uri)
    experiment_name = cfg.tracking.experiment_name
    mlflow.set_experiment(experiment_name)

    # 1. Find the latest run in the experiment that has a model logged
    experiment = mlflow.get_experiment_by_name(experiment_name)
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )
    
    if runs.empty:
        logger.error("No MLflow runs found. Please train the model first.")
        return

    latest_run_id = runs.iloc[0].run_id
    logger.info(f"Loading model from run ID: {latest_run_id}")

    # 2. Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_uri = f"runs:/{latest_run_id}/tornet_3dcnn_model"
    
    try:
        model = mlflow.pytorch.load_model(model_uri)
        model.to(device)
        model.eval()
    except Exception as e:
        logger.error(f"Failed to load model from MLflow: {e}")
        return

    # 3. Load Data
    processed_data_path = Path(cfg.paths.processed_data_dir) / str(cfg.api.dataset.target_year)
    if not processed_data_path.exists():
        logger.error(f"Processed data not found at {processed_data_path}.")
        return

    dataset = TornetDataset(processed_data_path)
    dataloader = DataLoader(dataset, batch_size=cfg.training.get("batch_size", 16), shuffle=False)

    # 4. Run Inference
    logger.info("Running inference on evaluation dataset...")
    y_true = []
    y_scores = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # Apply sigmoid to get probabilities for the positive class (Tornado)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            
            y_scores.extend(probs)
            y_true.extend(labels.numpy())

    if not y_true:
        logger.warning("No data found for evaluation.")
        return

    # 5. Compute Metrics (ROC & AUC)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    logger.info(f"Calculated ROC AUC: {roc_auc:.4f}")

    # Resume the MLflow run to attach evaluation metrics and artifacts to the SAME run
    with mlflow.start_run(run_id=latest_run_id):
        
        mlflow.log_metric("eval_roc_auc", roc_auc)

        # 6. Create Plotly Graph
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', 
                                 name=f'ROC Curve (AUC = {roc_auc:.3f})',
                                 line=dict(color='darkorange', width=2)))
        # Diagonal reference line
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                                 name='Random Classifier',
                                 line=dict(color='navy', width=2, dash='dash')))
        
        fig.update_layout(
            title='ROC Curve - 3D CNN Tornado Detection',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            xaxis=dict(range=[0.0, 1.0]),
            yaxis=dict(range=[0.0, 1.05]),
            template='plotly_white'
        )

        # 7. Save Plotly figure to HTML and log to MLflow
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_path = os.path.join(tmpdir, "roc_curve.html")
            fig.write_html(plot_path)
            
            logger.info("Logging Plotly HTML artifact to MLflow...")
            mlflow.log_artifact(plot_path, artifact_path="evaluation_plots")

    logger.info("Evaluation complete. Results and interactive plots are in MLflow!")

if __name__ == "__main__":
    evaluate()