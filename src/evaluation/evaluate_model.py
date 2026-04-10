import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig
import mlflow
import mlflow.pytorch
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, classification_report, confusion_matrix
import tempfile
from datetime import datetime
import os
from datasets.tornet_dataset import TornetDataset

# Import models to ensure they are available for MLflow loading
from training.models import get_model

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- MAIN EVALUATION PIPELINE ---
@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def evaluate(cfg: DictConfig):
    logger.info(f"--- Starting {cfg.model.name} Model Evaluation ---")

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
    model_uri = f"runs:/{latest_run_id}/model"
    
    try:
        model = mlflow.pytorch.load_model(model_uri)
        model.to(device)
        model.eval()
    except Exception as e:
        logger.error(f"Failed to load model from MLflow: {e}")
        return

    # 3. Load Data
    processed_data_path = Path(cfg.paths.processed_data_dir) / str(cfg.api.dataset.target_year)
    catalog_path = Path(cfg.api.dataset.raw_path) / "catalog.csv"
    if not processed_data_path.exists():
        logger.error(f"Processed data not found at {processed_data_path}.")
        return

    dataset = TornetDataset(data_dir=processed_data_path, catalog_path=catalog_path)
    dataloader = DataLoader(dataset, batch_size=cfg.model.batch_size, shuffle=False, num_workers=4, pin_memory=True)

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

    # 5. Compute Metrics (ROC & AUC, Precision-Recall, etc.)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    logger.info(f"Calculated ROC AUC: {roc_auc:.4f}")

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)
    logger.info(f"Calculated Average Precision: {avg_precision:.4f}")

    # Convert probabilities to binary predictions for classification report
    y_pred = [1 if score > 0.5 else 0 for score in y_scores]
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    logger.info(f"Classification Report:\n{classification_report(y_true, y_pred, zero_division=0)}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    logger.info(f"Confusion Matrix:\n{cm}")

    # Resume the MLflow run to attach evaluation metrics and artifacts to the SAME run
    with mlflow.start_run(run_id=latest_run_id):
        
        mlflow.log_metric("eval_roc_auc", roc_auc)
        mlflow.log_metric("eval_avg_precision", avg_precision)
        mlflow.log_metric("eval_precision", report['weighted avg']['precision'])
        mlflow.log_metric("eval_recall", report['weighted avg']['recall'])
        mlflow.log_metric("eval_f1_score", report['weighted avg']['f1-score'])
        mlflow.log_metric("eval_accuracy", report['accuracy'])

        # 6. Create Plotly Graphs
        # ROC Curve
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', 
                                     name=f'ROC Curve (AUC = {roc_auc:.3f})',
                                     line=dict(color='darkorange', width=2)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                                     name='Random Classifier',
                                     line=dict(color='navy', width=2, dash='dash')))
        
        fig_roc.update_layout(
            title=f'ROC Curve - {cfg.model.name} Tornado Detection',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            xaxis=dict(range=[0.0, 1.0]),
            yaxis=dict(range=[0.0, 1.05]),
            template='plotly_white'
        )

        # Precision-Recall Curve
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(x=recall, y=precision, mode='lines',
                                    name=f'PR Curve (AP = {avg_precision:.3f})',
                                    line=dict(color='blue', width=2)))
        
        fig_pr.update_layout(
            title=f'Precision-Recall Curve - {cfg.model.name} Tornado Detection',
            xaxis_title='Recall',
            yaxis_title='Precision',
            xaxis=dict(range=[0.0, 1.0]),
            yaxis=dict(range=[0.0, 1.05]),
            template='plotly_white'
        )

        classes = ['No Tornado', 'Tornado']
        z_cm = cm.tolist()
        
        # Create an annotated heatmap (includes values in cells)
        fig_cm = ff.create_annotated_heatmap(
            z=z_cm,
            x=classes,
            y=classes,
            colorscale='Oranges', # Choose visually appropriate scale
            showscale=True,
            annotation_text=z_cm # Display matrix values
        )
        
        fig_cm.update_layout(
            title=f'Confusion Matrix - {cfg.model.name} Tornado Detection',
            xaxis_title='Predicted Class',
            yaxis_title='True Class',
            template='plotly_white'
        )

        '''
        # 7. Save Plotly figures to HTML and log to MLflow
        with tempfile.TemporaryDirectory() as tmpdir:
            roc_plot_path = os.path.join(tmpdir, "roc_curve.html")
            pr_plot_path = os.path.join(tmpdir, "pr_curve.html")
            fig_roc.write_html(roc_plot_path)
            fig_pr.write_html(pr_plot_path)
            
            logger.info("Logging Plotly HTML artifacts to MLflow...")
            mlflow.log_figure(roc_plot_path, artifact_path="evaluation_plots")
            mlflow.log_figure(pr_plot_path, artifact_path="evaluation_plots")
            '''
        
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        mlflow.log_figure(fig_roc, f"roc_curve_plot_{cfg.model.name}_{current_time}.html")
        mlflow.log_figure(fig_pr, f"precision_recall_curve_plot_{cfg.model.name}_{current_time}.html")
        mlflow.log_figure(fig_cm, f"confusion_matrix_{cfg.model.name}_{current_time}.html")

    logger.info("Evaluation complete. Results, metrics, and interactive plots are in MLflow!")

if __name__ == "__main__":
    evaluate()