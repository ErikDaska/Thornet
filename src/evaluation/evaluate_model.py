import logging
import os
import random
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, accuracy_score
)
from torch.utils.data import DataLoader, Subset
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

def get_balanced_samples(pool_a, pool_b, target_a=5, target_b=5):
    """Helper to try to get 5 of A and 5 of B. If one is short, fill with the other."""
    samp_a = random.sample(pool_a, min(len(pool_a), target_a))
    samp_b = random.sample(pool_b, min(len(pool_b), target_b))

    # Fill gaps if pool A was short
    short_a = target_a - len(samp_a)
    if short_a > 0:
        extra_b = [x for x in pool_b if x not in samp_b]
        samp_b += random.sample(extra_b, min(len(extra_b), short_a))

    # Fill gaps if pool B was short
    short_b = target_b - len(samp_b)
    if short_b > 0:
        extra_a = [x for x in pool_a if x not in samp_a]
        samp_a += random.sample(extra_a, min(len(extra_a), short_b))

    return samp_a + samp_b


def plot_image_grid(dataset, indices, title, filename):
    """Plots a 2x5 grid of the DBZ channel for the given indices with auto-scaling."""
    if not indices:
        logger.warning(f"No indices provided for {title}. Skipping plot.")
        return

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))  # Made slightly wider to fit colorbars
    fig.suptitle(title, fontsize=18, fontweight='bold')
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        if i >= 10: break  # Max 10 images

        # Load the specific scan
        features, label = dataset[idx]

        # Extract Channel 0 (DBZ Reflectivity), Sweep 0
        dbz_slice = features[0, 0, :, :].numpy()

        ax = axes[i]

        # We removed vmin/vmax to allow auto-scaling.
        # Swapped to 'turbo' (or 'viridis') which is much better for analyzing fluid dynamics/radar
        im = ax.imshow(dbz_slice, cmap='turbo', aspect='auto')
        ax.set_title(f"True Label: {int(label.item())}")
        ax.axis('off')

        # Add a tiny colorbar to each image so you can read the actual dBZ values
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

    # Hide any unused subplots if we had fewer than 10 samples
    for j in range(i + 1, 10):
        axes[j].axis('off')

    plt.tight_layout()
    mlflow.log_figure(fig, filename)
    plt.close(fig)

# --- MAIN EVALUATION PIPELINE ---
@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def evaluate(cfg: DictConfig):
    logger.info(f"--- Starting {cfg.model.name} Model Evaluation ---")

    # Set up MLflow
    mlflow.set_tracking_uri(cfg.tracking.uri)
    experiment_name = cfg.tracking.experiment_name
    mlflow.set_experiment(experiment_name)

    # Find the latest run in the experiment that has a model logged
    logger.info("Searching for the latest successful training run...")
    runs = mlflow.search_runs(
        experiment_names=[cfg.tracking.experiment_name],
        filter_string="tags.mlflow.runName LIKE '3dcnn_training_%'",
        order_by=["start_time DESC"],
        max_results=1
    )
    
    if runs.empty:
        logger.error("No MLflow runs found. Please train the model first.")
        return

    latest_run_id = runs.iloc[0].run_id
    logger.info(f"Loading model from run ID: {latest_run_id}")

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_uri = f"runs:/{latest_run_id}/model"
    model = mlflow.pytorch.load_model(model_uri).to(device)
    model.eval()

    # Load Test Dataset
    processed_data_path = Path(cfg.paths.processed_data_dir) / str(cfg.api.dataset.target_year)
    catalog_path = Path(cfg.api.dataset.raw_path) / "catalog.csv"
    test_indices_path = Path(cfg.paths.processed_data_dir) / f"test_indices_{cfg.api.dataset.target_year}.csv"

    if not test_indices_path.exists():
        logger.error(f"Test indices not found at {test_indices_path}! Cannot guarantee clean evaluation.")
        return

    logger.info("Loading full dataset and applying test splits...")
    full_dataset = TornetDataset(data_dir=processed_data_path, catalog_path=catalog_path)

    test_indices = pd.read_csv(test_indices_path)['test_index'].tolist()
    test_dataset = Subset(full_dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    # RUN EVALUATION LOOP
    all_preds, all_labels, all_probs = [], [], []

    logger.info("Running test set through the model...")
    y_true = []
    y_scores = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            
            y_scores.extend(probs)
            y_true.extend(labels.numpy())

    if not y_true:
        logger.warning("No data found for evaluation.")
        return
    
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    # CALCULATE METRICS ---
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)


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

        # PLOT 1: Data Distribution (Tornado vs No Tornado)
        logger.info("Plotting data distribution...")
        fig1, ax1 = plt.subplots(figsize=(6, 6))
        class_counts = pd.Series(y_true).value_counts()
        labels_map = {0.0: "No Tornado", 1.0: "Tornado"}
        ax1.pie(class_counts, labels=[labels_map.get(k, k) for k in class_counts.index],
                autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#ff9999'])
        ax1.set_title("Test Set Data Distribution")
        mlflow.log_figure(fig1, "data_distribution.png")
        plt.close(fig1)
        
        mlflow.log_metric("eval_roc_auc", roc_auc)

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

    logger.info("Evaluation complete. Results and interactive plots are in MLflow!")

if __name__ == "__main__":
    evaluate()