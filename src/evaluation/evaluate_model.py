import logging
import os
import random
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, roc_curve, auc, precision_recall_curve,
    average_precision_score, classification_report, confusion_matrix
)
from torch.utils.data import DataLoader, Subset

import hydra
from omegaconf import DictConfig
import mlflow
import mlflow.pytorch
import plotly.graph_objects as go
import plotly.figure_factory as ff

from datasets.tornet_dataset import TornetDataset
from training.models import get_model # Ensures models are in context for MLflow

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from models.GRADCAM3D import GradCAM_Dynamic

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


def plot_image_grid(dataset, indices, title, filename, channel_idx=0, cmap='turbo'):
    """Plots a 2x5 grid of a specific radar channel for the given indices."""
    if not indices:
        logger.warning(f"No indices provided for {title}. Skipping plot.")
        return

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    fig.suptitle(title, fontsize=18, fontweight='bold')
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        if i >= 10: break

        # Load the specific scan
        features, label = dataset[idx]

        # Extract the requested Channel (0=DBZ, 1=VEL), Sweep 0
        data_slice = features[channel_idx, :, :, 0].numpy()

        ax = axes[i]

        # For Velocity (VEL), it is highly recommended to center the colormap at 0.
        # Since the data might be normalized (e.g., to [0,1]), this attempts to auto-scale.
        im = ax.imshow(data_slice, cmap=cmap, aspect='auto')
        ax.set_title(f"True Label: {int(label.item())}")
        ax.axis('off')

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

    for j in range(i + 1, 10):
        axes[j].axis('off')

    plt.tight_layout()
    mlflow.log_figure(fig, filename)
    plt.close(fig)

def plot_gradcam_grid(model, dataset, indices, title, filename, device):
    """Plots a 2x5 grid of Grad-CAM overlays for the given indices."""
    if not indices:
        logger.warning(f"No indices provided for {title}. Skipping Grad-CAM plot.")
        return

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    fig.suptitle(title, fontsize=18, fontweight='bold')
    axes = axes.flatten()

    # Initialize our custom 3D CAM extractor instead of the buggy library version
    cam_extractor = GradCAM_Dynamic(model=model, target_layer=model.block4[-2])

    with torch.set_grad_enabled(True):
        for i, idx in enumerate(indices):
            if i >= 10: break

            # Load the specific scan
            features, label = dataset[idx]
            input_tensor = features.unsqueeze(0).to(device)

            # 1. Generate the dynamic heatmap
            # Update the class name initialization at the top of the function to GradCAM_Dynamic too!
            cam_raw = cam_extractor(input_tensor)

            # 2. Slice the CAM only if it is still 3D. If the model already pooled
            #    the depth away, it's already a 2D map [H, W], so no slicing is needed!
            if cam_raw.dim() == 3:
                cam_slice = cam_raw[:, :, 0]
            else:
                cam_slice = cam_raw

            # 3. Get original image dimensions for resizing
            dbz_slice = features[0, :, :, 0].numpy()
            original_h, original_w = dbz_slice.shape

            # 4. Safely resize the small CAM slice back to original dimensions using PyTorch
            cam_resized = F.interpolate(
                cam_slice.unsqueeze(0).unsqueeze(0), # Add fake batch/channel dims for interpolate
                size=(original_h, original_w),
                mode='bilinear',
                align_corners=False
            ).squeeze().cpu().numpy()

            # Normalize background image to [0, 1] for the overlay utility
            img_normalized = (dbz_slice - np.min(dbz_slice)) / (np.max(dbz_slice) - np.min(dbz_slice) + 1e-8)
            rgb_image = np.stack((img_normalized,) * 3, axis=-1)

            # Overlay heatmap on radar image
            visualization = show_cam_on_image(rgb_image, cam_resized, use_rgb=True)

            ax = axes[i]
            ax.imshow(visualization, aspect='auto')
            ax.set_title(f"True Label: {int(label.item())}")
            ax.axis('off')

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

    # FIXED: Dynamically search for the current model's latest run, not just 3dcnn
    logger.info("Searching for the latest successful training run...")
    runs = mlflow.search_runs(
        experiment_names=[cfg.tracking.experiment_name],
        filter_string=f"tags.mlflow.runName LIKE '{cfg.model.name}_training_%'",
        order_by=["start_time DESC"],
        max_results=1
    )
    
    if runs.empty:
        logger.error(f"No MLflow runs found for {cfg.model.name}. Please train the model first.")
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
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    # --- RUN EVALUATION LOOP ---
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

    # FIXED: Properly convert populated lists to arrays
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    y_pred = (y_scores > 0.5).astype(int) # Create binary predictions from scores

    # --- CALCULATE METRICS ---
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    logger.info(f"Calculated ROC AUC: {roc_auc:.4f}")

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)
    logger.info(f"Calculated Average Precision: {avg_precision:.4f}")

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    logger.info(f"Classification Report:\n{classification_report(y_true, y_pred, zero_division=0)}")

    cm = confusion_matrix(y_true, y_pred)
    logger.info(f"Confusion Matrix:\n{cm}")

    # Resume the MLflow run to attach evaluation metrics and artifacts
    with mlflow.start_run(run_id=latest_run_id):

        mlflow.log_metric("eval_roc_auc", roc_auc)
        mlflow.log_metric("eval_avg_precision", avg_precision)
        mlflow.log_metric("eval_precision", report['weighted avg']['precision'])
        mlflow.log_metric("eval_recall", report['weighted avg']['recall'])
        mlflow.log_metric("eval_f1_score", report['weighted avg']['f1-score'])
        mlflow.log_metric("eval_accuracy", acc)

        # --- PLOTLY GRAPHS ---
        logger.info("Generating and logging Plotly evaluation graphs...")
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Data Distribution (Converted to Plotly)
        class_counts = pd.Series(y_true).value_counts()
        labels_map = {0.0: "No Tornado", 1.0: "Tornado"}
        pie_labels = [labels_map.get(k, k) for k in class_counts.index]

        fig_pie = go.Figure(data=[go.Pie(labels=pie_labels, values=class_counts.values, hole=.3, marker_colors=['#66b3ff', '#ff9999'])])
        fig_pie.update_layout(title="Test Set Data Distribution", template='plotly_white')
        mlflow.log_figure(fig_pie, f"data_distribution_{cfg.model.name}_{current_time}.html")

        # 2. ROC Curve
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                     name=f'ROC Curve (AUC = {roc_auc:.3f})',
                                     line=dict(color='darkorange', width=2)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                     name='Random Classifier',
                                     line=dict(color='navy', width=2, dash='dash')))
        fig_roc.update_layout(
            title=f'ROC Curve - {cfg.model.name}',
            xaxis_title='False Positive Rate', yaxis_title='True Positive Rate',
            xaxis=dict(range=[0.0, 1.0]), yaxis=dict(range=[0.0, 1.05]),
            template='plotly_white'
        )
        mlflow.log_figure(fig_roc, f"roc_curve_plot_{cfg.model.name}_{current_time}.html")

        # 3. Precision-Recall Curve
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(x=recall, y=precision, mode='lines',
                                    name=f'PR Curve (AP = {avg_precision:.3f})',
                                    line=dict(color='blue', width=2)))
        fig_pr.update_layout(
            title=f'Precision-Recall Curve - {cfg.model.name}',
            xaxis_title='Recall', yaxis_title='Precision',
            xaxis=dict(range=[0.0, 1.0]), yaxis=dict(range=[0.0, 1.05]),
            template='plotly_white'
        )
        mlflow.log_figure(fig_pr, f"precision_recall_curve_{cfg.model.name}_{current_time}.html")

        # 4. Confusion Matrix
        classes = ['No Tornado', 'Tornado']
        z_cm = cm.tolist()
        fig_cm = ff.create_annotated_heatmap(
            z=z_cm, x=classes, y=classes,
            colorscale='Oranges', showscale=True, annotation_text=z_cm
        )
        fig_cm.update_layout(
            title=f'Confusion Matrix - {cfg.model.name}',
            xaxis_title='Predicted Class', yaxis_title='True Class',
            template='plotly_white'
        )
        mlflow.log_figure(fig_cm, f"confusion_matrix_{cfg.model.name}_{current_time}.html")

        logger.info("Plotting Image Grids...")
        # Plot Reflectivity (DBZ) - Channel 0
        plot_image_grid(test_dataset, correct_samples, "Correctly Classified: Reflectivity (DBZ)",
                        "correct_dbz_grid.png", channel_idx=0, cmap='turbo')
        plot_image_grid(test_dataset, incorrect_samples, "Incorrectly Classified: Reflectivity (DBZ)",
                        "incorrect_dbz_grid.png", channel_idx=0, cmap='turbo')

        # Plot Velocity (VEL) - Channel 1
        plot_image_grid(test_dataset, correct_samples, "Correctly Classified: Velocity (VEL)",
                        "correct_vel_grid.png", channel_idx=1, cmap='RdBu_r')
        plot_image_grid(test_dataset, incorrect_samples, "Incorrectly Classified: Velocity (VEL)",
                        "incorrect_vel_grid.png", channel_idx=1, cmap='RdBu_r')

        logger.info("Plotting Grad-CAM Explanation Grids...")
        plot_gradcam_grid(model, test_dataset, correct_samples, "Grad-CAM: Correct Predictions", "gradcam_correct.png",
                          device)
        plot_gradcam_grid(model, test_dataset, incorrect_samples, "Grad-CAM: Incorrect Predictions",
                          "gradcam_incorrect.png", device)
    logger.info("Evaluation complete. Results and interactive plots are in MLflow!")

if __name__ == "__main__":
    evaluate()