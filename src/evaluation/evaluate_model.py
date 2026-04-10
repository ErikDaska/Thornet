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

import hydra
from omegaconf import DictConfig
import mlflow
import mlflow.pytorch
from torch.utils.data import DataLoader, Subset
from datasets.tornet_dataset import TornetDataset

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
    logger.info("--- Starting Model Evaluation ---")

    # Set up MLflow
    mlflow.set_registry_uri("sqlite:////mlflow/mlflow.db")
    mlflow.set_tracking_uri(cfg.tracking.uri)
    experiment_name = cfg.tracking.experiment_name
    os.environ['MLFLOW_ARTIFACT_ROOT'] = "/mlruns"
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
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds = (probs > 0.5).astype(float)

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy().flatten())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    # CALCULATE METRICS ---
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.0  # Occurs if only one class is present in the test set

    logger.info(f"Test Results -> Acc: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

    # LOGGING AND PLOTTING TO MLFLOW
    run_name = f"evaluation_{cfg.api.dataset.target_year}"
    with mlflow.start_run(run_name=run_name):
        # Log Metrics
        mlflow.log_metrics({
            "test_accuracy": acc,
            "test_precision": prec,
            "test_recall": rec,
            "test_f1": f1,
            "test_auc": auc
        })

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

        # PLOT 2: Confusion Matrix
        logger.info("Plotting confusion matrix...")
        cm = confusion_matrix(y_true, y_pred)
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                    xticklabels=["Predicted No Tor", "Predicted Tor"],
                    yticklabels=["Actual No Tor", "Actual Tor"])
        ax2.set_title("Confusion Matrix")
        mlflow.log_figure(fig2, "confusion_matrix.png")
        plt.close(fig2)

        # PLOT 3 & 4: Image Analysis (Categorize indices)
        logger.info("Identifying images for visual grids...")
        tp_idx, tn_idx, fp_idx, fn_idx = [], [], [], []

        for i in range(len(y_true)):
            if y_true[i] == 1.0 and y_pred[i] == 1.0:
                tp_idx.append(i)
            elif y_true[i] == 0.0 and y_pred[i] == 0.0:
                tn_idx.append(i)
            elif y_true[i] == 0.0 and y_pred[i] == 1.0:
                fp_idx.append(i)
            elif y_true[i] == 1.0 and y_pred[i] == 0.0:
                fn_idx.append(i)

        # Get samples for grids
        correct_samples = get_balanced_samples(tp_idx, tn_idx, target_a=5, target_b=5)
        incorrect_samples = get_balanced_samples(fp_idx, fn_idx, target_a=5, target_b=5)

        logger.info("Plotting Image Grids...")
        plot_image_grid(test_dataset, correct_samples, "Correctly Classified Scans", "correct_predictions_grid.png")
        plot_image_grid(test_dataset, incorrect_samples, "Incorrectly Classified Scans",
                        "incorrect_predictions_grid.png")

    logger.info("Evaluation complete! Check MLflow UI for metrics and plots.")

if __name__ == "__main__":
    evaluate()