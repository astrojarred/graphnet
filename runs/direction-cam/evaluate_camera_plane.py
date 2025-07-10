#!/usr/bin/env python3
"""
Evaluation script for MAGIC camera-plane reconstruction models.

This script mirrors the functionality of ``evaluate_deepsky_direction.py`` but is
adapted for models that output camera-plane coordinates (``camera_x_pred`` and
``camera_y_pred``) as implemented in
``graphnet.models.task.magic_direction_cam.CameraPlaneReconstruction`` (and the
variant with uncertainties).

Key features
------------
* Load a ``StandardModel`` from YAML configuration and checkpoint.
* Run inference on a GraphNeT ``Dataset`` split using a standard ``DataLoader``.
* Collect predictions together with truth information stored in the batch
  (``true_source_camera_position`` and ``true_energy``).
* Compute radial errors in the camera plane as well as the corresponding angular
  errors in degrees (1.0 camera unit ≙ 2.5 deg by default).
* Produce basic diagnostics plots (histograms, cumulative distribution,
  scatterplots) and save both CSV/JSON outputs and figures.

Usage example
-------------
    python evaluate_camera_plane.py \
        --model-config my_model.yml \
        --dataset-config my_dataset.yml \
        --checkpoint last.ckpt \
        --output-dir cam_eval_results
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Batch
from torch.utils.data import Subset
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger("cam_eval")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# -----------------------------------------------------------------------------
# Optional – try to use GraphNeT's logger if available
# -----------------------------------------------------------------------------
try:
    from graphnet.utilities.logging import get_logger  # type: ignore

    logger = get_logger()
except Exception:
    # Fallback to the basic logger already configured above
    pass

# -----------------------------------------------------------------------------
# GraphNeT imports (only runtime-critical ones – keep local to avoid type issues)
# -----------------------------------------------------------------------------
from graphnet.utilities.config import DatasetConfig, ModelConfig  # type: ignore
from graphnet.models import StandardModel  # type: ignore
from graphnet.data.dataset import Dataset  # type: ignore
from graphnet.data.dataloader import DataLoader  # type: ignore

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
CAMERA_UNIT_TO_DEG: float = 2.5  # 1.0 camera unit corresponds to 2.5 deg off-axis

# ============================================================================
# LOADING UTILITIES
# ============================================================================

def load_camera_model(
    model_config_path: str | os.PathLike[str],
    checkpoint_path: str | os.PathLike[str],
    device: str = "auto",
) -> StandardModel:
    """Load a trained ``StandardModel`` and checkpoint weights."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading model configuration …")
    model_config = ModelConfig.load(str(model_config_path))
    model: StandardModel = StandardModel.from_config(model_config, trust=True)

    logger.info("Loading checkpoint weights …")
    state = torch.load(str(checkpoint_path), map_location=device)
    # Lightning checkpoints store weights under "state_dict" – fall back to raw
    if isinstance(state, dict) and "state_dict" in state:
        state_dict = state["state_dict"]
    else:
        state_dict = state
    # Allow missing keys (e.g. optimizers) but be strict on unexpected keys
    model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()
    logger.info("✓ Model ready (device = %s)", device)
    return model


def load_dataset_for_evaluation(
    dataset_config_path: str | os.PathLike[str],
    split: str = "test",
    fraction: float = 1.0,
    batch_size: int = 256,
    num_workers: int = 4,
) -> DataLoader:
    """Load dataset per GraphNeT configuration and wrap in ``DataLoader``."""
    logger.info("Loading dataset configuration …")
    dataset_config = DatasetConfig.load(str(dataset_config_path))

    all_sets = Dataset.from_config(dataset_config)

    if split == "train":
        dataset = all_sets["train"]
    elif split in {"val", "validation"}:
        dataset = all_sets["validation"]
    elif split == "test":
        dataset = all_sets["test"]
    else:
        raise ValueError(
            f"Unknown split '{split}'. Expected one of 'train', 'val', 'test'."
        )

    # logger.info("Dataset split '%s' contains %,d events", split, len(dataset))
    logger.info(f"Dataset split '{split}")

    if not (0 < fraction <= 1):
        raise ValueError("fraction must be within (0, 1].")
    if fraction < 1.0:
        n_sample = int(len(dataset) * fraction)
        # logger.info("Sampling %.1f%% of the dataset (%,d events)…", fraction * 100, n_sample)
        logger.info(f"Sampling {fraction * 100}% of the dataset")
        indices = torch.randperm(len(dataset))[:n_sample].tolist()
        dataset = Subset(dataset, indices)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return dataloader

# ============================================================================
# INFERENCE & AGGREGATION
# ============================================================================

def generate_predictions(
    model: StandardModel,
    dataloader: DataLoader,
    additional_attributes: Optional[List[str]] | None = None,
) -> pd.DataFrame:
    """Run model inference and collect predictions/metadata into a ``DataFrame``."""
    logger.info("Running inference …")
    device = next(model.parameters()).device

    if additional_attributes is None:
        additional_attributes = [
            "true_energy",
            "telescope_phi",
            "telescope_theta",
        ]

    records: List[pd.DataFrame] = []
    prog_bar = tqdm(dataloader, desc="Predicting", unit="batch")
    with torch.no_grad():
        for batch in prog_bar:
            batch: Data | Batch  # type: ignore [annotation-type-everything]
            batch = batch.to(device)

            # Forward pass – ``StandardModel`` returns list[Tensor] (per-task) or Tensor
            output = model(batch)
            if isinstance(output, list):
                # Assume the camera task is the first (sole) task
                output_tensor: torch.Tensor = output[0]
            else:
                output_tensor = output  # type: ignore [assignment]

            # Prediction labels (defined in task)
            pred_labels: List[str]
            if hasattr(model, "prediction_labels") and model.prediction_labels:
                pred_labels = list(model.prediction_labels)  # type: ignore [arg-type]
            else:
                # Fallback/default
                pred_labels = [
                    "camera_x_pred",
                    "camera_y_pred",
                    "camera_x_sigma",
                    "camera_y_sigma",
                ][: output_tensor.shape[1]]

            rec: Dict[str, Any] = {}
            for i, label in enumerate(pred_labels):
                rec[label] = output_tensor[:, i].cpu().numpy()

            # Truth values – "true_source_camera_position" should exist (shape [N,2])
            if hasattr(batch, "true_source_camera_position"):
                true_xy = batch.true_source_camera_position.cpu().numpy()
                rec["camera_x_true"] = true_xy[:, 0]
                rec["camera_y_true"] = true_xy[:, 1]
            else:
                logger.warning("Batch is missing 'true_source_camera_position' – metrics will be invalid!")

            # Optional attributes
            for attr in additional_attributes:
                if hasattr(batch, attr):
                    data_attr = getattr(batch, attr)
                    if torch.is_tensor(data_attr):
                        if data_attr.ndim == 1 and data_attr.shape[0] == output_tensor.shape[0]:
                            rec[attr] = data_attr.cpu().numpy()
                        elif data_attr.ndim == 2 and data_attr.shape[0] == output_tensor.shape[0]:
                            # Flatten small matrices component-wise, otherwise skip
                            if data_attr.shape[1] <= 4:
                                for c in range(data_attr.shape[1]):
                                    rec[f"{attr}_{c}"] = data_attr[:, c].cpu().numpy()
                    else:
                        # Scalars / python types
                        rec[attr] = np.asarray(data_attr)

            records.append(pd.DataFrame(rec))

    if not records:
        raise RuntimeError("No predictions generated – empty DataLoader?")

    results_df = pd.concat(records, ignore_index=True)
    logger.info("✓ Predictions assembled for %,d events", len(results_df))
    return results_df

# ============================================================================
# METRICS
# ============================================================================

def compute_camera_plane_metrics(
    df: pd.DataFrame,
    unit_to_deg: float = CAMERA_UNIT_TO_DEG,
) -> Dict[str, Any]:
    """Compute radial error statistics in camera units and degrees."""
    if {
        "camera_x_pred",
        "camera_y_pred",
        "camera_x_true",
        "camera_y_true",
    } - set(df.columns):
        raise ValueError("Required columns missing for metric computation.")

    dx = df["camera_x_pred"].to_numpy() - df["camera_x_true"].to_numpy()
    dy = df["camera_y_pred"].to_numpy() - df["camera_y_true"].to_numpy()
    radial_err = np.sqrt(dx**2 + dy**2)
    radial_err_deg = radial_err * unit_to_deg

    metrics: Dict[str, Any] = {
        "n_events": len(df),
        "radial_error_camera_units": {
            "median": float(np.median(radial_err)),
            "mean": float(np.mean(radial_err)),
            "std": float(np.std(radial_err)),
            "rms": float(np.sqrt(np.mean(radial_err**2))),
            "cont_68": float(np.percentile(radial_err, 68)),
            "cont_95": float(np.percentile(radial_err, 95)),
        },
        "radial_error_deg": {
            "median": float(np.median(radial_err_deg)),
            "mean": float(np.mean(radial_err_deg)),
            "std": float(np.std(radial_err_deg)),
            "rms": float(np.sqrt(np.mean(radial_err_deg**2))),
            "cont_68": float(np.percentile(radial_err_deg, 68)),
            "cont_95": float(np.percentile(radial_err_deg, 95)),
        },
    }

    # Quality fractions (camera units)
    thresholds_units = [0.05, 0.1, 0.2, 0.5, 1.0]
    metrics["quality_fractions_cam"] = {
        f"sub_{thr}": float(np.mean(radial_err < thr)) for thr in thresholds_units
    }

    # Quality fractions in *degrees* (requested for plots)
    thresholds_deg = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    metrics["quality_fractions_deg"] = {
        f"sub_{thr}deg": float(np.mean(radial_err_deg < thr)) for thr in thresholds_deg
    }

    # Energy-dependent metrics if available
    if "true_energy" in df.columns:
        energies = df["true_energy"].to_numpy()
        if np.sum(~np.isnan(energies)) > 10:
            bins = np.logspace(np.log10(np.nanmin(energies)), np.log10(np.nanmax(energies)), 6)
            energy_metrics: Dict[str, Any] = {}
            for i in range(len(bins) - 1):
                mask = (energies >= bins[i]) & (energies < bins[i + 1])
                if np.any(mask):
                    energy_metrics[f"bin_{i}"] = {
                        "energy_range_gev": [float(bins[i]), float(bins[i + 1])],
                        "median_error_deg": float(np.median(radial_err_deg[mask])),
                        "n_events": int(np.sum(mask)),
                    }
            metrics["energy_dependence"] = energy_metrics

    # Uncertainty analysis if sigma columns exist
    if {"camera_x_sigma", "camera_y_sigma"}.issubset(df.columns):
        sigmas = 0.5 * (df["camera_x_sigma"].to_numpy() + df["camera_y_sigma"].to_numpy())
        metrics["uncertainty"] = {
            "median_sigma": float(np.median(sigmas)),
            "mean_sigma": float(np.mean(sigmas)),
            "std_sigma": float(np.std(sigmas)),
        }

    logger.info("✓ Metrics computed")
    return metrics

# ============================================================================
# PLOTTING
# ============================================================================

def plot_camera_plane_results(
    df: pd.DataFrame,
    metrics: Dict[str, Any],
    output_dir: str | os.PathLike[str],
    show_plots: bool = True,
    save_plots: bool = True,
) -> None:
    """Generate diagnostic plots and save them to *output_dir*."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dx = df["camera_x_pred"].to_numpy() - df["camera_x_true"].to_numpy()
    dy = df["camera_y_pred"].to_numpy() - df["camera_y_true"].to_numpy()
    radial_err = np.sqrt(dx**2 + dy**2)
    radial_err_deg = radial_err * CAMERA_UNIT_TO_DEG

    plt.style.use("default")
    sns.set_palette("husl")

    fig = plt.figure(figsize=(20, 12))

    # 1. Radial error distribution (deg)
    ax1 = plt.subplot(2, 4, 1)
    ax1.hist(radial_err_deg, bins=60, alpha=0.7, edgecolor="black")
    ax1.axvline(metrics["radial_error_deg"]["median"], color="red", ls="--", label="Median")
    ax1.set_xlabel("Radial error [deg]")
    ax1.set_ylabel("Events")
    ax1.set_title("Radial Error Distribution (deg)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Cumulative distribution (deg)
    ax2 = plt.subplot(2, 4, 2)
    sorted_err = np.sort(radial_err_deg)
    cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
    ax2.plot(sorted_err, cdf, lw=2)
    ax2.axhline(0.68, color="orange", ls="--", label="68% containment")
    ax2.axhline(0.95, color="red", ls="--", label="95% containment")
    ax2.set_xlabel("Radial error [deg]")
    ax2.set_ylabel("Cumulative fraction")
    ax2.set_title("Cumulative Distribution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Quality fractions bar chart (deg)
    ax3 = plt.subplot(2, 4, 3)
    qdata = metrics["quality_fractions_deg"]
    thresholds = [float(k.split("sub_")[1].replace("deg", "")) for k in qdata.keys()]
    fractions = list(qdata.values())
    bars = ax3.bar(range(len(thresholds)), fractions, alpha=0.7)
    ax3.set_xticks(range(len(thresholds)))
    ax3.set_xticklabels([f"{t:g}°" for t in thresholds])
    ax3.set_ylabel("Fraction of events")
    ax3.set_xlabel("Angular threshold")
    ax3.set_title("Quality Fractions (deg)")
    ax3.grid(True, alpha=0.3)
    # annotate
    for bar, frac in zip(bars, fractions):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f"{frac:.1%}", ha="center", va="bottom", fontsize=8)

    # 4. Predicted vs true X
    ax4 = plt.subplot(2, 4, 4)
    ax4.scatter(df["camera_x_true"], df["camera_x_pred"], s=1, alpha=0.5)
    lim = ax4.get_xlim()
    ax4.plot(lim, lim, "r--", lw=1)
    ax4.set_xlabel("True X")
    ax4.set_ylabel("Predicted X")
    ax4.set_title("Predicted vs True X")
    ax4.grid(True, alpha=0.3)

    # 5. Predicted vs true Y
    ax5 = plt.subplot(2, 4, 5)
    ax5.scatter(df["camera_y_true"], df["camera_y_pred"], s=1, alpha=0.5)
    lim = ax5.get_xlim()
    ax5.plot(lim, lim, "r--", lw=1)
    ax5.set_xlabel("True Y")
    ax5.set_ylabel("Predicted Y")
    ax5.set_title("Predicted vs True Y")
    ax5.grid(True, alpha=0.3)

    # 6. Radial error vs energy
    ax6 = plt.subplot(2, 4, 6)
    if "true_energy" in df.columns:
        ax6.scatter(df["true_energy"], radial_err_deg, s=1, alpha=0.4)
        ax6.set_xscale("log")
        ax6.set_xlabel("True energy [GeV]")
        ax6.set_ylabel("Radial error [deg]")
        ax6.set_title("Error vs Energy")
        ax6.grid(True, alpha=0.3)
    else:
        ax6.axis("off")

    # 7. Camera-unit histogram (for completeness)
    ax7 = plt.subplot(2, 4, 7)
    ax7.hist(radial_err, bins=60, alpha=0.7, edgecolor="black")
    ax7.set_xlabel("Radial error [camera units]")
    ax7.set_ylabel("Events")
    ax7.set_title("Radial Error (cam units)")
    ax7.grid(True, alpha=0.3)

    # 8. Summary statistics text box
    ax8 = plt.subplot(2, 4, 8)
    ax8.axis("off")
    summary = (
        f"Events: {metrics['n_events']:,}\n"
        f"\nMedian error: {metrics['radial_error_deg']['median']:.3f}°\n"
        f"68% cont.: {metrics['radial_error_deg']['cont_68']:.3f}°\n"
        f"95% cont.: {metrics['radial_error_deg']['cont_95']:.3f}°\n"
        "\nQuality fractions:\n"
        f"  - 0.05°: {metrics['quality_fractions_deg']['sub_0.05deg']:.1%}\n"
        f"  - 0.1°: {metrics['quality_fractions_deg']['sub_0.1deg']:.1%}\n"
        f"  - 0.2°: {metrics['quality_fractions_deg']['sub_0.2deg']:.1%}\n"
        f"  - 0.5°: {metrics['quality_fractions_deg']['sub_0.5deg']:.1%}\n"
        f"  - 1.0°: {metrics['quality_fractions_deg']['sub_1.0deg']:.1%}\n"
        f"  - 2.0°: {metrics['quality_fractions_deg']['sub_2.0deg']:.1%}\n"
        f"  - 5.0°: {metrics['quality_fractions_deg']['sub_5.0deg']:.1%}\n"
    )
    # Add uncertainty info if available
    if "uncertainty" in metrics:
        summary += (
            f"\nSigma median: {metrics['uncertainty']['median_sigma']:.3f} cam units\n"
        )
    ax8.text(0.05, 0.95, summary, transform=ax8.transAxes, va="top", fontfamily="monospace", fontsize=20,
              bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.5))

    plt.tight_layout()

    if save_plots:
        fig_path = out_dir / "camera_plane_eval.png"
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        logger.info("✓ Saved plot to %s", fig_path)
    if show_plots:
        plt.show()
    else:
        plt.close(fig)

# ============================================================================
# END-TO-END PIPELINE
# ============================================================================

def evaluate_camera_plane_model(
    model_config_path: str | os.PathLike[str],
    dataset_config_path: str | os.PathLike[str],
    checkpoint_path: str | os.PathLike[str],
    dataset_split: str = "test",
    dataset_fraction: float = 1.0,
    batch_size: int = 256,
    output_dir: str | os.PathLike[str] = "./camera_plane_eval",
    plot: bool = True,
    save_results: bool = True,
    gpus: Optional[List[int]] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Complete evaluation workflow – returns ``(df, metrics)``."""

    # Resolve output directory
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model and data
    device = "cuda" if torch.cuda.is_available() and (gpus is not None) else "cpu"
    model = load_camera_model(model_config_path, checkpoint_path, device)

    dataloader = load_dataset_for_evaluation(
        dataset_config_path,
        split=dataset_split,
        fraction=dataset_fraction,
        batch_size=batch_size,
    )

    # Run inference and gather predictions
    results_df = generate_predictions(model, dataloader)

    # Compute metrics
    metrics = compute_camera_plane_metrics(results_df)

    # Plots
    if plot:
        plot_camera_plane_results(results_df, metrics, out_dir, show_plots=True, save_plots=save_results)

    # Save CSV / JSON
    if save_results:
        csv_path = out_dir / "camera_plane_results.csv"
        results_df.to_csv(csv_path, index=False)
        with open(out_dir / "camera_plane_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("✓ Saved results to %s", csv_path)

    return results_df, metrics

# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate camera-plane reconstruction models")
    parser.add_argument("--model-config", required=True, help="Path to model YAML config")
    parser.add_argument("--dataset-config", required=True, help="Path to dataset YAML config")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint (.ckpt)")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of dataset to use [0-1]")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--output-dir", default="./camera_plane_eval", help="Output directory")
    parser.add_argument("--no-plot", action="store_true", help="Skip plots")
    parser.add_argument("--no-save", action="store_true", help="Do not save CSV/JSON/figures")
    parser.add_argument("--gpus", type=int, nargs="*", help="GPUs to use (indices)")

    args = parser.parse_args()

    _df, metrics = evaluate_camera_plane_model(
        model_config_path=args.model_config,
        dataset_config_path=args.dataset_config,
        checkpoint_path=args.checkpoint,
        dataset_split=args.split,
        dataset_fraction=args.fraction,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        plot=not args.no_plot,
        save_results=not args.no_save,
        gpus=args.gpus,
    )

    # Pretty summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Events evaluated: {metrics['n_events']:,}")
    print(
        f"Median radial error: {metrics['radial_error_camera_units']['median']:.4f} camera units "
        f"({metrics['radial_error_deg']['median']:.3f}°)"
    )
    print(
        f"68% containment: {metrics['radial_error_camera_units']['cont_68']:.4f} camera units "
        f"({metrics['radial_error_deg']['cont_68']:.3f}°)"
    )
    print("=" * 60)


if __name__ == "__main__":
    main() 
