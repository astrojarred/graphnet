#!/usr/bin/env python3
"""
Evaluation script for MAGIC energy reconstruction models.

This script evaluates energy reconstruction models trained with
``graphnet.models.task.magic_energy_reco.MAGICEnergyReconstruction``.

Key features
------------
* Load a ``StandardModel`` from YAML configuration and checkpoint.
* Run inference on a GraphNeT ``Dataset`` split using a standard ``DataLoader``.
* Collect predictions together with truth information stored in the batch
  (``true_energy``).
* Compute energy resolution and bias as a function of true energy.
* Produce diagnostic plots:
  - Energy resolution and bias vs E_true
  - Correlation plot of E_true vs E_reco
  - Error distributions
  - Energy-dependent metrics

Usage example
-------------
    python evaluate_energy_reco.py \
        --model-config my_model.yml \
        --dataset-config my_dataset.yml \
        --checkpoint last.ckpt \
        --output-dir energy_eval_results
"""
from __future__ import annotations

import argparse
import json
import logging
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
logger = logging.getLogger("energy_eval")
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

# ============================================================================
# LOADING UTILITIES
# ============================================================================

def load_energy_model(
    model_config_path: str | os.PathLike[str],
    checkpoint_path: str | os.PathLike[str],
    device: str = "auto",
) -> StandardModel:
    """Load a trained ``StandardModel`` and checkpoint weights."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading model configuration …")
    config_path = Path(model_config_path).resolve()
    logger.info(f"Model config path: {config_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Model config file not found: {config_path}")
    if config_path.suffix.lower() not in (".yml", ".yaml"):
        raise ValueError(
            f"Model config file must have .yml or .yaml extension. Got: '{config_path.suffix}' "
            f"(full path: {config_path})"
        )
    model_config = ModelConfig.load(str(config_path))
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
    config_path = Path(dataset_config_path).resolve()
    logger.info(f"Dataset config path: {config_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Dataset config file not found: {config_path}")
    if config_path.suffix.lower() not in (".yml", ".yaml"):
        raise ValueError(
            f"Dataset config file must have .yml or .yaml extension. Got: '{config_path.suffix}' "
            f"(full path: {config_path})"
        )
    dataset_config = DatasetConfig.load(str(config_path))

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

    logger.info(f"Dataset split '{split}' contains {len(dataset):,} events")

    if not (0 < fraction <= 1):
        raise ValueError("fraction must be within (0, 1].")
    if fraction < 1.0:
        n_sample = int(len(dataset) * fraction)
        logger.info(f"Sampling {fraction * 100:.1f}% of the dataset ({n_sample:,} events)…")
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
    energy_in_log10: bool = True,
    device: Optional[str | torch.device] = None,
) -> pd.DataFrame:
    """Run model inference and collect predictions/metadata into a ``DataFrame``."""
    logger.info("Running inference …")
    if device is None:
        device = next(model.parameters()).device
    else:
        # Ensure device is a torch.device
        if isinstance(device, str):
            device = torch.device(device)

    if additional_attributes is None:
        additional_attributes = [
            "event_id",
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
                # Assume the energy task is the first (sole) task
                output_tensor: torch.Tensor = output[0]
            else:
                output_tensor = output  # type: ignore [assignment]

            # Prediction labels (defined in task)
            pred_labels: List[str]
            if hasattr(model, "prediction_labels") and model.prediction_labels:
                pred_labels = list(model.prediction_labels)  # type: ignore [arg-type]
            else:
                # Fallback/default
                pred_labels = ["energy_pred"]

            rec: Dict[str, Any] = {}
            for i, label in enumerate(pred_labels):
                pred_values = output_tensor[:, i].cpu().numpy()
                
                # Convert from log10 space to linear space if needed
                if energy_in_log10 and label == "energy_pred":
                    pred_values = np.power(10.0, pred_values)
                    rec[label] = pred_values
                else:
                    rec[label] = pred_values

            # Truth values – "true_energy" should exist
            if hasattr(batch, "true_energy"):
                true_energy = batch.true_energy.cpu().numpy()
                rec["true_energy"] = true_energy
            else:
                logger.warning("Batch is missing 'true_energy' – metrics will be invalid!")

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
    logger.info(f"✓ Predictions assembled for {len(results_df):,} events")
    return results_df

# ============================================================================
# METRICS
# ============================================================================

def compute_energy_metrics(
    df: pd.DataFrame,
    energy_pred_key: str = "energy_pred",
    energy_true_key: str = "true_energy",
    energy_unit: str = "GeV",
) -> Dict[str, Any]:
    """Compute energy resolution and bias statistics."""
    if energy_pred_key not in df.columns or energy_true_key not in df.columns:
        raise ValueError(f"Required columns missing: need '{energy_pred_key}' and '{energy_true_key}'")

    pred_energy = df[energy_pred_key].to_numpy()
    true_energy = df[energy_true_key].to_numpy()

    # Remove any invalid values
    valid_mask = np.isfinite(pred_energy) & np.isfinite(true_energy) & (true_energy > 0) & (pred_energy > 0)
    pred_energy = pred_energy[valid_mask]
    true_energy = true_energy[valid_mask]

    if len(pred_energy) == 0:
        raise ValueError("No valid energy values found after filtering")

    # Compute relative errors
    relative_error = (pred_energy - true_energy) / true_energy
    abs_relative_error = np.abs(relative_error)

    # Energy resolution (68% containment of relative error)
    resolution_68 = np.percentile(abs_relative_error, 68)
    resolution_95 = np.percentile(abs_relative_error, 95)

    # Energy bias (mean relative error)
    bias = np.mean(relative_error)

    metrics: Dict[str, Any] = {
        "n_events": len(pred_energy),
        "n_events_valid": int(np.sum(valid_mask)),
        "energy_resolution": {
            "resolution_68": float(resolution_68),
            "resolution_95": float(resolution_95),
            "median_abs_error": float(np.median(abs_relative_error)),
            "mean_abs_error": float(np.mean(abs_relative_error)),
            "std_error": float(np.std(relative_error)),
        },
        "energy_bias": {
            "mean_bias": float(bias),
            "median_bias": float(np.median(relative_error)),
        },
        "energy_range": {
            "min_true_energy": float(np.min(true_energy)),
            "max_true_energy": float(np.max(true_energy)),
            "min_pred_energy": float(np.min(pred_energy)),
            "max_pred_energy": float(np.max(pred_energy)),
        },
    }

    # Energy-dependent metrics if enough events
    if len(pred_energy) > 100:
        # Use log-spaced bins
        min_e = np.min(true_energy)
        max_e = np.max(true_energy)
        n_bins = min(10, int(np.log10(max_e / min_e) * 2))
        if n_bins >= 2:
            bins = np.logspace(np.log10(min_e), np.log10(max_e), n_bins + 1)
            energy_metrics: Dict[str, Any] = {}
            for i in range(len(bins) - 1):
                mask = (true_energy >= bins[i]) & (true_energy < bins[i + 1])
                if np.sum(mask) > 10:  # Need at least 10 events per bin
                    bin_pred = pred_energy[mask]
                    bin_true = true_energy[mask]
                    bin_rel_error = (bin_pred - bin_true) / bin_true
                    energy_metrics[f"bin_{i}"] = {
                        "energy_range": [float(bins[i]), float(bins[i + 1])],
                        "n_events": int(np.sum(mask)),
                        "resolution_68": float(np.percentile(np.abs(bin_rel_error), 68)),
                        "bias": float(np.mean(bin_rel_error)),
                    }
            metrics["energy_dependence"] = energy_metrics

    logger.info("✓ Metrics computed")
    return metrics

# ============================================================================
# PLOTTING
# ============================================================================

def plot_energy_results(
    df: pd.DataFrame,
    metrics: Dict[str, Any],
    output_dir: str | os.PathLike[str],
    show_plots: bool = True,
    save_plots: bool = True,
    energy_pred_key: str = "energy_pred",
    energy_true_key: str = "true_energy",
    energy_unit: str = "GeV",
) -> None:
    """Generate diagnostic plots and save them to *output_dir*."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_energy = df[energy_pred_key].to_numpy()
    true_energy = df[energy_true_key].to_numpy()

    # Remove invalid values
    valid_mask = np.isfinite(pred_energy) & np.isfinite(true_energy) & (true_energy > 0) & (pred_energy > 0)
    pred_energy = pred_energy[valid_mask]
    true_energy = true_energy[valid_mask]

    relative_error = (pred_energy - true_energy) / true_energy
    abs_relative_error = np.abs(relative_error)

    plt.style.use("default")
    sns.set_palette("husl")

    fig = plt.figure(figsize=(20, 12))

    # 1. Energy resolution and bias vs E_true (main plot)
    ax1 = plt.subplot(2, 4, 1)
    
    # Bin data for resolution and bias
    min_e = np.min(true_energy)
    max_e = np.max(true_energy)
    n_bins = 20
    bins = np.logspace(np.log10(min_e), np.log10(max_e), n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    
    resolution_68 = []
    resolution_95 = []
    bias_mean = []
    bias_median = []
    n_per_bin = []
    
    for i in range(len(bins) - 1):
        mask = (true_energy >= bins[i]) & (true_energy < bins[i + 1])
        if np.sum(mask) > 5:
            bin_rel_error = relative_error[mask]
            resolution_68.append(np.percentile(np.abs(bin_rel_error), 68))
            resolution_95.append(np.percentile(np.abs(bin_rel_error), 95))
            bias_mean.append(np.mean(bin_rel_error))
            bias_median.append(np.median(bin_rel_error))
            n_per_bin.append(np.sum(mask))
        else:
            resolution_68.append(np.nan)
            resolution_95.append(np.nan)
            bias_mean.append(np.nan)
            bias_median.append(np.nan)
            n_per_bin.append(0)
    
    resolution_68 = np.array(resolution_68)
    resolution_95 = np.array(resolution_95)
    bias_mean = np.array(bias_mean)
    bias_median = np.array(bias_median)
    
    valid_bins = ~np.isnan(resolution_68)
    
    ax1_twin = ax1.twinx()
    
    # Plot resolution (solid lines)
    ax1.plot(bin_centers[valid_bins], resolution_68[valid_bins], 'o-', color='blue', label='Resolution (68%)', linewidth=2, markersize=4)
    ax1.plot(bin_centers[valid_bins], resolution_95[valid_bins], 's-', color='blue', alpha=0.6, label='Resolution (95%)', linewidth=2, markersize=4)
    
    # Plot bias (dashed lines)
    ax1_twin.plot(bin_centers[valid_bins], bias_mean[valid_bins], '--', color='red', label='Bias (mean)', linewidth=2)
    ax1_twin.plot(bin_centers[valid_bins], bias_median[valid_bins], '--', color='orange', alpha=0.7, label='Bias (median)', linewidth=2)
    ax1_twin.axhline(0, color='black', linestyle=':', linewidth=1, alpha=0.5)
    
    ax1.set_xscale('log')
    ax1.set_xlabel(f'E_true [{energy_unit}]', fontsize=12)
    ax1.set_ylabel('Energy Resolution (|ΔE/E|)', fontsize=12, color='blue')
    ax1_twin.set_ylabel('Energy Bias (ΔE/E)', fontsize=12, color='red')
    ax1.set_title('Energy Resolution and Bias vs E_true', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=9)
    ax1_twin.legend(loc='upper right', fontsize=9)
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1_twin.tick_params(axis='y', labelcolor='red')

    # 2. Correlation plot: E_true vs E_reco
    ax2 = plt.subplot(2, 4, 2)
    
    # Use 2D histogram for better visualization with many points
    # Create log-spaced bins for both axes
    min_e = min(np.min(true_energy), np.min(pred_energy))
    max_e = max(np.max(true_energy), np.max(pred_energy))
    n_bins_2d = 50
    bins_2d = np.logspace(np.log10(min_e), np.log10(max_e), n_bins_2d + 1)
    
    # Create 2D histogram
    H, xedges, yedges = np.histogram2d(
        true_energy, pred_energy, bins=[bins_2d, bins_2d]
    )
    
    # Mask zeros for better visualization
    H_masked = np.ma.masked_where(H == 0, H)
    
    # Use pcolormesh for proper log-scale display
    X, Y = np.meshgrid(xedges, yedges)
    im = ax2.pcolormesh(
        X, Y, H_masked,
        cmap='viridis',
        norm=plt.colors.LogNorm(vmin=max(1, H[H > 0].min()), vmax=H.max()),
        shading='flat'
    )
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2, pad=0.02)
    cbar.set_label('Counts', fontsize=10)
    
    # Perfect correlation line
    lims = [min_e, max_e]
    ax2.plot(lims, lims, 'r--', lw=2, label='Perfect correlation', alpha=0.8, zorder=10)
    
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel(f'E_true [{energy_unit}]', fontsize=12)
    ax2.set_ylabel(f'E_reco [{energy_unit}]', fontsize=12)
    ax2.set_title('E_true vs E_reco', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9, loc='lower right')
    ax2.grid(True, alpha=0.3, which='both')

    # 3. Relative error distribution
    ax3 = plt.subplot(2, 4, 3)
    ax3.hist(relative_error, bins=60, alpha=0.7, edgecolor="black", color='steelblue')
    ax3.axvline(0, color="red", ls="--", label="Zero bias")
    ax3.axvline(np.median(relative_error), color="orange", ls="--", label=f"Median: {np.median(relative_error):.4f}")
    ax3.set_xlabel("Relative Error (ΔE/E)", fontsize=12)
    ax3.set_ylabel("Events", fontsize=12)
    ax3.set_title("Relative Error Distribution", fontsize=14, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # 4. Absolute relative error distribution
    ax4 = plt.subplot(2, 4, 4)
    ax4.hist(abs_relative_error, bins=60, alpha=0.7, edgecolor="black", color='coral')
    ax4.axvline(metrics["energy_resolution"]["resolution_68"], color="red", ls="--", label=f"68%: {metrics['energy_resolution']['resolution_68']:.4f}")
    ax4.axvline(metrics["energy_resolution"]["resolution_95"], color="darkred", ls="--", label=f"95%: {metrics['energy_resolution']['resolution_95']:.4f}")
    ax4.set_xlabel("|Relative Error| (|ΔE/E|)", fontsize=12)
    ax4.set_ylabel("Events", fontsize=12)
    ax4.set_title("Absolute Relative Error Distribution", fontsize=14, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    # 5. Relative error vs E_true (scatter)
    ax5 = plt.subplot(2, 4, 5)
    if len(relative_error) > 10000:
        sample_idx = np.random.choice(len(relative_error), size=min(5000, len(relative_error)), replace=False)
        ax5.scatter(true_energy[sample_idx], relative_error[sample_idx], s=1, alpha=0.3, c='blue')
    else:
        ax5.scatter(true_energy, relative_error, s=1, alpha=0.3, c='blue')
    ax5.axhline(0, color="red", ls="--", linewidth=1)
    ax5.set_xscale('log')
    ax5.set_xlabel(f'E_true [{energy_unit}]', fontsize=12)
    ax5.set_ylabel("Relative Error (ΔE/E)", fontsize=12)
    ax5.set_title("Relative Error vs E_true", fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # 6. Resolution vs E_true (zoomed)
    ax6 = plt.subplot(2, 4, 6)
    ax6.plot(bin_centers[valid_bins], resolution_68[valid_bins], 'o-', color='blue', label='68% containment', linewidth=2, markersize=4)
    ax6.plot(bin_centers[valid_bins], resolution_95[valid_bins], 's-', color='blue', alpha=0.6, label='95% containment', linewidth=2, markersize=4)
    ax6.set_xscale('log')
    ax6.set_xlabel(f'E_true [{energy_unit}]', fontsize=12)
    ax6.set_ylabel('Energy Resolution (|ΔE/E|)', fontsize=12)
    ax6.set_title('Resolution vs E_true', fontsize=14, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)

    # 7. Bias vs E_true (zoomed)
    ax7 = plt.subplot(2, 4, 7)
    ax7.plot(bin_centers[valid_bins], bias_mean[valid_bins], 'o-', color='red', label='Mean bias', linewidth=2, markersize=4)
    ax7.plot(bin_centers[valid_bins], bias_median[valid_bins], 's-', color='orange', alpha=0.7, label='Median bias', linewidth=2, markersize=4)
    ax7.axhline(0, color='black', linestyle=':', linewidth=1, alpha=0.5)
    ax7.set_xscale('log')
    ax7.set_xlabel(f'E_true [{energy_unit}]', fontsize=12)
    ax7.set_ylabel('Energy Bias (ΔE/E)', fontsize=12)
    ax7.set_title('Bias vs E_true', fontsize=14, fontweight='bold')
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3)

    # 8. Summary statistics text box
    ax8 = plt.subplot(2, 4, 8)
    ax8.axis("off")
    summary = (
        f"Events: {metrics['n_events']:,}\n"
        f"\nResolution (68%): {metrics['energy_resolution']['resolution_68']:.4f}\n"
        f"Resolution (95%): {metrics['energy_resolution']['resolution_95']:.4f}\n"
        f"\nBias (mean): {metrics['energy_bias']['mean_bias']:.4f}\n"
        f"Bias (median): {metrics['energy_bias']['median_bias']:.4f}\n"
        f"\nEnergy range:\n"
        f"  True: {metrics['energy_range']['min_true_energy']:.2e} - {metrics['energy_range']['max_true_energy']:.2e} {energy_unit}\n"
        f"  Pred: {metrics['energy_range']['min_pred_energy']:.2e} - {metrics['energy_range']['max_pred_energy']:.2e} {energy_unit}\n"
    )
    ax8.text(0.05, 0.95, summary, transform=ax8.transAxes, va="top", fontfamily="monospace", fontsize=11,
              bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.5))

    plt.tight_layout()

    if save_plots:
        fig_path = out_dir / "energy_reco_eval.png"
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        logger.info("✓ Saved plot to %s", fig_path)
    if show_plots:
        plt.show()
    else:
        plt.close(fig)

# ============================================================================
# END-TO-END PIPELINE
# ============================================================================

def evaluate_energy_reco_model(
    model_config_path: str | os.PathLike[str],
    dataset_config_path: str | os.PathLike[str],
    checkpoint_path: str | os.PathLike[str],
    dataset_split: str = "test",
    dataset_fraction: float = 1.0,
    batch_size: int = 256,
    output_dir: str | os.PathLike[str] = "./energy_reco_eval",
    plot: bool = True,
    save_results: bool = True,
    gpus: Optional[List[int]] | None = None,
    energy_in_log10: bool = True,
    energy_unit: str = "GeV",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Complete evaluation workflow – returns ``(df, metrics)``."""

    # Resolve output directory
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model and data
    if torch.cuda.is_available() and gpus is not None and len(gpus) > 0:
        # Use the first GPU from the list
        device = f"cuda:{gpus[0]}"
        logger.info(f"Using GPU {gpus[0]}")
    elif torch.cuda.is_available():
        device = "cuda"
        logger.info("Using default CUDA device")
    else:
        device = "cpu"
        logger.info("CUDA not available, using CPU")
    model = load_energy_model(model_config_path, checkpoint_path, device)

    dataloader = load_dataset_for_evaluation(
        dataset_config_path,
        split=dataset_split,
        fraction=dataset_fraction,
        batch_size=batch_size,
    )

    # Run inference and gather predictions
    results_df = generate_predictions(model, dataloader, energy_in_log10=energy_in_log10, device=device)

    # Compute metrics
    metrics = compute_energy_metrics(results_df, energy_unit=energy_unit)

    # Plots
    if plot:
        plot_energy_results(results_df, metrics, out_dir, show_plots=False, save_plots=save_results, energy_unit=energy_unit)

    # Save CSV / JSON
    if save_results:
        csv_path = out_dir / "energy_reco_results.csv"
        results_df.to_csv(csv_path, index=False)
        parquet_path = out_dir / "energy_reco_results.parquet"
        results_df.to_parquet(parquet_path, index=False)
        with open(out_dir / "energy_reco_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("✓ Saved results to %s", csv_path)

    return results_df, metrics

# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate energy reconstruction models")
    parser.add_argument("--model-config", required=True, help="Path to model YAML config")
    parser.add_argument("--dataset-config", required=True, help="Path to dataset YAML config")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint (.ckpt)")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of dataset to use [0-1]")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--output-dir", default="./energy_reco_eval", help="Output directory")
    parser.add_argument("--no-plot", action="store_true", help="Skip plots")
    parser.add_argument("--no-save", action="store_true", help="Do not save CSV/JSON/figures")
    parser.add_argument("--gpus", type=int, nargs="*", help="GPUs to use (indices)")
    parser.add_argument("--energy-in-log10", action="store_true", default=True, help="Predictions are in log10 space (default: True)")
    parser.add_argument("--energy-unit", type=str, default="GeV", help="Energy unit for plots (default: GeV)")

    args = parser.parse_args()

    _df, metrics = evaluate_energy_reco_model(
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
        energy_in_log10=args.energy_in_log10,
        energy_unit=args.energy_unit,
    )

    # Pretty summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Events evaluated: {metrics['n_events']:,}")
    print(
        f"Energy resolution (68%): {metrics['energy_resolution']['resolution_68']:.4f}"
    )
    print(
        f"Energy resolution (95%): {metrics['energy_resolution']['resolution_95']:.4f}"
    )
    print(
        f"Energy bias (mean): {metrics['energy_bias']['mean_bias']:.4f}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()

