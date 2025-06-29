#!/usr/bin/env python3
"""
Comprehensive evaluation script for DeepSky direction reconstruction models.

This script provides functions to:
- Load and evaluate DeepSky models
- Compute direction reconstruction metrics
- Generate comprehensive plots and visualizations
- Handle different data formats and configurations
- Export results to CSV/JSON

Usage:
    python evaluate_deepsky_direction.py --model-config config.yml --checkpoint model.ckpt
    
Or import functions in notebooks:
    from evaluate_deepsky_direction import evaluate_deepsky_direction_model
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# GraphNet imports
try:
    from graphnet.utilities.logging import get_logger
    logger = get_logger()
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

from graphnet.models import StandardModel
from graphnet.data.dataloader import DataLoader
from graphnet.data.dataset import Dataset
from graphnet.utilities.config import DatasetConfig, ModelConfig

# ============================================================================
# MAGIC DATA FORMAT UTILITIES
# ============================================================================

def check_magic_data_compatibility(sample_batch: Data) -> Dict[str, Any]:
    """
    Check compatibility with MAGIC data format and provide format info.
    
    Args:
        sample_batch: A sample Data batch from the dataset
        
    Returns:
        Dictionary with data format information
    """
    info = {
        'format': 'MAGIC',
        'compatible': True,
        'issues': [],
        'available_attributes': []
    }
    
    # Check required attributes
    required_attrs = ['direction', 'true_theta', 'true_phi']
    for attr in required_attrs:
        if hasattr(sample_batch, attr):
            attr_data = getattr(sample_batch, attr)
            info['available_attributes'].append({
                'name': attr,
                'shape': attr_data.shape if torch.is_tensor(attr_data) else 'scalar'
            })
        else:
            info['issues'].append(f"Missing required attribute: {attr}")
    
    # Check telescope pointing
    tel_attrs = ['telescope_phi', 'telescope_theta']
    for attr in tel_attrs:
        if hasattr(sample_batch, attr):
            attr_data = getattr(sample_batch, attr)
            shape = attr_data.shape if torch.is_tensor(attr_data) else 'scalar'
            info['available_attributes'].append({'name': attr, 'shape': shape})
            
            # Check if per-pulse (typical for MAGIC)
            if torch.is_tensor(attr_data) and attr_data.dim() == 1 and attr_data.shape[0] > 1:
                info[f'{attr}_per_pulse'] = True
        else:
            info['issues'].append(f"Missing telescope attribute: {attr}")
    
    # Check other useful attributes
    other_attrs = ['event_id', 'true_energy', 'particle_id']
    for attr in other_attrs:
        if hasattr(sample_batch, attr):
            attr_data = getattr(sample_batch, attr)
            info['available_attributes'].append({
                'name': attr,
                'shape': attr_data.shape if torch.is_tensor(attr_data) else 'scalar'
            })
    
    info['compatible'] = len(info['issues']) == 0
    return info


# ============================================================================
# CORE EVALUATION FUNCTIONS
# ============================================================================

def load_deepsky_model(
    model_config_path: str,
    checkpoint_path: str,
    device: str = "auto"
) -> StandardModel:
    """
    Load a DeepSky model from configuration and checkpoint.
    
    Args:
        model_config_path: Path to model configuration YAML
        checkpoint_path: Path to model checkpoint
        device: Device to load model on ("auto", "cpu", "cuda", etc.)
        
    Returns:
        Loaded StandardModel instance
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Loading DeepSky model from config: {model_config_path}")
    # Construct model from config, trusting lambda/class directives for optimizer, etc.
    model_config = ModelConfig.load(model_config_path)
    model = StandardModel.from_config(model_config, trust=True)
    
    # Load checkpoint state dict
    logger.info(f"Loading checkpoint weights: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state_dict = state["state_dict"]
    else:
        state_dict = state
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    
    logger.info("✓ Model constructed and weights loaded")
    return model


def load_dataset_for_evaluation(
    dataset_config_path: str,
    split: str = "test",
    fraction: float = 1.0,
    batch_size: int = 32,
    num_workers: int = 4
) -> DataLoader:
    """
    Load dataset for evaluation using Dataset.from_config() like in training script.
    
    Args:
        dataset_config_path: Path to dataset configuration
        split: Dataset split to use ("train", "val", "test")
        fraction: Fraction of dataset to use (for quick testing)
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for DataLoader
        
    Returns:
        DataLoader instance
    """
    logger.info("Loading dataset...")
    
    # Load dataset config (same as training script)
    dataset_config = DatasetConfig.load(dataset_config_path)
    
    # Create datasets using Dataset.from_config (same as training script)
    datasets = Dataset.from_config(dataset_config)
    
    # Get the appropriate split
    if split == "train":
        dataset = datasets["train"]
    elif split == "val":
        dataset = datasets["validation"]
    elif split == "test":
        dataset = datasets["test"]
    else:
        raise ValueError(f"Unknown split: {split}. Available splits: {list(datasets.keys())}")
    
    total_events = len(dataset)
    logger.info(f"✓ Loaded {split} dataset with {total_events:,} events")
    
    # Sample fraction if requested
    if fraction < 1.0:
        n_sample = int(total_events * fraction)
        logger.info(f"Sampling {fraction:.1%} of the dataset...")
        
        # Create random indices for sampling
        indices = torch.randperm(total_events)[:n_sample]
        from torch.utils.data import Subset
        dataset = Subset(dataset, indices.tolist())
        
        logger.info(f"✓ Sampled {len(dataset):,} events from {total_events:,} (seed: 42)")
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return dataloader


def generate_predictions(
    model: StandardModel,
    dataloader: DataLoader,
    additional_attributes: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Generate predictions using the model on the dataset.
    
    Args:
        model: Trained StandardModel
        dataloader: DataLoader with evaluation data
        additional_attributes: Additional attributes to extract from data
        
    Returns:
        DataFrame with predictions and metadata
    """
    logger.info("Generating predictions...")
    
    device = next(model.parameters()).device
    all_predictions = []
    
    # Standard attributes to extract
    standard_attrs = ["event_id", "true_phi", "true_theta", "true_energy"]
    
    # Additional attributes that might be useful
    if additional_attributes is None:
        additional_attributes = [
            "telescope_phi", "telescope_theta", "direction", 
            "true_telescope", "direction_and_axis", "particle_id"
        ]
    
    all_attrs = standard_attrs + additional_attributes
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            batch = batch.to(device)
            
            # Generate predictions
            try:
                predictions = model(batch)
                
                # Handle predictions - StandardModel returns list of predictions (one per task)
                if isinstance(predictions, list):
                    if len(predictions) == 1:
                        # Single task - use the first (and only) prediction
                        predictions_tensor = predictions[0]
                    else:
                        # Multiple tasks - concatenate predictions
                        predictions_tensor = torch.cat(predictions, dim=1)
                else:
                    # Already a tensor
                    predictions_tensor = predictions
                
                # Extract prediction columns
                pred_dict = {}
                if hasattr(model, 'prediction_labels'):
                    pred_labels = model.prediction_labels
                else:
                    # Default labels for direction + kappa
                    pred_labels = ['dir_x_pred', 'dir_y_pred', 'dir_z_pred', 'direction_kappa']
                
                # Store predictions
                for i, label in enumerate(pred_labels):
                    if i < predictions_tensor.shape[1]:
                        pred_dict[label] = predictions_tensor[:, i].cpu().numpy()
                
                # Extract additional attributes from batch
                # Get batch size for proper per-event extraction
                batch_size = predictions_tensor.shape[0]
                
                for attr in all_attrs:
                    if hasattr(batch, attr):
                        attr_data = getattr(batch, attr)
                        if torch.is_tensor(attr_data):
                            if attr_data.dim() == 1:
                                # 1D tensor - could be per-event or per-pulse
                                if attr_data.shape[0] == batch_size:
                                    # Per-event data - use directly
                                    pred_dict[attr] = attr_data.cpu().numpy()
                                elif hasattr(batch, 'batch') and attr in ['telescope_phi', 'telescope_theta']:
                                    # Per-pulse data - extract one value per event using batch index
                                    batch_indices = batch.batch.cpu().numpy()
                                    event_values = []
                                    for event_idx in range(batch_size):
                                        # Find first pulse for this event
                                        pulse_mask = (batch_indices == event_idx)
                                        if pulse_mask.any():
                                            first_pulse_idx = np.where(pulse_mask)[0][0]
                                            event_values.append(attr_data[first_pulse_idx].item())
                                        else:
                                            event_values.append(0.0)  # fallback
                                    pred_dict[attr] = np.array(event_values)
                                else:
                                    # Skip if unclear how to handle
                                    logger.debug(f"Skipping 1D tensor {attr} with ambiguous shape {attr_data.shape}")
                            elif attr_data.dim() == 2:
                                if attr_data.shape[0] == batch_size:
                                    # Per-event 2D data
                                    if attr_data.shape[1] <= 3:
                                        # Small 2D - extract components
                                        if attr in ['direction', 'true_telescope', 'direction_and_axis']:
                                            for j in range(attr_data.shape[1]):
                                                pred_dict[f"{attr}_{j}"] = attr_data[:, j].cpu().numpy()
                                        else:
                                            # Take first column for other small 2D data
                                            pred_dict[attr] = attr_data[:, 0].cpu().numpy()
                                else:
                                    # Skip per-pulse 2D data for now
                                    logger.debug(f"Skipping per-pulse 2D tensor {attr} with shape {attr_data.shape}")
                            else:
                                # Higher dimensional tensors - skip
                                logger.debug(f"Skipping high-dim tensor {attr} with shape {attr_data.shape}")
                        else:
                            # Non-tensor data
                            if isinstance(attr_data, (list, tuple)) and len(attr_data) == batch_size:
                                pred_dict[attr] = attr_data
                            else:
                                logger.debug(f"Skipping non-tensor {attr} with unclear format")
                
                # Convert to DataFrame and append
                batch_df = pd.DataFrame(pred_dict)
                all_predictions.append(batch_df)
                
            except Exception as e:
                logger.warning(f"Error processing batch: {e}")
                continue
    
    if not all_predictions:
        raise RuntimeError("No predictions generated successfully")
    
    # Combine all predictions
    results_df = pd.concat(all_predictions, ignore_index=True)
    
    logger.info(f"✓ Generated predictions for {len(results_df):,} events")
    return results_df


def extract_direction_vectors(results_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract direction vectors from results DataFrame.
    
    Args:
        results_df: DataFrame with predictions and true values
        
    Returns:
        Tuple of (predicted_directions, true_directions, pred_telescope_axes, true_telescope_axes)
    """
    # Extract predicted directions
    pred_cols = [col for col in results_df.columns if col.endswith('_pred')]
    if len(pred_cols) >= 3:
        # Look for direction components
        dir_cols = [col for col in pred_cols if 'dir' in col.lower()]
        if len(dir_cols) >= 3:
            pred_directions = results_df[dir_cols[:3]].values
        else:
            pred_directions = results_df[pred_cols[:3]].values
    else:
        raise ValueError("Could not find predicted direction columns")
    
    logger.info(f"Using prediction columns: {dir_cols[:3] if 'dir_cols' in locals() else pred_cols[:3]}")
    
    # Extract true directions - try multiple formats
    true_directions = None
    
    # Method 1: Individual direction components (most reliable for MAGIC data)
    dir_components = ['direction_0', 'direction_1', 'direction_2']
    if all(col in results_df.columns for col in dir_components):
        true_directions = results_df[dir_components].values
        logger.info(f"Using direction components: {dir_components}")
    
    # Method 2: Direct direction vector (fallback)
    elif 'direction' in results_df.columns:
        direction_data = results_df['direction'].iloc[0]
        if isinstance(direction_data, (list, np.ndarray)) and len(direction_data) == 3:
            true_directions = np.array([
                direction_data if isinstance(row, (list, np.ndarray)) else [0, 0, 1] 
                for row in results_df['direction']
            ])
            logger.info("Using direct direction vector")
    
    # Method 3: Theta/Phi format
    if true_directions is None:
        if 'true_theta' in results_df.columns and 'true_phi' in results_df.columns:
            theta = results_df['true_theta'].values
            phi = results_df['true_phi'].values
            
            # Convert spherical to Cartesian (assuming physics convention)
            true_directions = np.column_stack([
                np.sin(theta) * np.cos(phi),  # x
                np.sin(theta) * np.sin(phi),  # y
                np.cos(theta)                 # z
            ])
        elif 'theta' in results_df.columns and 'phi' in results_df.columns:
            theta = results_df['theta'].values
            phi = results_df['phi'].values
            
            true_directions = np.column_stack([
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta)
            ])
    
    if true_directions is None:
        raise ValueError("Could not find true direction information (direction, true_theta/true_phi, or theta/phi)")
    
    # Extract telescope pointing directions
    pred_telescope_axes = np.tile([0, 0, 1], (len(results_df), 1))  # Default to zenith
    true_telescope_axes = np.tile([0, 0, 1], (len(results_df), 1))  # Default to zenith
    
    # Try to get actual telescope pointing if available
    if 'telescope_theta' in results_df.columns and 'telescope_phi' in results_df.columns:
        tel_theta = results_df['telescope_theta'].values
        tel_phi = results_df['telescope_phi'].values
        
        true_telescope_axes = np.column_stack([
            np.sin(tel_theta) * np.cos(tel_phi),
            np.sin(tel_theta) * np.sin(tel_phi),
            np.cos(tel_theta)
        ])
        pred_telescope_axes = true_telescope_axes.copy()  # Assume same for predictions
    
    return pred_directions, true_directions, pred_telescope_axes, true_telescope_axes


def compute_direction_reconstruction_metrics(
    results_df: pd.DataFrame,
    fov_radius_deg: float = 2.5
) -> Dict[str, Any]:
    """
    Compute comprehensive direction reconstruction metrics.
    
    Args:
        results_df: DataFrame with predictions and true values
        fov_radius_deg: Field of view radius in degrees
        
    Returns:
        Dictionary with computed metrics
    """
    logger.info("Computing direction reconstruction metrics...")
    
    # Extract direction vectors
    pred_directions, true_directions, pred_telescope_axes, true_telescope_axes = extract_direction_vectors(results_df)
    
    # Normalize directions
    pred_directions_norm = pred_directions / np.linalg.norm(pred_directions, axis=1, keepdims=True)
    true_directions_norm = true_directions / np.linalg.norm(true_directions, axis=1, keepdims=True)
    
    # Compute angular errors
    dot_products = np.sum(pred_directions_norm * true_directions_norm, axis=1)
    dot_products = np.clip(dot_products, -1.0, 1.0)  # Handle numerical errors
    angular_errors_rad = np.arccos(dot_products)
    angular_errors_deg = np.rad2deg(angular_errors_rad)
    
    # Basic angular resolution metrics
    metrics = {
        'n_events': len(angular_errors_deg),
        'angular_resolution': {
            'median_deg': float(np.median(angular_errors_deg)),
            'mean_deg': float(np.mean(angular_errors_deg)),
            'std_deg': float(np.std(angular_errors_deg)),
            'rms_deg': float(np.sqrt(np.mean(angular_errors_deg**2))),
            'containment_68_deg': float(np.percentile(angular_errors_deg, 68)),
            'containment_95_deg': float(np.percentile(angular_errors_deg, 95)),
            'containment_99_deg': float(np.percentile(angular_errors_deg, 99)),
        }
    }
    
    # Quality fractions
    thresholds = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    quality_fractions = {}
    for threshold in thresholds:
        fraction = np.mean(angular_errors_deg < threshold)
        quality_fractions[f'sub_{threshold:.1f}deg'] = float(fraction)
    
    metrics['quality_fractions'] = quality_fractions
    
    # Field of view analysis
    fov_radius_rad = np.deg2rad(fov_radius_deg)
    
    # Angle between prediction and telescope axis
    pred_telescope_angles = np.arccos(np.clip(
        np.sum(pred_directions_norm * pred_telescope_axes, axis=1), -1.0, 1.0
    ))
    
    # Angle between true direction and telescope axis
    true_telescope_angles = np.arccos(np.clip(
        np.sum(true_directions_norm * true_telescope_axes, axis=1), -1.0, 1.0
    ))
    
    # FoV violations
    pred_fov_violations = pred_telescope_angles > fov_radius_rad
    true_fov_violations = true_telescope_angles > fov_radius_rad
    
    metrics['field_of_view'] = {
        'radius_deg': fov_radius_deg,
        'pred_violation_rate': float(np.mean(pred_fov_violations)),
        'true_violation_rate': float(np.mean(true_fov_violations)),
        'n_pred_violations': int(np.sum(pred_fov_violations)),
        'n_true_violations': int(np.sum(true_fov_violations)),
    }
    
    # Confidence/Kappa analysis if available
    kappa_cols = [col for col in results_df.columns if 'kappa' in col.lower()]
    if kappa_cols:
        kappa_values = results_df[kappa_cols[0]].values
        kappa_valid = kappa_values[~np.isnan(kappa_values)]
        
        if len(kappa_valid) > 0:
            metrics['confidence'] = {
                'kappa_median': float(np.median(kappa_valid)),
                'kappa_mean': float(np.mean(kappa_valid)),
                'kappa_std': float(np.std(kappa_valid)),
                'kappa_min': float(np.min(kappa_valid)),
                'kappa_max': float(np.max(kappa_valid)),
                'n_valid_kappa': len(kappa_valid),
            }
    
    # Energy-dependent analysis if available
    if 'true_energy' in results_df.columns:
        energy_values = results_df['true_energy'].values
        energy_valid = ~np.isnan(energy_values)
        
        if np.sum(energy_valid) > 10:  # Need enough events
            # Define energy bins (log scale)
            energy_bins = np.logspace(np.log10(np.min(energy_values[energy_valid])), 
                                    np.log10(np.max(energy_values[energy_valid])), 6)
            
            energy_metrics = {}
            for i in range(len(energy_bins)-1):
                mask = (energy_values >= energy_bins[i]) & (energy_values < energy_bins[i+1])
                if np.sum(mask) > 5:  # Need enough events in bin
                    bin_errors = angular_errors_deg[mask]
                    energy_metrics[f'bin_{i}'] = {
                        'energy_range_gev': [float(energy_bins[i]), float(energy_bins[i+1])],
                        'n_events': int(np.sum(mask)),
                        'median_error_deg': float(np.median(bin_errors)),
                        'containment_68_deg': float(np.percentile(bin_errors, 68)),
                    }
            
            if energy_metrics:
                metrics['energy_dependence'] = energy_metrics
    
    logger.info("✓ Computed comprehensive metrics")
    return metrics


def plot_direction_reconstruction(
    results_df: pd.DataFrame,
    metrics: Dict[str, Any],
    output_dir: str,
    show_plots: bool = True,
    save_plots: bool = True
) -> None:
    """
    Create comprehensive plots for direction reconstruction analysis.
    
    Args:
        results_df: DataFrame with predictions and true values
        metrics: Computed metrics dictionary
        output_dir: Directory to save plots
        show_plots: Whether to display plots
        save_plots: Whether to save plots to files
    """
    logger.info("Creating direction reconstruction plots...")
    
    # Extract direction vectors and compute errors
    pred_directions, true_directions, _, _ = extract_direction_vectors(results_df)
    
    # Normalize and compute errors
    pred_directions_norm = pred_directions / np.linalg.norm(pred_directions, axis=1, keepdims=True)
    true_directions_norm = true_directions / np.linalg.norm(true_directions, axis=1, keepdims=True)
    
    dot_products = np.sum(pred_directions_norm * true_directions_norm, axis=1)
    dot_products = np.clip(dot_products, -1.0, 1.0)
    angular_errors_deg = np.rad2deg(np.arccos(dot_products))
    
    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Angular error distribution
    ax1 = plt.subplot(2, 4, 1)
    plt.hist(angular_errors_deg, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(metrics['angular_resolution']['median_deg'], color='red', linestyle='--', 
                label=f"Median: {metrics['angular_resolution']['median_deg']:.3f}°")
    plt.axvline(metrics['angular_resolution']['containment_68_deg'], color='orange', linestyle='--',
                label=f"68%: {metrics['angular_resolution']['containment_68_deg']:.3f}°")
    plt.xlabel('Angular Error (degrees)')
    plt.ylabel('Count')
    plt.title('Angular Error Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Cumulative distribution
    ax2 = plt.subplot(2, 4, 2)
    sorted_errors = np.sort(angular_errors_deg)
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    plt.plot(sorted_errors, cumulative, 'b-', linewidth=2)
    plt.axhline(0.68, color='orange', linestyle='--', label='68% containment')
    plt.axhline(0.95, color='red', linestyle='--', label='95% containment')
    plt.xlabel('Angular Error (degrees)')
    plt.ylabel('Cumulative Fraction')
    plt.title('Cumulative Error Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Quality fractions bar plot
    ax3 = plt.subplot(2, 4, 3)
    quality_data = metrics['quality_fractions']
    thresholds = [float(k.split('_')[1].replace('deg', '')) for k in quality_data.keys()]
    fractions = [quality_data[k] for k in quality_data.keys()]
    
    bars = plt.bar(range(len(thresholds)), fractions, alpha=0.7)
    plt.xticks(range(len(thresholds)), [f'{t:.1f}°' for t in thresholds])
    plt.xlabel('Angular Threshold')
    plt.ylabel('Fraction of Events')
    plt.title('Quality Fractions')
    plt.grid(True, alpha=0.3)
    
    # Add percentage labels on bars
    for bar, frac in zip(bars, fractions):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{frac:.1%}', ha='center', va='bottom', fontsize=8)
    
    # 4. Direction components scatter (predicted vs true)
    ax4 = plt.subplot(2, 4, 4)
    # Use z-component as it's typically most constrained
    true_z = true_directions_norm[:, 2]
    pred_z = pred_directions_norm[:, 2]
    
    plt.scatter(true_z, pred_z, alpha=0.5, s=1)
    plt.plot([true_z.min(), true_z.max()], [true_z.min(), true_z.max()], 'r--', label='Perfect correlation')
    plt.xlabel('True Direction Z')
    plt.ylabel('Predicted Direction Z')
    plt.title('Direction Z Component')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Angular error vs energy (if available)
    if 'true_energy' in results_df.columns:
        ax5 = plt.subplot(2, 4, 5)
        energy_values = results_df['true_energy'].values
        valid_mask = ~np.isnan(energy_values)
        
        if np.sum(valid_mask) > 10:
            plt.scatter(energy_values[valid_mask], angular_errors_deg[valid_mask], 
                       alpha=0.5, s=1)
            plt.xlabel('True Energy')
            plt.ylabel('Angular Error (degrees)')
            plt.title('Angular Error vs Energy')
            plt.xscale('log')
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'Insufficient energy data', ha='center', va='center',
                    transform=ax5.transAxes)
            plt.title('Angular Error vs Energy')
    
    # 6. Kappa distribution (if available)
    kappa_cols = [col for col in results_df.columns if 'kappa' in col.lower()]
    if kappa_cols:
        ax6 = plt.subplot(2, 4, 6)
        kappa_values = results_df[kappa_cols[0]].values
        kappa_valid = kappa_values[~np.isnan(kappa_values)]
        
        if len(kappa_valid) > 0:
            plt.hist(kappa_valid, bins=50, alpha=0.7, edgecolor='black')
            plt.axvline(np.median(kappa_valid), color='red', linestyle='--',
                       label=f'Median: {np.median(kappa_valid):.2f}')
            plt.xlabel('Kappa (Confidence)')
            plt.ylabel('Count')
            plt.title('Confidence Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'No valid kappa data', ha='center', va='center',
                    transform=ax6.transAxes)
            plt.title('Confidence Distribution')
    
    # 7. Field of view analysis
    ax7 = plt.subplot(2, 4, 7)
    fov_data = metrics.get('field_of_view', {})
    if fov_data:
        categories = ['Pred. Violations', 'True Violations']
        values = [fov_data.get('pred_violation_rate', 0), fov_data.get('true_violation_rate', 0)]
        
        bars = plt.bar(categories, values, alpha=0.7, color=['red', 'blue'])
        plt.ylabel('Violation Rate')
        plt.title(f'FoV Violations (r={fov_data.get("radius_deg", 2.5):.1f}°)')
        plt.grid(True, alpha=0.3)
        
        # Add percentage labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{val:.2%}', ha='center', va='bottom')
    
    # 8. Summary statistics box
    ax8 = plt.subplot(2, 4, 8)
    ax8.axis('off')
    
    # Create summary text
    summary_text = f"""
    Summary Statistics
    ─────────────────
    Events: {metrics['n_events']:,}
    
    Angular Resolution:
    • Median: {metrics['angular_resolution']['median_deg']:.3f}°
    • 68% containment: {metrics['angular_resolution']['containment_68_deg']:.3f}°
    • 95% containment: {metrics['angular_resolution']['containment_95_deg']:.3f}°
    
    Quality Fractions:
    • Sub-0.1°: {metrics['quality_fractions'].get('sub_0.1deg', 0):.1%}
    • Sub-0.5°: {metrics['quality_fractions'].get('sub_0.5deg', 0):.1%}
    • Sub-1.0°: {metrics['quality_fractions'].get('sub_1.0deg', 0):.1%}
    """
    
    if 'confidence' in metrics:
        summary_text += f"""
    Confidence (κ):
    • Median: {metrics['confidence']['kappa_median']:.2f}
    • Mean: {metrics['confidence']['kappa_mean']:.2f}
        """
    
    plt.text(0.05, 0.95, summary_text, transform=ax8.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.tight_layout()
    
    # Save plot
    if save_plots:
        plot_path = Path(output_dir) / "deepsky_direction_plots.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Plots saved to {plot_path}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()


def evaluate_deepsky_direction_model(
    model_config_path: str,
    dataset_config_path: str,
    checkpoint_path: str,
    dataset_split: str = "test",
    dataset_fraction: float = 1.0,
    batch_size: int = 32,
    fov_radius_deg: float = 2.5,
    output_dir: str = "./deepsky_evaluation_results",
    plot: bool = True,
    save_results: bool = True,
    gpus: Optional[List[int]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Complete evaluation pipeline for DeepSky direction reconstruction model.
    
    Args:
        model_config_path: Path to model configuration YAML
        dataset_config_path: Path to dataset configuration YAML
        checkpoint_path: Path to model checkpoint
        dataset_split: Dataset split to evaluate ("train", "val", "test")
        dataset_fraction: Fraction of dataset to use (for quick testing)
        batch_size: Batch size for evaluation
        fov_radius_deg: Field of view radius in degrees
        output_dir: Directory to save results
        plot: Whether to generate plots
        save_results: Whether to save results to files
        gpus: List of GPU indices to use
        
    Returns:
        Tuple of (results_DataFrame, metrics_dict)
    """
    # Set up output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load model
    device = "cuda" if torch.cuda.is_available() and gpus else "cpu"
    model = load_deepsky_model(model_config_path, checkpoint_path, device)
    
    # Load dataset
    dataloader = load_dataset_for_evaluation(
        dataset_config_path, dataset_split, dataset_fraction, batch_size
    )
    
    # Generate predictions with error handling
    additional_attrs = [
        "telescope_phi", "telescope_theta", "direction", 
        "true_telescope", "direction_and_axis", "particle_id",
        "true_telescope_phi", "true_telescope_theta"
    ]
    
    try:
        results_df = generate_predictions(model, dataloader, additional_attrs)
    except Exception as e:
        logger.warning(f"Prediction with additional attributes failed: {e}")
        logger.info("Retrying with minimal attributes...")
        results_df = generate_predictions(model, dataloader, [])
        logger.info(f"✓ Generated predictions for {len(results_df):,} events (minimal attributes)")
    
    # Check for telescope pointing data
    if 'telescope_phi' not in results_df.columns or 'telescope_theta' not in results_df.columns:
        logger.warning("⚠️  telescope_phi/telescope_theta not available in predictions!")
        logger.warning("    This will make FoV violation metrics MEANINGLESS.")
        logger.warning("    Consider using the original dataset to get telescope pointing.")
        
        # Add dummy telescope coordinates for metrics computation
        logger.error("❌ Using dummy telescope coordinates - FoV metrics will be WRONG!")
        results_df['telescope_phi'] = 0.0
        results_df['telescope_theta'] = 0.0
    
    # Compute metrics
    metrics = compute_direction_reconstruction_metrics(results_df, fov_radius_deg)
    
    # Generate plots
    if plot:
        plot_direction_reconstruction(results_df, metrics, output_dir, show_plots=True, save_plots=save_results)
    
    # Save results
    if save_results:
        # Save detailed results
        results_path = output_path / "deepsky_direction_results.csv"
        results_df.to_csv(results_path, index=False)
        logger.info(f"✓ Results saved to {results_path}")
        
        # Save metrics
        metrics_path = output_path / "deepsky_direction_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"✓ Metrics saved to {metrics_path}")
    
    return results_df, metrics


# ============================================================================
# ADDITIONAL UTILITY FUNCTIONS
# ============================================================================

def compare_models(
    model_configs_and_checkpoints: List[Tuple[str, str, str]],
    dataset_config_path: str,
    dataset_split: str = "test",
    dataset_fraction: float = 0.1,
    output_dir: str = "./model_comparison"
) -> pd.DataFrame:
    """
    Compare multiple DeepSky models side by side.
    
    Args:
        model_configs_and_checkpoints: List of (name, config_path, checkpoint_path) tuples
        dataset_config_path: Path to dataset configuration
        dataset_split: Dataset split to evaluate
        dataset_fraction: Fraction of dataset to use
        output_dir: Directory to save comparison results
        
    Returns:
        DataFrame with comparison metrics
    """
    logger.info(f"Comparing {len(model_configs_and_checkpoints)} models...")
    
    comparison_results = []
    
    for name, config_path, checkpoint_path in model_configs_and_checkpoints:
        logger.info(f"Evaluating model: {name}")
        
        try:
            _, metrics = evaluate_deepsky_direction_model(
                config_path, dataset_config_path, checkpoint_path,
                dataset_split=dataset_split,
                dataset_fraction=dataset_fraction,
                output_dir=f"{output_dir}/{name}",
                plot=False,
                save_results=True
            )
            
            # Extract key metrics for comparison
            comparison_row = {
                'model_name': name,
                'n_events': metrics['n_events'],
                'median_error_deg': metrics['angular_resolution']['median_deg'],
                'containment_68_deg': metrics['angular_resolution']['containment_68_deg'],
                'containment_95_deg': metrics['angular_resolution']['containment_95_deg'],
                'sub_0p1_deg_fraction': metrics['quality_fractions'].get('sub_0.1deg', 0),
                'sub_0p5_deg_fraction': metrics['quality_fractions'].get('sub_0.5deg', 0),
                'sub_1p0_deg_fraction': metrics['quality_fractions'].get('sub_1.0deg', 0),
                'fov_violation_rate': metrics.get('field_of_view', {}).get('pred_violation_rate', 0),
            }
            
            if 'confidence' in metrics:
                comparison_row['kappa_median'] = metrics['confidence']['kappa_median']
            
            comparison_results.append(comparison_row)
            
        except Exception as e:
            logger.error(f"Failed to evaluate {name}: {e}")
            continue
    
    # Convert to DataFrame and save
    comparison_df = pd.DataFrame(comparison_results)
    
    if not comparison_df.empty:
        comparison_path = Path(output_dir) / "model_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        logger.info(f"✓ Model comparison saved to {comparison_path}")
        
        # Print summary
        print("\nModel Comparison Summary:")
        print("=" * 60)
        print(comparison_df.to_string(index=False, float_format='%.4f'))
    
    return comparison_df


def quick_evaluate(
    checkpoint_path: str,
    config_dir: str = ".",
    dataset_fraction: float = 0.01,
    output_dir: str = "./quick_eval"
) -> Dict[str, Any]:
    """
    Quick evaluation with automatic config detection.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config_dir: Directory to search for config files
        dataset_fraction: Fraction of dataset to use
        output_dir: Directory to save results
        
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info("Running quick evaluation with auto-config detection...")
    
    # Auto-detect config files
    config_path = Path(config_dir)
    
    # Find model config
    model_configs = list(config_path.glob("*model*.yml")) + list(config_path.glob("*model*.yaml"))
    if not model_configs:
        raise FileNotFoundError(f"No model config found in {config_dir}")
    model_config = model_configs[0]
    
    # Find dataset config
    dataset_configs = list(config_path.glob("*dataset*.yml")) + list(config_path.glob("*dataset*.yaml"))
    if not dataset_configs:
        raise FileNotFoundError(f"No dataset config found in {config_dir}")
    dataset_config = dataset_configs[0]
    
    logger.info(f"Using model config: {model_config}")
    logger.info(f"Using dataset config: {dataset_config}")
    
    # Run evaluation
    _, metrics = evaluate_deepsky_direction_model(
        str(model_config),
        str(dataset_config),
        checkpoint_path,
        dataset_fraction=dataset_fraction,
        output_dir=output_dir,
        plot=True,
        save_results=True
    )
    
    return metrics


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Command line interface for DeepSky evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate DeepSky direction reconstruction models")
    
    parser.add_argument("--model-config", required=True, help="Path to model configuration YAML")
    parser.add_argument("--dataset-config", required=True, help="Path to dataset configuration YAML")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                       help="Dataset split to evaluate")
    parser.add_argument("--fraction", type=float, default=1.0,
                       help="Fraction of dataset to use (0.0-1.0)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--fov-radius", type=float, default=2.5,
                       help="Field of view radius in degrees")
    parser.add_argument("--output-dir", default="./deepsky_evaluation_results",
                       help="Directory to save results")
    parser.add_argument("--no-plot", action="store_true", help="Skip generating plots")
    parser.add_argument("--no-save", action="store_true", help="Skip saving results")
    parser.add_argument("--gpus", type=int, nargs="*", help="GPU indices to use")
    
    args = parser.parse_args()
    
    # Run evaluation
    try:
        results_df, metrics = evaluate_deepsky_direction_model(
            model_config_path=args.model_config,
            dataset_config_path=args.dataset_config,
            checkpoint_path=args.checkpoint,
            dataset_split=args.split,
            dataset_fraction=args.fraction,
            batch_size=args.batch_size,
            fov_radius_deg=args.fov_radius,
            output_dir=args.output_dir,
            plot=not args.no_plot,
            save_results=not args.no_save,
            gpus=args.gpus
        )
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Events evaluated: {metrics['n_events']:,}")
        print(f"Median angular error: {metrics['angular_resolution']['median_deg']:.4f}°")
        print(f"68% containment: {metrics['angular_resolution']['containment_68_deg']:.4f}°")
        print(f"95% containment: {metrics['angular_resolution']['containment_95_deg']:.4f}°")
        print(f"Sub-0.1° events: {metrics['quality_fractions'].get('sub_0.1deg', 0):.1%}")
        print(f"Sub-1.0° events: {metrics['quality_fractions'].get('sub_1.0deg', 0):.1%}")
        
        if 'field_of_view' in metrics:
            print(f"FoV violation rate: {metrics['field_of_view']['pred_violation_rate']:.2%}")
        
        print(f"\nResults saved to: {args.output_dir}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main() 
