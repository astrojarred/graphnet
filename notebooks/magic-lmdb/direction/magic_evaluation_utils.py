#!/usr/bin/env python3
"""
MAGIC Direction Reconstruction Evaluation Utilities

Fixed evaluation functions for MAGIC telescope direction reconstruction models.
Handles multi-task models with both classification and VMF outputs properly.

Usage:
    from magic_evaluation_utils import evaluate_from_checkpoint, process_results
    
    results, metrics = evaluate_from_checkpoint(
        model_type="classifier",
        checkpoint_path="path/to/checkpoint.ckpt",
        dataset_config_path="path/to/dataset.yml"
    )
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F

from graphnet.models import StandardModel
from graphnet.data.dataloader import DataLoader
from graphnet.data.dataset import Dataset
from graphnet.data.dataset.dataset import EnsembleDataset
from graphnet.utilities.config import DatasetConfig


def identify_output_format(results_df):
    """Identify the output format of the model predictions.
    
    FIXED: Properly handles multi-task models with both classification and VMF outputs.
    Priority: Use classification outputs over VMF outputs when both are present.
    """
    columns = results_df.columns.tolist()
    
    # Check for direct angle predictions (highest priority - these are trained outputs)
    if 'azimuth_pred' in columns and 'zenith_pred' in columns:
        return 'angles'  # Direct angle predictions from classification task
    
    # Check for VMF format (but deprioritized if angles exist)
    elif ('dir_x_pred' in columns and 'dir_y_pred' in columns and 'dir_z_pred' in columns) or \
         ('dir_x' in columns and 'dir_y' in columns and 'dir_z' in columns):
        return 'vmf'  # VMF format with direction vectors
        
    # Check for classification bins
    elif any('bin_' in col or 'class' in col or 'logit' in col for col in columns):
        return 'classification'  # Classification format
    
    # Handle generic prediction columns from multi-task models
    elif 'pred_0' in columns and 'pred_1' in columns:
        # Count prediction columns
        pred_columns = [col for col in columns if col.startswith('pred_')]
        num_pred_cols = len(pred_columns)
        
        if num_pred_cols == 6:
            # Multi-task model: 2 classification + 4 VMF
            print(f"✓ Detected multi-task model with {num_pred_cols} prediction columns")
            return 'multitask_classification_vmf'
        elif num_pred_cols == 4:
            # VMF only model
            print(f"✓ Detected VMF model with {num_pred_cols} prediction columns")
            return 'generic_vmf'
        elif num_pred_cols == 2:
            # Classification only model
            print(f"✓ Detected classification model with {num_pred_cols} prediction columns")
            return 'generic_classification'
        else:
            print(f"⚠️  Unknown multi-task format with {num_pred_cols} prediction columns")
            return 'generic_unknown'
        
    else:
        print("Available columns:", columns)
        raise ValueError("Unknown output format. Cannot identify prediction columns.")


def angles_to_direction_vector(theta, phi):
    """Convert spherical coordinates to 3D direction vector."""
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    
    x = sin_theta * cos_phi
    y = sin_theta * sin_phi
    z = cos_theta
    
    return x, y, z


def direction_vector_to_angles(x, y, z):
    """Convert 3D direction vector to spherical coordinates."""
    # Normalize to ensure unit vectors
    norm = np.sqrt(x**2 + y**2 + z**2)
    x, y, z = x/norm, y/norm, z/norm
    
    # Convert to spherical coordinates
    theta = np.arccos(np.clip(z, -1.0 + 1e-7, 1.0 - 1e-7))  # zenith angle
    phi = np.arctan2(y, x)  # azimuth angle
    
    return theta, phi


def bins_to_angles(bin_predictions, num_bins=64):
    """Convert classification bin predictions to angles.
    
    Args:
        bin_predictions: Array of bin indices or logits
        num_bins: Total number of angular bins used
    """
    if bin_predictions.ndim > 1:
        # If logits, get the argmax
        bin_idx = np.argmax(bin_predictions, axis=1)
    else:
        bin_idx = bin_predictions.astype(int)
    
    # Convert bin indices back to angles
    # This should match the binning strategy used in MAGICDirectionClassification
    theta_bins = int(num_bins ** 0.5)
    phi_bins = int(num_bins / theta_bins)
    
    theta_bin_idx = bin_idx // phi_bins
    phi_bin_idx = bin_idx % phi_bins
    
    # Convert bin indices back to continuous angles (center of bin)
    theta = (theta_bin_idx + 0.5) / theta_bins * np.pi
    phi = (phi_bin_idx + 0.5) / phi_bins * 2 * np.pi
    
    return theta, phi


def angular_difference(pred_x, pred_y, pred_z, true_x, true_y, true_z):
    """Calculate angular difference between predicted and true directions."""
    # Normalize predicted vectors
    pred_norm = np.sqrt(pred_x**2 + pred_y**2 + pred_z**2)
    pred_x /= pred_norm
    pred_y /= pred_norm  
    pred_z /= pred_norm
    
    # Normalize true vectors (should already be normalized)
    true_norm = np.sqrt(true_x**2 + true_y**2 + true_z**2)
    true_x /= true_norm
    true_y /= true_norm
    true_z /= true_norm
    
    # Calculate dot product
    dot_product = pred_x * true_x + pred_y * true_y + pred_z * true_z
    
    # Clamp to avoid numerical issues
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    # Calculate angle in radians, convert to degrees
    angle_rad = np.arccos(dot_product)
    angle_deg = np.rad2deg(angle_rad)
    
    return angle_deg


def process_results(results_df, num_bins=64):
    """Process model results and compute angular errors.
    
    FIXED: Properly handles multi-task models by prioritizing classification outputs.
    
    Args:
        results_df: DataFrame with model predictions and true values
        num_bins: Number of bins used in classification (if applicable)
    
    Returns:
        DataFrame with added angular error columns
    """
    results = results_df.copy()
    
    # Convert true angles to direction vectors
    results[['true_x', 'true_y', 'true_z']] = pd.DataFrame(
        np.column_stack(angles_to_direction_vector(results['true_theta'], results['true_phi'])),
        index=results.index
    )
    
    # Identify output format and process accordingly
    output_format = identify_output_format(results)
    print(f"Detected output format: {output_format}")
    
    if output_format == 'angles':
        # FIXED: Direct angle predictions (azimuth_pred, zenith_pred) - USE THESE!
        print("✅ Using direct angle predictions from classification task")
        theta_pred = results['zenith_pred']   # Zenith
        phi_pred = results['azimuth_pred']    # Azimuth
        results['theta_pred'] = theta_pred
        results['phi_pred'] = phi_pred
        
        # Convert to direction vectors
        pred_x, pred_y, pred_z = angles_to_direction_vector(theta_pred, phi_pred)
        results['dir_x_pred'] = pred_x
        results['dir_y_pred'] = pred_y
        results['dir_z_pred'] = pred_z
        
    elif output_format == 'vmf':
        # VMF format: already have direction vectors
        print("✅ Using VMF direction vectors")
        
        # Handle both naming conventions
        if 'dir_x_pred' in results.columns:
            print("   Using '_pred' suffix columns")
            pred_x = results['dir_x_pred']
            pred_y = results['dir_y_pred'] 
            pred_z = results['dir_z_pred']
            results['dir_x_pred'] = pred_x
            results['dir_y_pred'] = pred_y
            results['dir_z_pred'] = pred_z
        else:
            print("   Using standard VMF columns (dir_x, dir_y, dir_z)")
            pred_x = results['dir_x']
            pred_y = results['dir_y'] 
            pred_z = results['dir_z']
            results['dir_x_pred'] = pred_x
            results['dir_y_pred'] = pred_y
            results['dir_z_pred'] = pred_z
        
        # Store kappa if available
        if 'kappa' in results.columns:
            results['kappa_pred'] = results['kappa']
        elif 'kappa_pred' in results.columns:
            pass  # Already have it
        
        # Convert to angles for comparison
        theta_pred, phi_pred = direction_vector_to_angles(pred_x, pred_y, pred_z)
        results['theta_pred'] = theta_pred
        results['phi_pred'] = phi_pred
        
    elif output_format == 'classification':
        # Classification format: convert bins to angles, then to direction vectors
        print("✅ Using classification bin predictions")
        # Look for the classification columns
        bin_columns = [col for col in results.columns if 'bin_' in col or 'class' in col or 'logit' in col]
        
        if len(bin_columns) == 1:
            # Single column with bin indices
            bin_predictions = results[bin_columns[0]].values
        else:
            # Multiple columns with logits - need to reconstruct
            bin_predictions = results[bin_columns].values
        
        theta_pred, phi_pred = bins_to_angles(bin_predictions, num_bins)
        results['theta_pred'] = theta_pred
        results['phi_pred'] = phi_pred
        
        # Convert to direction vectors
        pred_x, pred_y, pred_z = angles_to_direction_vector(theta_pred, phi_pred)
        results['dir_x_pred'] = pred_x
        results['dir_y_pred'] = pred_y
        results['dir_z_pred'] = pred_z
    
    elif output_format == 'multitask_classification_vmf':
        # Multi-task model with 6 outputs total
        print("✅ Using multi-task model outputs")
        print("   Analyzing 6 prediction columns...")
        
        # Check if first 2 columns are the transformed classification outputs (theta, phi angles)
        # The MAGICDirectionClassification task may have transform_inference that converts
        # classification logits to angles, so pred_0, pred_1 could be the final angles
        
        # Try to determine which outputs are angles vs. bin logits
        # If pred_0, pred_1 are in reasonable angle ranges, use them as angles
        pred_0_range = (results['pred_0'].min(), results['pred_0'].max())
        pred_1_range = (results['pred_1'].min(), results['pred_1'].max())
        
        print(f"   pred_0 range: {pred_0_range}")
        print(f"   pred_1 range: {pred_1_range}")
        
        # Check if these look like angles (theta: 0-π, phi: 0-2π)
        if (0 <= pred_0_range[0] and pred_0_range[1] <= np.pi + 0.5 and
            0 <= pred_1_range[0] and pred_1_range[1] <= 2*np.pi + 0.5):
            print("   pred_0, pred_1 appear to be converted angles (theta, phi)")
            theta_pred = results['pred_0'].values
            phi_pred = results['pred_1'].values
        else:
            print("   pred_0, pred_1 don't look like angles - assuming they're the VMF outputs")
            # In this case, maybe the order is different. Try VMF in pred_0-3
            print("   Assuming pred_0-3 are VMF (dir_x, dir_y, dir_z, kappa)")
            vmf_x = results['pred_0'].values
            vmf_y = results['pred_1'].values
            vmf_z = results['pred_2'].values
            
            # Normalize and convert to angles
            norm = np.sqrt(vmf_x**2 + vmf_y**2 + vmf_z**2)
            vmf_x, vmf_y, vmf_z = vmf_x/norm, vmf_y/norm, vmf_z/norm
            
            theta_pred, phi_pred = direction_vector_to_angles(vmf_x, vmf_y, vmf_z)
        
        results['theta_pred'] = theta_pred
        results['phi_pred'] = phi_pred
        
        # Convert to direction vectors
        pred_x, pred_y, pred_z = angles_to_direction_vector(theta_pred, phi_pred)
        results['dir_x_pred'] = pred_x
        results['dir_y_pred'] = pred_y
        results['dir_z_pred'] = pred_z
        
        # Store all outputs for analysis
        for i in range(6):
            results[f'raw_pred_{i}'] = results[f'pred_{i}']
        
    elif output_format == 'generic_vmf':
        # Generic VMF model with 4 outputs
        print("✅ Using generic VMF outputs")
        results['dir_x_pred'] = results['pred_0']
        results['dir_y_pred'] = results['pred_1']
        results['dir_z_pred'] = results['pred_2']
        results['kappa_pred'] = results['pred_3']
        
        # Convert to angles
        theta_pred, phi_pred = direction_vector_to_angles(
            results['dir_x_pred'], results['dir_y_pred'], results['dir_z_pred']
        )
        results['theta_pred'] = theta_pred
        results['phi_pred'] = phi_pred
        
    elif output_format == 'generic_classification':
        # Generic 2-output classification model
        print("✅ Using generic classification outputs")
        theta_pred = results['pred_0'].values
        phi_pred = results['pred_1'].values
        
        results['theta_pred'] = theta_pred
        results['phi_pred'] = phi_pred
        
        # Convert to direction vectors
        pred_x, pred_y, pred_z = angles_to_direction_vector(theta_pred, phi_pred)
        results['dir_x_pred'] = pred_x
        results['dir_y_pred'] = pred_y
        results['dir_z_pred'] = pred_z
    
    # Calculate angular error
    results['angular_error_deg'] = angular_difference(
        results['dir_x_pred'], results['dir_y_pred'], results['dir_z_pred'],
        results['true_x'], results['true_y'], results['true_z']
    )
    
    return results


def evaluate_performance(results_df, kappa_threshold=0):
    """Evaluate angular resolution performance.
    
    Args:
        results_df: DataFrame with processed results
        kappa_threshold: Minimum kappa value for filtering (if available)
    
    Returns:
        Dictionary with performance metrics
    """
    # Apply kappa threshold if kappa is available
    if 'kappa_pred' in results_df.columns:
        filtered_results = results_df[results_df['kappa_pred'] > kappa_threshold]
        print(f"Events passing kappa > {kappa_threshold}: {len(filtered_results)}/{len(results_df)} ({100*len(filtered_results)/len(results_df):.1f}%)")
    else:
        filtered_results = results_df
        print("No kappa predictions available - using all events")
    
    if len(filtered_results) == 0:
        print("No events pass the kappa threshold!")
        return {}
    
    angular_errors = filtered_results['angular_error_deg']
    
    metrics = {
        'median_resolution': angular_errors.median(),
        'mean_resolution': angular_errors.mean(),
        'std_resolution': angular_errors.std(),
        'containment_68': angular_errors.quantile(0.68),
        'containment_90': angular_errors.quantile(0.90),
        'containment_95': angular_errors.quantile(0.95),
        'num_events': len(filtered_results),
        'num_events_total': len(results_df)
    }
    
    return metrics


def plot_performance(results_df, kappa_threshold=0, max_angle=10):
    """Plot performance metrics and distributions."""
    # Apply kappa threshold if available
    if 'kappa_pred' in results_df.columns:
        filtered_results = results_df[results_df['kappa_pred'] > kappa_threshold]
    else:
        filtered_results = results_df
    
    if len(filtered_results) == 0:
        print("No events to plot!")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Angular resolution distribution
    axes[0, 0].hist(filtered_results['angular_error_deg'], bins=50, alpha=0.7, 
                   range=(0, max_angle))
    axes[0, 0].set_xlabel('Angular Error (degrees)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Angular Resolution Distribution')
    axes[0, 0].axvline(filtered_results['angular_error_deg'].median(), 
                      color='red', linestyle='--', 
                      label=f'Median: {filtered_results["angular_error_deg"].median():.3f}°')
    axes[0, 0].legend()
    
    # True vs predicted zenith angles
    axes[0, 1].scatter(np.rad2deg(filtered_results['true_theta']), 
                      np.rad2deg(filtered_results['theta_pred']), 
                      alpha=0.5, s=1)
    axes[0, 1].plot([0, 90], [0, 90], 'r--', alpha=0.8)
    axes[0, 1].set_xlabel('True Zenith (degrees)')
    axes[0, 1].set_ylabel('Predicted Zenith (degrees)')
    axes[0, 1].set_title('Zenith Angle Reconstruction')
    
    # True vs predicted azimuth angles  
    axes[1, 0].scatter(np.rad2deg(filtered_results['true_phi']), 
                      np.rad2deg(filtered_results['phi_pred']), 
                      alpha=0.5, s=1)
    axes[1, 0].plot([-180, 180], [-180, 180], 'r--', alpha=0.8)
    axes[1, 0].set_xlabel('True Azimuth (degrees)')
    axes[1, 0].set_ylabel('Predicted Azimuth (degrees)')
    axes[1, 0].set_title('Azimuth Angle Reconstruction')
    
    # Angular error vs energy (if available)
    if 'true_energy' in filtered_results.columns:
        axes[1, 1].scatter(filtered_results['true_energy'], 
                          filtered_results['angular_error_deg'], 
                          alpha=0.5, s=1)
        axes[1, 1].set_xlabel('True Energy')
        axes[1, 1].set_ylabel('Angular Error (degrees)')
        axes[1, 1].set_title('Angular Error vs Energy')
        axes[1, 1].set_yscale('log')
    else:
        # Plot cumulative distribution instead
        sorted_errors = np.sort(filtered_results['angular_error_deg'])
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        axes[1, 1].plot(sorted_errors, cumulative)
        axes[1, 1].set_xlabel('Angular Error (degrees)')
        axes[1, 1].set_ylabel('Cumulative Fraction')
        axes[1, 1].set_title('Cumulative Distribution')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def kappa_performance_analysis(results_df, max_kappa=150):
    """Analyze performance as a function of kappa threshold."""
    if 'kappa_pred' not in results_df.columns:
        print("No kappa predictions available for threshold analysis")
        return
    
    kappa_cuts = range(0, max_kappa, 10)
    metrics = []
    
    for kappa_cut in kappa_cuts:
        filtered = results_df[results_df['kappa_pred'] > kappa_cut]
        if len(filtered) > 0:
            metrics.append({
                'kappa_cut': kappa_cut,
                'median_resolution': filtered['angular_error_deg'].median(),
                'containment_68': filtered['angular_error_deg'].quantile(0.68),
                'containment_95': filtered['angular_error_deg'].quantile(0.95),
                'efficiency': len(filtered) / len(results_df)
            })
    
    metrics_df = pd.DataFrame(metrics)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Resolution vs kappa cut
    axes[0].plot(metrics_df['kappa_cut'], metrics_df['median_resolution'], label='Median')
    axes[0].plot(metrics_df['kappa_cut'], metrics_df['containment_68'], label='68% Containment')
    axes[0].set_xlabel('Kappa Cut')
    axes[0].set_ylabel('Angular Resolution (degrees)')
    axes[0].set_title('Resolution vs Kappa Cut')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Efficiency vs kappa cut
    axes[1].plot(metrics_df['kappa_cut'], metrics_df['efficiency'] * 100)
    axes[1].set_xlabel('Kappa Cut')
    axes[1].set_ylabel('Efficiency (%)')
    axes[1].set_title('Event Efficiency vs Kappa Cut')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return metrics_df


def create_model_from_type(
    model_type: str,
    nb_inputs: int = 7,
    # Transformer parameters
    hidden_dim: int = 256,
    num_layers: int = 8,
    num_heads: int = 8,
    mlp_ratio: int = 4,
    dropout: float = 0.1,
    use_cross_attention: bool = True,
    pool_telescopes: str = "attention",
    # Classifier parameters
    num_fine_bins: int = 64,
    roi_radius: float = 0.5,
    num_coarse_bins: int = 8,
    use_dynedge: bool = True,
    # Hybrid parameters
    ensemble_method: str = "attention",
    # Graph parameters
    nb_nearest_neighbours: int = 16,
):
    """Create model directly using the same function as training script.
    
    This recreates the create_model function from your training script.
    """
    try:
        # Try to import your create_model function directly
        import sys
        import importlib.util
        
        # Add current directory to path to find your script
        current_dir = Path.cwd()
        if str(current_dir) not in sys.path:
            sys.path.insert(0, str(current_dir))
        
        # Try importing from magic_baseline_transform or current namespace
        try:
            from magic_baseline_transform import create_model
            print("✓ Using create_model from magic_baseline_transform.py")
            return create_model(
                model_type=model_type,
                nb_inputs=nb_inputs,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                use_cross_attention=use_cross_attention,
                pool_telescopes=pool_telescopes,
                num_fine_bins=num_fine_bins,
                roi_radius=roi_radius,
                num_coarse_bins=num_coarse_bins,
                use_dynedge=use_dynedge,
                ensemble_method=ensemble_method,
                nb_nearest_neighbours=nb_nearest_neighbours,
            )
        except ImportError:
            print("Could not import create_model function, recreating it...")
            
    except Exception as e:
        print(f"Import failed: {e}, recreating create_model function...")
    
    # Fallback: recreate the model creation logic
    from graphnet.models import StandardModel
    from graphnet.models.graphs import KNNGraph
    from graphnet.models.detector.magic import MAGICDetectorFixed
    from graphnet.models.graphs.nodes import NodesAsPulses
    from graphnet.models.gnn import (
        MAGICTransformer,
        MAGICDirectionClassifier, 
        MAGICHybridModel,
    )
    from graphnet.models.task.magic_reconstruction import (
        MAGICDirectionReconstructionVMF,
        MAGICDirectionClassification,
    )
    from graphnet.training.loss_functions import (
        MAGICFocalLoss,
        MAGICVMFLoss,
    )
    
    print(f"Creating model with type '{model_type}' (fallback method)")
    
    # Create graph definition
    graph_definition = KNNGraph(
        detector=MAGICDetectorFixed(),
        nb_nearest_neighbours=nb_nearest_neighbours,
        node_definition=NodesAsPulses(),
        columns=[0, 1, 2],  # x_cam, y_cam, t for k-NN construction
    )
    
    # Create backbone based on model type
    if model_type == "transformer":
        backbone = MAGICTransformer(
            nb_inputs=graph_definition.nb_outputs,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            max_pixels=2100,
            use_cross_attention=use_cross_attention,
            pool_telescopes=pool_telescopes,
        )
        
        vmf_loss = MAGICVMFLoss(prediction_kappa_index=3)
        task = MAGICDirectionReconstructionVMF(
            hidden_size=backbone.nb_outputs,
            loss_function=vmf_loss,
        )
        tasks = [task]
        
    elif model_type == "classifier":
        backbone = MAGICDirectionClassifier(
            nb_inputs=graph_definition.nb_outputs,
            hidden_dim=hidden_dim,
            num_fine_bins=num_fine_bins,
            roi_radius=roi_radius,
            num_coarse_bins=num_coarse_bins,
            backbone_layers=[256, 512, 512, 256],
            use_dynedge=use_dynedge,
            global_pooling=["mean", "max", "sum"],
        )
        
        classification_task = MAGICDirectionClassification(
            hidden_size=backbone.nb_outputs,
            num_bins=num_fine_bins + num_coarse_bins,
            loss_function=MAGICFocalLoss(alpha=1.0, gamma=2.0),
        )
        
        vmf_task = MAGICDirectionReconstructionVMF(
            hidden_size=backbone.nb_outputs,
            loss_function=MAGICVMFLoss(prediction_kappa_index=3),
        )
        
        tasks = [classification_task, vmf_task]
        
    elif model_type == "hybrid":
        backbone = MAGICHybridModel(
            nb_inputs=graph_definition.nb_outputs,
            hidden_dim=hidden_dim,
            transformer_layers=num_layers // 2,
            transformer_heads=num_heads,
            num_fine_bins=num_fine_bins,
            roi_radius=roi_radius,
            ensemble_method=ensemble_method,
        )
        
        task = MAGICDirectionReconstructionVMF(
            hidden_size=backbone.nb_outputs,
            loss_function=MAGICVMFLoss(prediction_kappa_index=3),
        )
        tasks = [task]
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create optimizer (minimal config for inference)
    optimizer_class = torch.optim.AdamW
    optimizer_kwargs = {"lr": 1e-3}
    
    # Create StandardModel
    model = StandardModel(
        graph_definition=graph_definition,
        backbone=backbone,
        tasks=tasks,
        optimizer_class=optimizer_class,
        optimizer_kwargs=optimizer_kwargs,
    )
    
    return model


def load_model_from_checkpoint(
    model_type: str,
    checkpoint_path: str, 
    backbone_only: bool = True,
    **model_kwargs
):
    """Load a model from model type and checkpoint file.
    
    Args:
        model_type: Type of model ("transformer", "classifier", "hybrid")
        checkpoint_path: Path to model checkpoint (.ckpt file)
        backbone_only: Whether to load only backbone weights (default True)
        **model_kwargs: Additional model parameters
    
    Returns:
        Loaded model ready for inference
    """
    print(f"Creating {model_type} model...")
    model = create_model_from_type(model_type=model_type, **model_kwargs)
    
    if checkpoint_path:
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        old_state_dict = checkpoint['state_dict']
        
        if backbone_only:
            # Load only backbone weights, let tasks reinitialize
            pretrained_dict = {
                k: v for k, v in old_state_dict.items() if k.startswith("backbone.")
            }
            print(f"✓ Loading backbone weights only ({len(pretrained_dict)} parameters)")
        else:
            pretrained_dict = old_state_dict
            print(f"✓ Loading all weights ({len(pretrained_dict)} parameters)")
        
        # Load weights (strict=False allows for missing task head weights)
        missing_keys, unexpected_keys = model.load_state_dict(pretrained_dict, strict=False)
        if missing_keys and backbone_only:
            print(f"Missing keys: {len(missing_keys)} (task heads will be randomly initialized)")
        elif unexpected_keys:
            print(f"Unexpected keys: {len(unexpected_keys)}")
    
    # CRITICAL: Set model to inference mode
    model.eval()
    for task in model._tasks:
        if hasattr(task, 'inference'):
            task.inference()
    
    return model


def run_inference(
    model, 
    dataset_config_path: str,
    test_split: str = "test",
    batch_size: int = 32,
    num_workers: int = 4,
    gpus: list = [0],
    additional_attributes: list = None
):
    """Run inference on test dataset.
    
    FIXED: Properly handles multi-task models and sets inference mode.
    
    Args:
        model: Loaded model ready for inference
        dataset_config_path: Path to dataset configuration YAML file
        test_split: Name of test split in dataset config (default: "test")
        batch_size: Batch size for inference
        num_workers: Number of data loading workers
        gpus: List of GPU IDs to use
        additional_attributes: Additional attributes to include in output
    
    Returns:
        DataFrame with predictions and additional attributes
    """
    if additional_attributes is None:
        additional_attributes = ["true_phi", "true_theta", "event_id"]
        if hasattr(model, 'target_labels') and "true_energy" in model.target_labels:
            additional_attributes.append("true_energy")
    
    print(f"Loading dataset from config: {dataset_config_path}")
    dataset_config = DatasetConfig.load(dataset_config_path)
    datasets = Dataset.from_config(dataset_config)
    
    # Get test dataset
    test_datasets = [datasets[key] for key in datasets if key.startswith(test_split)]
    if not test_datasets:
        available_splits = list(datasets.keys())
        raise ValueError(f"No '{test_split}' split found. Available splits: {available_splits}")
    
    test_dataset = EnsembleDataset(test_datasets)
    print(f"Test dataset contains {len(test_dataset)} events")
    
    # Create test dataloader
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False  # Important: no shuffling for consistent results
    )
    
    print("Running inference...")
    
    # CRITICAL: Ensure model is in inference mode
    model.eval()
    for task in model._tasks:
        if hasattr(task, 'inference'):
            task.inference()
    
    try:
        # Try the built-in predict_as_dataframe first
        results_df = model.predict_as_dataframe(
            test_dataloader,
            additional_attributes=additional_attributes,
            gpus=gpus
        )
        print(f"✓ Inference completed: {len(results_df)} predictions")
        print("Prediction columns:", results_df.columns.tolist())
        return results_df
        
    except Exception as e:
        print(f"Error during inference: {e}")
        print("Attempting manual inference...")
        
        # Manual inference with proper handling
        model.eval()
        
        # Move model to GPU if available
        device = torch.device(f'cuda:{gpus[0]}' if gpus and torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        print(f"Model moved to device: {device}")
        
        # Test with first batch to see output structure
        print("Testing model output with first batch...")
        first_batch = next(iter(test_dataloader))
        first_batch = first_batch.to(device)
        
        with torch.no_grad():
            pred = model(first_batch)
            if isinstance(pred, (list, tuple)):
                print(f"Model outputs {len(pred)} tasks:")
                for i, p in enumerate(pred):
                    print(f"  Task {i}: shape {p.shape}")
                # Concatenate all task outputs
                pred_concat = torch.cat(pred, dim=1)
                print(f"  Concatenated shape: {pred_concat.shape}")
                total_outputs = pred_concat.shape[1]
            else:
                print(f"Model output shape: {pred.shape}")
                total_outputs = pred.shape[1]
            
            print(f"Total output columns: {total_outputs}")
        
        # Process all batches
        predictions = []
        attributes = {attr: [] for attr in additional_attributes}
        
        print(f"Processing {len(test_dataloader)} batches...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_dataloader):
                if batch_idx % 100 == 0:
                    print(f"Processing batch {batch_idx}/{len(test_dataloader)}")
                
                # Move batch to device
                batch = batch.to(device)
                
                # Get predictions
                pred = model(batch)
                
                # Handle multi-task outputs
                if isinstance(pred, (list, tuple)):
                    # Multiple tasks - concatenate outputs
                    pred_concat = torch.cat(pred, dim=1)
                    predictions.append(pred_concat.cpu())
                else:
                    # Single task
                    predictions.append(pred.cpu())
                
                # Get additional attributes
                for attr in additional_attributes:
                    if hasattr(batch, attr):
                        attr_values = getattr(batch, attr).cpu().numpy()
                        attributes[attr].extend(attr_values)
                    elif isinstance(batch, dict) and attr in batch:
                        attr_values = batch[attr].cpu().numpy()
                        attributes[attr].extend(attr_values)
                
                # Clear GPU cache periodically
                if batch_idx % 50 == 0:
                    torch.cuda.empty_cache()
        
        # Combine predictions
        all_predictions = torch.cat(predictions, dim=0).numpy()
        print(f"✓ Manual inference completed. Prediction shape: {all_predictions.shape}")
        
        # Create column names based on model structure
        task_names = [task.__class__.__name__ for task in model._tasks]
        print(f"Model tasks: {task_names}")
        
        # FIXED: Proper column naming for multi-task models
        pred_columns = []
        
        if len(model._tasks) == 2 and 'MAGICDirectionClassificationTask' in task_names:
            # Two tasks: classification + VMF - USE CLASSIFICATION OUTPUTS!
            print("✅ Multi-task model detected: using classification outputs")
            pred_columns = ["azimuth_pred", "zenith_pred", "dir_x_pred", "dir_y_pred", "dir_z_pred", "kappa_pred"]
            
        elif len(model._tasks) == 1:
            task_name = task_names[0]
            if "Classification" in task_name:
                pred_columns = ["azimuth_pred", "zenith_pred"]
            elif "VMF" in task_name:
                pred_columns = ["dir_x_pred", "dir_y_pred", "dir_z_pred", "kappa_pred"]
            else:
                pred_columns = [f"pred_{i}" for i in range(all_predictions.shape[1])]
        else:
            # Fallback
            pred_columns = [f"pred_{i}" for i in range(all_predictions.shape[1])]
        
        # Ensure column count matches
        if len(pred_columns) != all_predictions.shape[1]:
            print(f"Adjusting column count: {len(pred_columns)} -> {all_predictions.shape[1]}")
            pred_columns = [f"pred_{i}" for i in range(all_predictions.shape[1])]
        
        # Create DataFrame
        results_data = {}
        for i, col in enumerate(pred_columns):
            results_data[col] = all_predictions[:, i]
        
        for attr, values in attributes.items():
            if len(values) == len(all_predictions):
                results_data[attr] = values
            else:
                print(f"Warning: Attribute {attr} length mismatch: {len(values)} vs {len(all_predictions)}")
                # Truncate or pad to match
                if len(values) > len(all_predictions):
                    results_data[attr] = values[:len(all_predictions)]
                else:
                    # Pad with last value or NaN
                    padded_values = values + [values[-1] if values else 0] * (len(all_predictions) - len(values))
                    results_data[attr] = padded_values
        
        results_df = pd.DataFrame(results_data)
        print(f"✓ Manual inference completed: {len(results_df)} predictions")
        print("Final columns:", list(results_df.columns)[:10], "..." if len(results_df.columns) > 10 else "")
        
        return results_df


def evaluate_from_checkpoint(
    model_type: str,
    checkpoint_path: str,
    dataset_config_path: str,
    test_split: str = "test",
    batch_size: int = 32,
    num_workers: int = 4,
    gpus: list = [0],
    backbone_only: bool = True,
    save_results: bool = True,
    output_dir: str = "evaluation_results",
    # Model parameters (use defaults from training script)
    hidden_dim: int = 256,
    num_layers: int = 8,
    num_heads: int = 8,
    mlp_ratio: int = 4,
    dropout: float = 0.1,
    num_fine_bins: int = 64,
    roi_radius: float = 0.5,
    num_coarse_bins: int = 8,
    ensemble_method: str = "attention",
    nb_nearest_neighbours: int = 16,
    **model_kwargs
):
    """Complete evaluation pipeline from checkpoint.
    
    FIXED: Properly handles multi-task models and uses correct outputs.
    
    Args:
        model_type: Type of model ("transformer", "classifier", "hybrid")
        checkpoint_path: Path to model checkpoint (.ckpt file)
        dataset_config_path: Path to dataset configuration YAML file
        test_split: Name of test split in dataset config
        batch_size: Batch size for inference
        num_workers: Number of data loading workers  
        gpus: List of GPU IDs to use
        backbone_only: Whether to load only backbone weights
        save_results: Whether to save results to files
        output_dir: Directory to save results
        hidden_dim: Hidden dimension (should match training)
        num_layers: Number of layers (should match training)
        num_heads: Number of attention heads (should match training)
        mlp_ratio: MLP ratio (should match training)
        dropout: Dropout rate (should match training)
        num_fine_bins: Number of fine bins (should match training)
        roi_radius: ROI radius (should match training)
        num_coarse_bins: Number of coarse bins (should match training)
        ensemble_method: Ensemble method for hybrid models
        nb_nearest_neighbours: Number of nearest neighbors for graph
        **model_kwargs: Additional model parameters
    
    Returns:
        Tuple of (processed_results_df, performance_metrics)
    """
    # Combine all model parameters
    model_params = {
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'mlp_ratio': mlp_ratio,
        'dropout': dropout,
        'num_fine_bins': num_fine_bins,
        'roi_radius': roi_radius,
        'num_coarse_bins': num_coarse_bins,
        'ensemble_method': ensemble_method,
        'nb_nearest_neighbours': nb_nearest_neighbours,
        **model_kwargs
    }
    
    # Load model
    model = load_model_from_checkpoint(
        model_type, checkpoint_path, backbone_only, **model_params
    )
    
    # Run inference
    results_df = run_inference(
        model, dataset_config_path, test_split, batch_size, num_workers, gpus
    )
    
    # Process results - determine number of bins for classification models
    total_bins = num_fine_bins + num_coarse_bins if model_type == "classifier" else num_fine_bins
    processed_results = process_results(results_df, num_bins=total_bins)
    
    # Evaluate performance
    print("\nEvaluating performance...")
    performance_metrics = evaluate_performance(processed_results)
    
    # Print summary
    print("\n" + "="*50)
    print("PERFORMANCE SUMMARY")
    print("="*50)
    print(f"Model type: {model_type}")
    for metric, value in performance_metrics.items():
        if 'resolution' in metric or 'containment' in metric:
            print(f"{metric:25s}: {value:.4f}°")
        else:
            print(f"{metric:25s}: {value}")
    print("="*50)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_performance(processed_results)
    
    # Kappa analysis if available
    if 'kappa_pred' in processed_results.columns:
        print("\nAnalyzing kappa threshold performance...")
        kappa_metrics = kappa_performance_analysis(processed_results)
    
    # Save results if requested
    if save_results:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        checkpoint_name = Path(checkpoint_path).stem
        results_file = output_path / f"{checkpoint_name}_{model_type}_results.csv"
        metrics_file = output_path / f"{checkpoint_name}_{model_type}_metrics.json"
        
        processed_results.to_csv(results_file, index=False)
        print(f"✓ Results saved to: {results_file}")
        
        import json
        # Add model info to metrics
        metrics_with_info = {
            **performance_metrics,
            'model_type': model_type,
            'model_parameters': model_params
        }
        with open(metrics_file, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            json_metrics = {k: float(v) if isinstance(v, np.number) else v 
                          for k, v in metrics_with_info.items()}
            json.dump(json_metrics, f, indent=2)
        print(f"✓ Metrics saved to: {metrics_file}")
    
    return processed_results, performance_metrics


# Convenience functions
def quick_vmf_evaluation(checkpoint_path, dataset_config_path, backbone_only=True, **model_kwargs):
    """Quick evaluation for VMF models (dir_x, dir_y, dir_z, kappa format)."""
    return evaluate_from_checkpoint(
        model_type="transformer", 
        checkpoint_path=checkpoint_path, 
        dataset_config_path=dataset_config_path,
        backbone_only=backbone_only,
        **model_kwargs
    )


def quick_classification_evaluation(checkpoint_path, dataset_config_path, num_fine_bins=64, backbone_only=True, **model_kwargs):
    """Quick evaluation for classification models."""
    return evaluate_from_checkpoint(
        model_type="classifier",
        checkpoint_path=checkpoint_path, 
        dataset_config_path=dataset_config_path,
        num_fine_bins=num_fine_bins,
        backbone_only=backbone_only,
        **model_kwargs
    )


def quick_hybrid_evaluation(checkpoint_path, dataset_config_path, backbone_only=True, **model_kwargs):
    """Quick evaluation for hybrid models."""
    return evaluate_from_checkpoint(
        model_type="hybrid",
        checkpoint_path=checkpoint_path, 
        dataset_config_path=dataset_config_path,
        backbone_only=backbone_only,
        **model_kwargs
    )


def evaluate_magic_model(results_path, num_bins=64):
    """Complete evaluation pipeline for MAGIC direction reconstruction from CSV.
    
    Args:
        results_path: Path to CSV file with model predictions
        num_bins: Number of angular bins used (for classification models)
    
    Returns:
        Processed results DataFrame and performance metrics
    """
    print("Loading results...")
    results_df = pd.read_csv(results_path)
    print(f"Loaded {len(results_df)} events")
    print("Columns:", results_df.columns.tolist())
    
    print("\nProcessing results...")
    processed_results = process_results(results_df, num_bins=num_bins)
    
    print("\nEvaluating performance...")
    performance_metrics = evaluate_performance(processed_results)
    
    print("\nPerformance Summary:")
    for metric, value in performance_metrics.items():
        if 'resolution' in metric or 'containment' in metric:
            print(f"{metric}: {value:.4f} degrees")
        else:
            print(f"{metric}: {value}")
    
    print("\nGenerating plots...")
    plot_performance(processed_results)
    
    if 'kappa_pred' in processed_results.columns:
        print("\nAnalyzing kappa threshold performance...")
        kappa_metrics = kappa_performance_analysis(processed_results)
    
    return processed_results, performance_metrics


if __name__ == "__main__":
    print("MAGIC Evaluation Utils loaded successfully!")
    print("Available functions:")
    print("  - evaluate_from_checkpoint()")
    print("  - quick_classification_evaluation()")
    print("  - quick_vmf_evaluation()")
    print("  - quick_hybrid_evaluation()")
    print("  - evaluate_magic_model()")
    print("  - process_results()")
    print("  - evaluate_performance()")
    print("  - plot_performance()")
