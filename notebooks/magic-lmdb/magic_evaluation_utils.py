#!/usr/bin/env python3

"""
MAGIC Evaluation Utilities with Inverse Transforms
==================================================

Utility functions to convert standardized predictions back to physical
scales for proper evaluation and interpretation.
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from typing import Union, Tuple
import matplotlib.pyplot as plt


def load_energy_transformer(data_path: str, limits_file: str = None) -> QuantileTransformer:
    """Load the energy transformer from preprocessing limits file.
    
    Args:
        data_path: Path to the data directory (will search for limits files here)
        limits_file: Optional override path to specific limits file
        
    Returns:
        QuantileTransformer used during preprocessing
    """
    # First, try the explicit limits file if provided
    if limits_file and os.path.exists(limits_file):
        try:
            with open(limits_file, 'rb') as f:
                limits = pickle.load(f)
            
            if hasattr(limits, 'energy_transform') and limits.energy_transform is not None:
                print(f"✅ Loaded energy transformer from: {limits_file}")
                return limits.energy_transform
            elif 'energy_transformer' in limits:
                print(f"✅ Loaded energy transformer from: {limits_file}")
                return limits['energy_transformer']
                
        except Exception as e:
            print(f"Warning: Failed to load from {limits_file}: {e}")
    
    # Auto-search in data directory for common filenames
    data_dir = os.path.dirname(data_path) if os.path.isfile(data_path) else data_path
    
    # Common filenames to search for
    possible_files = [
        'energy_transformer.pkl',  # Put this first since it's most specific
        'preprocessing_limits.pkl',
        'limits.pkl', 
        'dataset_limits.pkl',
        'transforms.pkl',
        'statistics.pkl'
    ]
    
    print(f"🔍 Searching for preprocessing limits in: {data_dir}")
    
    for filename in possible_files:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            print(f"   Found: {filename}")
            try:
                with open(filepath, 'rb') as f:
                    limits = pickle.load(f)
                
                # Try different ways the transformer might be stored
                transformer = None
                
                # Check if the whole object IS the transformer
                if hasattr(limits, 'transform') and hasattr(limits, 'inverse_transform'):
                    transformer = limits
                    print(f"✅ Loaded energy transformer from: {filepath} (direct object)")
                
                # Check for energy_transform attribute
                elif hasattr(limits, 'energy_transform') and limits.energy_transform is not None:
                    transformer = limits.energy_transform
                    print(f"✅ Loaded energy transformer from: {filepath} (as attribute)")
                
                # Check for energy_transformer in dict
                elif isinstance(limits, dict) and 'energy_transformer' in limits:
                    transformer = limits['energy_transformer']
                    print(f"✅ Loaded energy transformer from: {filepath} (from dict)")
                
                # Check for other common keys
                elif isinstance(limits, dict):
                    for key in ['qt', 'quantile_transformer', 'energy_qt', 'transformer']:
                        if key in limits and hasattr(limits[key], 'transform'):
                            transformer = limits[key]
                            print(f"✅ Loaded energy transformer from: {filepath} (key: {key})")
                            break
                
                if transformer is not None:
                    return transformer
                else:
                    print(f"   No energy transformer found in {filename}")
                    if isinstance(limits, dict):
                        print(f"   Available keys: {list(limits.keys())}")
                    elif hasattr(limits, '__dict__'):
                        print(f"   Available attributes: {list(limits.__dict__.keys())}")
                        
            except Exception as e:
                print(f"   Failed to load {filename}: {e}")
                continue
    
    # If nothing found, create dummy transformer
    print(f"⚠️  No energy transformer found in {data_dir}")
    print("   Creating dummy transformer for evaluation.")
    print("   Results may not be in correct physical units!")
    
    # Create a dummy transformer
    qt = QuantileTransformer(n_quantiles=1000, output_distribution='uniform')
    # Dummy training data for log10(energy) in GeV (10 GeV to 100 TeV range)
    dummy_energies = np.random.uniform(1, 5, 10000).reshape(-1, 1)  
    qt.fit(dummy_energies)
    return qt


class MAGICInverseTransforms:
    """Inverse transforms for MAGIC standardized data."""
    
    def __init__(self, energy_transformer: QuantileTransformer = None):
        """Initialize with the energy transformer used during preprocessing."""
        self.energy_transformer = energy_transformer
    
    def inverse_energy_transform(self, standardized_energy: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Convert standardized energy back to physical scale (GeV).
        
        Args:
            standardized_energy: Energy values from QuantileTransformer
            
        Returns:
            Energy values in GeV
        """
        if self.energy_transformer is None:
            raise ValueError("Energy transformer not set. Call set_energy_transformer() first.")
        
        # Convert to numpy for sklearn
        if torch.is_tensor(standardized_energy):
            energy_np = standardized_energy.cpu().detach().numpy()
        else:
            energy_np = standardized_energy
            
        # Ensure 2D for sklearn
        if energy_np.ndim == 1:
            energy_np = energy_np.reshape(-1, 1)
            
        # Inverse quantile transform (back to log10 space)
        log10_energy = self.energy_transformer.inverse_transform(energy_np)
        
        # Convert from log10 back to linear energy (GeV)
        linear_energy = np.power(10, log10_energy).flatten()
        
        return linear_energy
    
    def inverse_theta_transform(self, standardized_theta: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Convert standardized theta back to radians.
        
        Args:
            standardized_theta: Theta values in (1 - cos(theta)) space
            
        Returns:
            Theta values in radians
        """
        if torch.is_tensor(standardized_theta):
            theta_std = standardized_theta.cpu().detach().numpy()
        else:
            theta_std = standardized_theta
            
        # standardized_theta = 1 - cos(theta)
        # cos(theta) = 1 - standardized_theta  
        # theta = arccos(1 - standardized_theta)
        cos_theta = 1.0 - theta_std
        cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Ensure valid range for arccos
        theta_rad = np.arccos(cos_theta)
        
        return theta_rad
    
    def inverse_phi_transform(self, standardized_phi: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Convert standardized phi back to radians.
        
        Args:
            standardized_phi: Phi values in [0,1] space
            
        Returns:
            Phi values in radians [0, 2π]
        """
        if torch.is_tensor(standardized_phi):
            phi_std = standardized_phi.cpu().detach().numpy()
        else:
            phi_std = standardized_phi
            
        # standardized_phi = phi / (2π)
        # phi = standardized_phi * 2π
        phi_rad = phi_std * (2.0 * np.pi)
        
        return phi_rad
    
    def set_energy_transformer(self, qt: QuantileTransformer):
        """Set the energy transformer."""
        self.energy_transformer = qt


def convert_results_to_physical_scale(results_df: pd.DataFrame, 
                                      inverse_transforms: MAGICInverseTransforms) -> pd.DataFrame:
    """Convert standardized predictions and targets to physical scales.
    
    Args:
        results_df: DataFrame with standardized predictions and targets
        inverse_transforms: Configured inverse transform object
        
    Returns:
        DataFrame with physical-scale values
    """
    df_physical = results_df.copy()
    
    # Convert energy predictions and targets (if they exist)
    if 'energy_pred' in df_physical.columns:
        df_physical['energy_pred_GeV'] = inverse_transforms.inverse_energy_transform(
            df_physical['energy_pred'].values
        )
    
    if 'true_energy' in df_physical.columns:
        df_physical['true_energy_GeV'] = inverse_transforms.inverse_energy_transform(
            df_physical['true_energy'].values
        )
    
    # Convert direction predictions and targets
    if 'theta_pred' in df_physical.columns:
        df_physical['theta_pred_rad'] = inverse_transforms.inverse_theta_transform(
            df_physical['theta_pred'].values
        )
        df_physical['theta_pred_deg'] = np.degrees(df_physical['theta_pred_rad'])
    
    if 'true_theta' in df_physical.columns:
        df_physical['true_theta_rad'] = inverse_transforms.inverse_theta_transform(
            df_physical['true_theta'].values
        )
        df_physical['true_theta_deg'] = np.degrees(df_physical['true_theta_rad'])
    
    if 'phi_pred' in df_physical.columns:
        df_physical['phi_pred_rad'] = inverse_transforms.inverse_phi_transform(
            df_physical['phi_pred'].values
        )
        df_physical['phi_pred_deg'] = np.degrees(df_physical['phi_pred_rad'])
    
    if 'true_phi' in df_physical.columns:
        df_physical['true_phi_rad'] = inverse_transforms.inverse_phi_transform(
            df_physical['true_phi'].values
        )
        df_physical['true_phi_deg'] = np.degrees(df_physical['true_phi_rad'])
    
    return df_physical


def evaluate_magic_performance(results_df: pd.DataFrame, 
                              inverse_transforms: MAGICInverseTransforms,
                              output_dir: str = "./results") -> dict:
    """Comprehensive evaluation of MAGIC model performance in physical units.
    
    Args:
        results_df: DataFrame with model predictions (standardized)
        inverse_transforms: Configured inverse transform object
        output_dir: Directory to save plots
        
    Returns:
        Dictionary with performance metrics
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to physical scales
    df_phys = convert_results_to_physical_scale(results_df, inverse_transforms)
    
    metrics = {}
    
    # 1. Classification Performance
    if 'particle_id' in df_phys.columns and 'gamma_prob' in df_phys.columns:
        y_true = df_phys['particle_id'].values
        y_prob = df_phys['gamma_prob'].values
        y_pred = (y_prob > 0.5).astype(int)
        
        accuracy = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_prob)
        
        metrics['classification'] = {
            'accuracy': accuracy,
            'auc': auc,
            'total_events': len(df_phys)
        }
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Gamma vs Proton Classification')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/roc_curve.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # 2. Energy Reconstruction (gamma events only)
    gamma_mask = df_phys['particle_id'] == 0 if 'particle_id' in df_phys.columns else np.ones(len(df_phys), dtype=bool)
    
    if gamma_mask.sum() > 0 and 'energy_pred_GeV' in df_phys.columns and 'true_energy_GeV' in df_phys.columns:
        df_gamma = df_phys[gamma_mask]
        
        y_true_energy = df_gamma['true_energy_GeV'].values
        y_pred_energy = df_gamma['energy_pred_GeV'].values
        
        # Remove any invalid predictions
        valid_mask = (y_pred_energy > 0) & (y_true_energy > 0) & np.isfinite(y_pred_energy) & np.isfinite(y_true_energy)
        y_true_energy = y_true_energy[valid_mask]
        y_pred_energy = y_pred_energy[valid_mask]
        
        if len(y_true_energy) > 0:
            # Relative error
            rel_error = (y_pred_energy - y_true_energy) / y_true_energy
            energy_resolution = np.std(rel_error) * 100  # Percentage
            energy_bias = np.mean(rel_error) * 100  # Percentage
            
            # Log-scale metrics (common in gamma-ray astronomy)
            log_true = np.log10(y_true_energy)
            log_pred = np.log10(y_pred_energy)
            log_bias = np.mean(log_pred - log_true)
            log_resolution = np.std(log_pred - log_true)
            
            metrics['energy'] = {
                'resolution_percent': energy_resolution,
                'bias_percent': energy_bias,
                'log_bias': log_bias,
                'log_resolution': log_resolution,
                'energy_range_GeV': (y_true_energy.min(), y_true_energy.max()),
                'gamma_events': len(y_true_energy)
            }
            
            # Energy correlation plot
            plt.figure(figsize=(10, 8))
            plt.subplot(2, 2, 1)
            plt.scatter(y_true_energy, y_pred_energy, alpha=0.6, s=20)
            plt.plot([y_true_energy.min(), y_true_energy.max()], 
                     [y_true_energy.min(), y_true_energy.max()], 'r--')
            plt.xlabel('True Energy (GeV)')
            plt.ylabel('Predicted Energy (GeV)')
            plt.title('Energy Correlation (Linear Scale)')
            plt.grid(True, alpha=0.3)
            
            # Log-scale correlation
            plt.subplot(2, 2, 2)
            plt.scatter(log_true, log_pred, alpha=0.6, s=20)
            plt.plot([log_true.min(), log_true.max()], 
                     [log_true.min(), log_true.max()], 'r--')
            plt.xlabel('True log₁₀(Energy/GeV)')
            plt.ylabel('Predicted log₁₀(Energy/GeV)')
            plt.title('Energy Correlation (Log Scale)')
            plt.grid(True, alpha=0.3)
            
            # Relative error distribution
            plt.subplot(2, 2, 3)
            plt.hist(rel_error, bins=50, alpha=0.7, density=True)
            plt.axvline(0, color='r', linestyle='--', label='Perfect reconstruction')
            plt.axvline(np.mean(rel_error), color='orange', linestyle='-', 
                       label=f'Mean = {np.mean(rel_error):.3f}')
            plt.xlabel('Relative Error (ΔE/E)')
            plt.ylabel('Density')
            plt.title(f'Energy Resolution = {energy_resolution:.1f}%')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Energy vs. relative error
            plt.subplot(2, 2, 4)
            plt.scatter(y_true_energy, rel_error, alpha=0.6, s=20)
            plt.axhline(0, color='r', linestyle='--')
            plt.xlabel('True Energy (GeV)')
            plt.ylabel('Relative Error (ΔE/E)')
            plt.title('Energy Bias vs True Energy')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/energy_performance.png', dpi=150, bbox_inches='tight')
            plt.close()
    
    # 3. Direction Reconstruction (gamma events only)
    if (gamma_mask.sum() > 0 and 
        all(col in df_phys.columns for col in ['theta_pred_rad', 'phi_pred_rad', 'true_theta_rad', 'true_phi_rad'])):
        
        df_gamma = df_phys[gamma_mask]
        
        theta_true = df_gamma['true_theta_rad'].values
        phi_true = df_gamma['true_phi_rad'].values
        theta_pred = df_gamma['theta_pred_rad'].values
        phi_pred = df_gamma['phi_pred_rad'].values
        
        # Remove invalid predictions
        valid_mask = (
            np.isfinite(theta_true) & np.isfinite(phi_true) &
            np.isfinite(theta_pred) & np.isfinite(phi_pred) &
            (theta_true >= 0) & (theta_true <= np.pi) &
            (theta_pred >= 0) & (theta_pred <= np.pi)
        )
        
        if valid_mask.sum() > 0:
            theta_true = theta_true[valid_mask]
            phi_true = phi_true[valid_mask]
            theta_pred = theta_pred[valid_mask]
            phi_pred = phi_pred[valid_mask]
            
            # Angular distance calculation (great circle distance)
            x1 = np.sin(theta_true) * np.cos(phi_true)
            y1 = np.sin(theta_true) * np.sin(phi_true)
            z1 = np.cos(theta_true)
            
            x2 = np.sin(theta_pred) * np.cos(phi_pred)
            y2 = np.sin(theta_pred) * np.sin(phi_pred)
            z2 = np.cos(theta_pred)
            
            dot_product = np.clip(x1*x2 + y1*y2 + z1*z2, -1, 1)
            angular_dist_rad = np.arccos(dot_product)
            angular_dist_deg = np.degrees(angular_dist_rad)
            
            mean_angular_error = float(np.mean(angular_dist_deg))
            angular_resolution_68 = float(np.percentile(angular_dist_deg, 68))
            angular_resolution_95 = float(np.percentile(angular_dist_deg, 95))
            
            metrics['direction'] = {
                'mean_angular_error_deg': mean_angular_error,
                'angular_resolution_68_deg': angular_resolution_68,
                'angular_resolution_95_deg': angular_resolution_95,
                'theta_range_deg': (np.degrees(theta_true.min()), np.degrees(theta_true.max())),
                'valid_events': len(theta_true)
            }
            
            # Direction plots
            plt.figure(figsize=(12, 8))
            
            # Angular distance distribution
            plt.subplot(2, 3, 1)
            plt.hist(angular_dist_deg, bins=50, alpha=0.7, density=True)
            plt.axvline(angular_resolution_68, color='r', linestyle='--', 
                       label=f'68% = {angular_resolution_68:.2f}°')
            plt.axvline(mean_angular_error, color='orange', linestyle='-',
                       label=f'Mean = {mean_angular_error:.2f}°')
            plt.xlabel('Angular Distance (degrees)')
            plt.ylabel('Density')
            plt.title('Angular Resolution Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Theta correlation
            plt.subplot(2, 3, 2)
            plt.scatter(np.degrees(theta_true), np.degrees(theta_pred), alpha=0.6, s=20)
            plt.plot([0, 90], [0, 90], 'r--')
            plt.xlabel('True Zenith (degrees)')
            plt.ylabel('Predicted Zenith (degrees)')
            plt.title('Zenith Angle Correlation')
            plt.grid(True, alpha=0.3)
            
            # Phi correlation
            plt.subplot(2, 3, 3)
            plt.scatter(np.degrees(phi_true), np.degrees(phi_pred), alpha=0.6, s=20)
            plt.plot([0, 360], [0, 360], 'r--')
            plt.xlabel('True Azimuth (degrees)')
            plt.ylabel('Predicted Azimuth (degrees)')
            plt.title('Azimuth Angle Correlation')
            plt.grid(True, alpha=0.3)
            
            # Sky map (true positions)
            plt.subplot(2, 3, 4)
            x_true = np.sin(theta_true) * np.cos(phi_true)
            y_true = np.sin(theta_true) * np.sin(phi_true)
            plt.scatter(x_true, y_true, alpha=0.6, s=20, c='blue', label='True')
            plt.xlabel('X (sin θ cos φ)')
            plt.ylabel('Y (sin θ sin φ)')
            plt.title('True Arrival Directions')
            plt.axis('equal')
            plt.grid(True, alpha=0.3)
            
            # Sky map (predicted positions)
            plt.subplot(2, 3, 5)
            x_pred = np.sin(theta_pred) * np.cos(phi_pred)
            y_pred = np.sin(theta_pred) * np.sin(phi_pred)
            plt.scatter(x_pred, y_pred, alpha=0.6, s=20, c='red', label='Predicted')
            plt.xlabel('X (sin θ cos φ)')
            plt.ylabel('Y (sin θ sin φ)')
            plt.title('Predicted Arrival Directions')
            plt.axis('equal')
            plt.grid(True, alpha=0.3)
            
            # Angular error vs zenith angle
            plt.subplot(2, 3, 6)
            plt.scatter(np.degrees(theta_true), angular_dist_deg, alpha=0.6, s=20)
            plt.xlabel('True Zenith Angle (degrees)')
            plt.ylabel('Angular Error (degrees)')
            plt.title('Angular Error vs Zenith')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/direction_performance.png', dpi=150, bbox_inches='tight')
            plt.close()
    
    return metrics


def evaluate_magic_results_comprehensive(results_df: pd.DataFrame, 
                                        data_path: str,
                                        output_dir: str, 
                                        logger,
                                        limits_file: str = None) -> dict:
    """Comprehensive MAGIC evaluation with automatic transformer loading."""
    logger.info("Evaluating results with inverse transforms...")
    
    # Load energy transformer (auto-search in data directory)
    energy_transformer = load_energy_transformer(data_path, limits_file)
    
    # Create inverse transforms
    inverse_transforms = MAGICInverseTransforms(energy_transformer)
    
    try:
        # Use comprehensive evaluation
        metrics = evaluate_magic_performance(
            results_df, 
            inverse_transforms, 
            output_dir
        )
        
        # Print summary
        print_magic_summary(metrics)
        
        # Also save metrics to file
        import json
        with open(f"{output_dir}/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"📊 Detailed evaluation complete! Plots and metrics saved to: {output_dir}")
        return metrics
        
    except Exception as e:
        logger.warning(f"Comprehensive evaluation failed: {e}")
        logger.info("Falling back to basic evaluation...")
        
        # Fall back to basic evaluation if something goes wrong
        return evaluate_magic_results_basic(results_df, output_dir, logger)


def evaluate_magic_results_basic(results_df: pd.DataFrame, output_dir: str, logger) -> dict:
    """Basic evaluation (fallback) - keeps original functionality."""
    logger.info("Running basic evaluation (standardized values)...")
    
    # Classification metrics
    y_true = results_df['particle_id'].values
    y_prob = results_df['gamma_prob'].values
    y_pred = (y_prob > 0.5).astype(int)
    
    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    
    # Energy metrics (gamma events only) - in standardized space
    gamma_mask = results_df['particle_id'] == 0
    if gamma_mask.sum() > 0:
        df_gamma = results_df[gamma_mask]
        y_true_energy = df_gamma['true_energy'].values
        y_pred_energy = df_gamma['energy_pred'].values
        
        energy_mae = float(np.mean(np.abs(y_pred_energy - y_true_energy)))
        energy_bias = float(np.mean((y_pred_energy - y_true_energy) / (y_true_energy + 1e-8)))
    else:
        energy_mae = energy_bias = 0
        
    # Direction metrics (gamma events only) - in standardized space
    if gamma_mask.sum() > 0:
        theta_true = df_gamma['true_theta'].values
        phi_true = df_gamma['true_phi'].values
        theta_pred = df_gamma['theta_pred'].values
        phi_pred = df_gamma['phi_pred'].values
        
        # Note: This is NOT proper angular distance since values are standardized
        theta_mae = float(np.mean(np.abs(theta_pred - theta_true)))
        phi_mae = float(np.mean(np.abs(phi_pred - phi_true)))
    else:
        theta_mae = phi_mae = 0
    
    # Simple plots
    plt.style.use('default')
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Gamma vs Proton Classification')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/roc_curve_basic.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Energy correlation (if gamma events exist) - in standardized space
    if gamma_mask.sum() > 0:
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true_energy, y_pred_energy, alpha=0.6, s=20)
        plt.plot([y_true_energy.min(), y_true_energy.max()], 
                 [y_true_energy.min(), y_true_energy.max()], 'r--')
        plt.xlabel('True Energy (Standardized)')
        plt.ylabel('Predicted Energy (Standardized)')
        plt.title('Energy Reconstruction (Standardized Space)')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/energy_correlation_basic.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # Print summary
    print("\n" + "="*50)
    print("MAGIC BASELINE GNN RESULTS (BASIC/STANDARDIZED)")
    print("="*50)
    print(f"📊 CLASSIFICATION:")
    print(f"   • AUC: {auc:.4f}")
    print(f"   • Accuracy: {accuracy:.4f}")
    print(f"   • Total events: {len(results_df)}")
    
    if gamma_mask.sum() > 0:
        print(f"\n⚡ ENERGY (Standardized space - Gamma events only):")
        print(f"   • MAE: {energy_mae:.4f}")
        print(f"   • Relative bias: {energy_bias*100:.2f}%")
        print(f"   • Gamma events: {gamma_mask.sum()}")
        
        print(f"\n🎯 DIRECTION (Standardized space - Gamma events only):")
        print(f"   • Theta MAE: {theta_mae:.4f}")
        print(f"   • Phi MAE: {phi_mae:.4f}")
    
    print(f"\n📁 Basic plots saved: roc_curve_basic.png, energy_correlation_basic.png")
    print("\n⚠️  NOTE: Energy and direction metrics are in standardized space!")
    print("    For physical interpretation, provide --limits-file for proper inverse transforms.")
    print("="*50)
    
    # Return basic metrics
    metrics = {
        'classification': {
            'accuracy': accuracy,
            'auc': auc,
            'total_events': len(results_df)
        }
    }
    
    if gamma_mask.sum() > 0:
        metrics['energy_basic'] = {
            'mae_standardized': energy_mae,
            'bias_standardized': energy_bias,
            'gamma_events': gamma_mask.sum()
        }
        metrics['direction_basic'] = {
            'theta_mae_standardized': theta_mae,
            'phi_mae_standardized': phi_mae
        }
    
    return metrics


def print_magic_summary(metrics: dict):
    """Print a comprehensive summary of MAGIC model performance."""
    print("\n" + "="*60)
    print("🔭 MAGIC MULTITASK GNN PERFORMANCE SUMMARY")
    print("="*60)
    
    if 'classification' in metrics:
        c = metrics['classification']
        print(f"📊 GAMMA/PROTON CLASSIFICATION:")
        print(f"   • AUC Score: {c['auc']:.4f}")
        print(f"   • Accuracy: {c['accuracy']:.4f}")
        print(f"   • Total events: {c['total_events']:,}")
    
    if 'energy' in metrics:
        e = metrics['energy']
        print(f"\n⚡ ENERGY RECONSTRUCTION (Gamma events):")
        print(f"   • Resolution (σ_rel): {e['resolution_percent']:.1f}%")
        print(f"   • Bias (μ_rel): {e['bias_percent']:+.1f}%")
        print(f"   • Log₁₀ resolution: {e['log_resolution']:.3f}")
        print(f"   • Log₁₀ bias: {e['log_bias']:+.3f}")
        print(f"   • Energy range: {e['energy_range_GeV'][0]:.0f} - {e['energy_range_GeV'][1]:.0f} GeV")
        print(f"   • Valid events: {e['gamma_events']:,}")
    
    if 'direction' in metrics:
        d = metrics['direction']
        print(f"\n🎯 DIRECTION RECONSTRUCTION (Gamma events):")
        print(f"   • Mean angular error: {d['mean_angular_error_deg']:.2f}°")
        print(f"   • 68% containment: {d['angular_resolution_68_deg']:.2f}°")
        print(f"   • 95% containment: {d['angular_resolution_95_deg']:.2f}°")
        print(f"   • Zenith range: {d['theta_range_deg'][0]:.1f}° - {d['theta_range_deg'][1]:.1f}°")
        print(f"   • Valid events: {d['valid_events']:,}")
    
    print("\n📈 Performance plots saved to output directory")
    print("="*60)


# Example usage:
if __name__ == "__main__":
    print("MAGIC evaluation utilities loaded successfully!")
    print("Use evaluate_magic_results_comprehensive() for full evaluation with automatic transformer loading.") 
