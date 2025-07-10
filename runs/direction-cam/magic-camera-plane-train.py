"""Training script for MAGIC Camera Plane direction reconstruction.

This script trains models for MAGIC telescope camera plane direction reconstruction
using energy-weighted Euclidean loss. Coordinates are in [-1.5, 1.5] range where
±1.0 corresponds to ±2.5° from camera center.
"""

import os
from pathlib import Path
from typing import List, Optional, Dict

import torch
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging

from graphnet.data.dataset import Dataset
from graphnet.data.dataloader import DataLoader
from graphnet.utilities.argparse import ArgumentParser
from graphnet.utilities.config import DatasetConfig
from graphnet.utilities.logging import Logger
from graphnet.data.dataset import EnsembleDataset

# Import camera plane components
from graphnet.models.task.magic_direction_cam import (
    EnergyWeightedEuclideanDistanceLoss,
)
from graphnet.models.task.magic_direction_cam import camera_to_sky_wrapper  # if you keep it there

# -----------------------------------------------------------------------------
# Logging callbacks adapted for camera plane coordinates
# -----------------------------------------------------------------------------

from pytorch_lightning.callbacks import Callback
torch.set_float32_matmul_precision("high")


class CameraPlaneMetricsLogger(Callback):
    """Log camera plane specific metrics during training."""

    def __init__(self, log_every_n_steps: int = 50):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Log training loss and camera plane specific metrics."""
        
        # Get batch size for proper logging
        if isinstance(batch, list):
            batch_size = len(batch)
        else:
            batch_size = 1
            
        # Log total training loss every step
        total_loss: Optional[torch.Tensor] = None
        if isinstance(outputs, torch.Tensor):
            total_loss = outputs.detach()
        elif isinstance(outputs, dict) and "loss" in outputs:
            total_loss = outputs["loss"].detach()
            
        if total_loss is not None:
            pl_module.log(
                "train_loss_step",
                total_loss,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                batch_size=batch_size,
            )

        # Log detailed metrics periodically
        if batch_idx % self.log_every_n_steps != 0:
            return

        # Additional forward pass to get predictions and detailed metrics
        with torch.no_grad():
            preds_list = pl_module(batch)

        # Log energy-weighted loss components if available
        if hasattr(pl_module, '_tasks') and hasattr(preds_list, '__iter__'):
            for task, preds in zip(pl_module._tasks, preds_list):
                loss_fn = getattr(task, "_loss_function", None)
                
                if isinstance(loss_fn, EnergyWeightedEuclideanDistanceLoss):
                    # Extract targets for detailed analysis
                    target_coords, target_energy = self._extract_camera_targets(batch)
                    if target_coords is not None and target_energy is not None:
                        # Log energy distribution
                        energy_mean = torch.mean(target_energy).item()
                        energy_std = torch.std(target_energy).item()
                        
                        pl_module.log("train_energy_mean", energy_mean, on_step=True, on_epoch=False, batch_size=batch_size)
                        pl_module.log("train_energy_std", energy_std, on_step=True, on_epoch=False, batch_size=batch_size)
                        
                        # Log camera coordinate statistics
                        pred_coords = preds[:, :2]  # First 2 components are coordinates
                        coord_errors = torch.linalg.vector_norm(pred_coords - target_coords, dim=1)
                        
                        pl_module.log("train_camera_error_mean", torch.mean(coord_errors), on_step=True, on_epoch=False, batch_size=batch_size)
                        pl_module.log("train_camera_error_std", torch.std(coord_errors), on_step=True, on_epoch=False, batch_size=batch_size)
                        
                        # Log prediction ranges
                        pl_module.log("train_pred_x_range", torch.max(pred_coords[:, 0]) - torch.min(pred_coords[:, 0]), on_step=True, on_epoch=False, batch_size=batch_size)
                        pl_module.log("train_pred_y_range", torch.max(pred_coords[:, 1]) - torch.min(pred_coords[:, 1]), on_step=True, on_epoch=False, batch_size=batch_size)

    def _extract_camera_targets(self, batch):
        """Extract camera coordinates and energy from batch."""
        try:
            if isinstance(batch, list):
                # Extract from list of data objects
                coords_list = []
                energy_list = []
                
                for data in batch:
                    if hasattr(data, 'camera_x') and hasattr(data, 'camera_y') and hasattr(data, 'true_energy'):
                        coords_list.append(torch.stack([data.camera_x, data.camera_y]))
                        energy_list.append(data.true_energy)
                    else:
                        return None, None
                
                target_coords = torch.stack(coords_list)
                target_energy = torch.stack(energy_list)
                
            else:
                # Extract from single data object
                if hasattr(batch, 'camera_x') and hasattr(batch, 'camera_y') and hasattr(batch, 'true_energy'):
                    target_coords = torch.stack([batch.camera_x, batch.camera_y], dim=1)
                    target_energy = batch.true_energy
                else:
                    return None, None
            
            return target_coords, target_energy
            
        except Exception:
            return None, None


class CameraPlaneValidationLogger(Callback):
    """Log validation metrics for camera plane reconstruction."""
    
    def __init__(self, 
                 log_every_n_batches: int = 1, 
                 accumulate_over_epoch: bool = True,
                 telescope_pointing_keys: Dict[str, str] | None = None):
        super().__init__()
        self.log_every_n_batches = log_every_n_batches
        self.accumulate_over_epoch = accumulate_over_epoch
        
        # Default telescope pointing keys
        self.telescope_pointing_keys = telescope_pointing_keys or {
            'phi': 'telescope_phi',
            'theta': 'telescope_theta'
        }
        
        # Storage for epoch-level accumulation
        self.val_camera_errors = []
        self.val_angular_errors = []
        
    def _extract_validation_data(self, batch, predictions):
        """Extract camera coordinates, telescope pointing, and predictions."""
        try:
            # Get predictions - first 2 components are camera coordinates
            if isinstance(predictions, list):
                pred_coords = predictions[0][:, :2]
            else:
                pred_coords = predictions[:, :2]
            
            # Extract true camera coordinates and telescope pointing
            true_coords_list = []
            telescope_phi_list = []
            telescope_theta_list = []
            
            if isinstance(batch, list):
                for data in batch:
                    if (hasattr(data, 'camera_x') and hasattr(data, 'camera_y') and 
                        hasattr(data, self.telescope_pointing_keys['phi']) and 
                        hasattr(data, self.telescope_pointing_keys['theta'])):
                        
                        true_coords_list.append(torch.stack([data.camera_x, data.camera_y]))
                        telescope_phi_list.append(getattr(data, self.telescope_pointing_keys['phi']))
                        telescope_theta_list.append(getattr(data, self.telescope_pointing_keys['theta']))
                    else:
                        return None, None, None, None
                        
                true_coords = torch.stack(true_coords_list)
                telescope_phi = torch.stack(telescope_phi_list)
                telescope_theta = torch.stack(telescope_theta_list)
                
            else:
                if (hasattr(batch, 'camera_x') and hasattr(batch, 'camera_y') and 
                    hasattr(batch, self.telescope_pointing_keys['phi']) and 
                    hasattr(batch, self.telescope_pointing_keys['theta'])):
                    
                    true_coords = torch.stack([batch.camera_x, batch.camera_y], dim=1)
                    telescope_phi = getattr(batch, self.telescope_pointing_keys['phi'])
                    telescope_theta = getattr(batch, self.telescope_pointing_keys['theta'])
                else:
                    return None, None, None, None
            
            return pred_coords, true_coords, telescope_phi, telescope_theta
            
        except Exception:
            return None, None, None, None
    
    def _convert_camera_to_angular_errors(self, pred_coords, true_coords, telescope_phi, telescope_theta):
        """Convert camera coordinate errors to angular errors in degrees."""
        try:
            # Convert camera coordinates back to sky coordinates
            pred_phi, pred_theta = camera_to_sky_wrapper(
                telescope_phi, telescope_theta,
                pred_coords[:, 0], pred_coords[:, 1],
            )
            
            true_phi, true_theta = camera_to_sky_wrapper(
                telescope_phi, telescope_theta, 
                true_coords[:, 0], true_coords[:, 1],
            )
            
            # Convert to direction vectors
            pred_directions = torch.stack([
                torch.sin(pred_theta) * torch.cos(pred_phi),
                torch.sin(pred_theta) * torch.sin(pred_phi),
                torch.cos(pred_theta)
            ], dim=1)
            
            true_directions = torch.stack([
                torch.sin(true_theta) * torch.cos(true_phi),
                torch.sin(true_theta) * torch.sin(true_phi),
                torch.cos(true_theta)
            ], dim=1)
            
            # Normalize directions
            pred_directions = torch.nn.functional.normalize(pred_directions, dim=1)
            true_directions = torch.nn.functional.normalize(true_directions, dim=1)
            
            # Compute angular errors
            dot_products = torch.sum(pred_directions * true_directions, dim=1)
            dot_products = torch.clamp(dot_products, -1.0 + 1e-7, 1.0 - 1e-7)
            angular_errors_rad = torch.acos(torch.abs(dot_products))
            angular_errors_deg = angular_errors_rad * 180.0 / torch.pi
            
            return angular_errors_deg
            
        except Exception:
            return None
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Compute and log validation metrics."""
 
        # Skip if not logging this batch
        if batch_idx % self.log_every_n_batches != 0:
            if self.accumulate_over_epoch:
                # Still accumulate for epoch-level metrics
                with torch.no_grad():
                    predictions = pl_module(batch)
                    pred_coords, true_coords, tel_phi, tel_theta = self._extract_validation_data(batch, predictions)
                    
                    if pred_coords is not None and true_coords is not None:
                        # Camera coordinate errors
                        camera_errors = torch.linalg.vector_norm(pred_coords - true_coords, dim=1)
                        self.val_camera_errors.extend(camera_errors.cpu().tolist())
                        
                        # Angular errors
                        if tel_phi is not None and tel_theta is not None:
                            angular_errors = self._convert_camera_to_angular_errors(pred_coords, true_coords, tel_phi, tel_theta)
                            if angular_errors is not None:
                                self.val_angular_errors.extend(angular_errors.cpu().tolist())
            return
 
        # Get batch size for proper logging
        if isinstance(batch, list):
            batch_size = len(batch)
        else:
            batch_size = 1
 
        # Compute predictions and extract data
        with torch.no_grad():
            predictions = pl_module(batch)
            pred_coords, true_coords, telescope_phi, telescope_theta = self._extract_validation_data(batch, predictions)
             
            if pred_coords is None or true_coords is None:
                return
                 
            # Compute camera coordinate errors
            camera_errors = torch.linalg.vector_norm(pred_coords - true_coords, dim=1)
             
            if len(camera_errors) == 0:
                return
 
            # Always accumulate errors for epoch-level metrics
            self.val_camera_errors.extend(camera_errors.cpu().tolist())
 
            # If accumulate_over_epoch is True, do not log per batch – we will log once in on_validation_epoch_end
            if self.accumulate_over_epoch:
                # Also accumulate angular errors if available and return early
                if telescope_phi is not None and telescope_theta is not None:
                    angular_errors = self._convert_camera_to_angular_errors(pred_coords, true_coords, telescope_phi, telescope_theta)
                    if angular_errors is not None and len(angular_errors) > 0:
                        self.val_angular_errors.extend(angular_errors.cpu().tolist())
                return
 
            # ------------------------------------------------------------
            # The following logging happens ONLY when accumulate_over_epoch is False
            # ------------------------------------------------------------
 
            # Batch-level camera metrics
            batch_camera_metrics = {
                'val_camera_error_median': torch.median(camera_errors).item(),
                'val_camera_error_mean': torch.mean(camera_errors).item(),
                'val_camera_error_std': torch.std(camera_errors).item(),
                'val_camera_error_68pct': torch.quantile(camera_errors, 0.68).item(),
                'val_camera_error_95pct': torch.quantile(camera_errors, 0.95).item(),
            }
 
            # Log camera coordinate metrics (batch level)
            pl_module.log_dict(
                batch_camera_metrics,
                on_step=False,
                on_epoch=False,
                prog_bar=False,
                batch_size=batch_size,
                sync_dist=True,
            )
            logger_obj = getattr(trainer, "logger", None)
            if logger_obj is not None and hasattr(logger_obj, "log_metrics"):
                logger_obj.log_metrics(batch_camera_metrics, step=trainer.global_step)
 
             # Compute angular errors if telescope pointing is available
            if telescope_phi is not None and telescope_theta is not None:
                angular_errors = self._convert_camera_to_angular_errors(pred_coords, true_coords, telescope_phi, telescope_theta)
                 
                if angular_errors is not None and len(angular_errors) > 0:
                    batch_angular_metrics = {
                        'val_angular_median_deg': torch.median(angular_errors).item(),
                        'val_angular_mean_deg': torch.mean(angular_errors).item(),
                        'val_angular_68pct_deg': torch.quantile(angular_errors, 0.68).item(),
                        'val_angular_95pct_deg': torch.quantile(angular_errors, 0.95).item(),
                    }
                    # Log angular metrics (batch level)
                    pl_module.log_dict(
                        batch_angular_metrics,
                        on_step=False,
                        on_epoch=False,
                        prog_bar=False,
                        batch_size=batch_size,
                        sync_dist=True,
                    )
                    logger_obj = getattr(trainer, "logger", None)
                    if logger_obj is not None and hasattr(logger_obj, "log_metrics"):
                        logger_obj.log_metrics(batch_angular_metrics, step=trainer.global_step)
    
    def on_validation_epoch_start(self, trainer, pl_module):
        """Reset accumulated errors at start of validation epoch."""
        if self.accumulate_over_epoch:
            self.val_camera_errors = []
            self.val_angular_errors = []
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Compute and log epoch-level metrics."""
        if not self.accumulate_over_epoch:
            return
            
        # Camera coordinate epoch metrics
        if len(self.val_camera_errors) > 0:
            camera_errors_tensor = torch.tensor(self.val_camera_errors)
            
            camera_epoch_metrics = {
                'val_camera_error_median_epoch': torch.median(camera_errors_tensor).item(),
                'val_camera_error_mean_epoch': torch.mean(camera_errors_tensor).item(),
                'val_camera_error_68pct_epoch': torch.quantile(camera_errors_tensor, 0.68).item(),
                'val_camera_error_95pct_epoch': torch.quantile(camera_errors_tensor, 0.95).item(),
                'val_camera_error_std_epoch': torch.std(camera_errors_tensor).item(),
            }
            
            # Log camera epoch metrics
            for metric_name, metric_value in camera_epoch_metrics.items():
                pl_module.log(
                    metric_name,
                    metric_value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=(metric_name == 'val_camera_error_68pct_epoch'),
                    sync_dist=True,
                )
 
            # Also forward the whole dict once to the logger (e.g. W&B)
            logger_obj = getattr(trainer, "logger", None)
            if logger_obj is not None and hasattr(logger_obj, "log_metrics"):
                logger_obj.log_metrics(camera_epoch_metrics, step=trainer.global_step)
        
        # Angular epoch metrics
        if len(self.val_angular_errors) > 0:
            angular_errors_tensor = torch.tensor(self.val_angular_errors)
            
            angular_epoch_metrics = {
                'val_angular_median_deg_epoch': torch.median(angular_errors_tensor).item(),
                'val_angular_mean_deg_epoch': torch.mean(angular_errors_tensor).item(),
                'val_angular_68pct_deg_epoch': torch.quantile(angular_errors_tensor, 0.68).item(),
                'val_angular_95pct_deg_epoch': torch.quantile(angular_errors_tensor, 0.95).item(),
                'val_angular_std_deg_epoch': torch.std(angular_errors_tensor).item(),
            }
            
            # Add quality fractions
            for threshold in [0.1, 0.2, 0.5, 1.0, 2.0]:
                fraction = torch.mean((angular_errors_tensor < threshold).float()).item()
                angular_epoch_metrics[f'val_angular_sub_{threshold:.1f}deg_epoch'] = fraction
            
            # Log angular epoch metrics
            for metric_name, metric_value in angular_epoch_metrics.items():
                pl_module.log(
                    metric_name,
                    metric_value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=(metric_name == 'val_angular_68pct_deg_epoch'),
                    sync_dist=True,
                )
 
            if getattr(trainer, "logger", None) is not None:
                logger_obj = getattr(trainer, "logger", None)
                if logger_obj is not None and hasattr(logger_obj, "log_metrics"):
                    logger_obj.log_metrics(angular_epoch_metrics, step=trainer.global_step)
        
        # Clear accumulated errors
        self.val_camera_errors = []
        self.val_angular_errors = []


class SmartSWACallback(Callback):
    """Custom SWA callback that uses scheduler's final LR instead of fixed value."""
    
    def __init__(self, swa_epoch_start: int = 2, fallback_lr: float = 1e-4):
        super().__init__()
        self.swa_epoch_start = swa_epoch_start
        self.fallback_lr = fallback_lr
        self.swa_callback = None
        self.scheduler_final_lr = None
        
    def on_fit_start(self, trainer, pl_module):
        """Initialize and determine scheduler's final LR."""
        self.scheduler_final_lr = self._estimate_final_lr_from_model(pl_module)
        
        if self.scheduler_final_lr is None:
            self.scheduler_final_lr = self.fallback_lr
            
        # Create the actual SWA callback with the determined LR
        self.swa_callback = StochasticWeightAveraging(
            swa_lrs=self.scheduler_final_lr, 
            swa_epoch_start=self.swa_epoch_start
        )
        
        # Initialize the SWA callback
        self.swa_callback.on_fit_start(trainer, pl_module)
        
        logger = Logger()
        logger.info(f"SmartSWA: Using scheduler final LR {self.scheduler_final_lr:.2e} for SWA")
    
    def _estimate_final_lr_from_model(self, pl_module):
        """Estimate final LR from model's scheduler configuration."""
        try:
            if hasattr(pl_module, '_scheduler_class') and hasattr(pl_module, '_scheduler_kwargs'):
                scheduler_class = pl_module._scheduler_class
                scheduler_kwargs = pl_module._scheduler_kwargs or {}
                base_lr = pl_module._optimizer_kwargs.get('lr', 1e-4) if hasattr(pl_module, '_optimizer_kwargs') else 1e-4
                
                # For PiecewiseLinearLR
                if 'PiecewiseLinearLR' in str(scheduler_class):
                    factors = scheduler_kwargs.get('factors', [1.0])
                    final_factor = factors[-1] if factors else 1.0
                    return base_lr * final_factor
                    
                # For CosineAnnealingLR
                elif 'CosineAnnealingLR' in str(scheduler_class):
                    return scheduler_kwargs.get('eta_min', base_lr * 0.01)
                    
                # For ReduceLROnPlateau
                elif 'ReduceLROnPlateau' in str(scheduler_class):
                    return scheduler_kwargs.get('min_lr', base_lr * 0.01)
                    
        except Exception:
            pass
            
        return None
    
    def on_train_start(self, trainer, pl_module):
        if self.swa_callback and hasattr(self.swa_callback, 'on_train_start'):
            self.swa_callback.on_train_start(trainer, pl_module)

    def on_train_epoch_start(self, trainer, pl_module):
        if self.swa_callback and hasattr(self.swa_callback, 'on_train_epoch_start'):
            if trainer.current_epoch >= self.swa_epoch_start and self.swa_callback._average_model is not None:
                self.swa_callback.on_train_epoch_start(trainer, pl_module)
    
    def on_train_epoch_end(self, trainer, pl_module):
        if self.swa_callback and hasattr(self.swa_callback, 'on_train_epoch_end'):
            if trainer.current_epoch >= self.swa_epoch_start and self.swa_callback._average_model is not None:
                self.swa_callback.on_train_epoch_end(trainer, pl_module)
            
    def on_train_end(self, trainer, pl_module):
        if self.swa_callback and hasattr(self.swa_callback, 'on_train_end'):
            self.swa_callback.on_train_end(trainer, pl_module)


def main(
    dataset_config_path: str,
    output_dir: str,
    model_config_path: Optional[str] = None,
    camera_x_key: str = "camera_x",
    camera_y_key: str = "camera_y",
    energy_key: str = "true_energy",
    telescope_phi_key: str = "telescope_phi",
    telescope_theta_key: str = "telescope_theta",
    gpus: Optional[List[int]] = None,
    max_epochs: int = 10,
    batch_size: int = 32,
    num_workers: int = 10,
    wandb: bool = False,
    wandb_project: str = "magic-camera-plane",
    wandb_run_id: Optional[str] = None,
    early_stopping_patience: int = 5,
    seed: int = 42,
    checkpoint_backbone_only: bool = False,
    precision: str = "32-true",
    fast_dev_run: bool = False,
    accumulate_grad_batches: int = 1,
    checkpoint_path: Optional[str] = None,
    resume: bool = False,
    limit_train_batches: Optional[float] = None,
    limit_val_batches: Optional[float] = None,
    val_check_interval: Optional[float] = None,
    gradient_clip_val: float = 0.0,
    use_swa: bool = False,
    swa_epoch_start: int = 2,
    swa_lrs: Optional[float] = None,
    use_uncertainty: bool = False,
    coord_range: float = 1.5,
    energy_scale: float = 100.0,
    energy_weight_scale: float = 0.575,
    energy_weight_offset: float = 1.075,
) -> None:
    """Train MAGIC Camera Plane direction reconstruction model.
    
    Args:
        dataset_config_path: Path to dataset configuration YAML file
        output_dir: Directory to save results and model
        model_config_path: Path to model configuration YAML file
        camera_x_key: Column name for camera X coordinate
        camera_y_key: Column name for camera Y coordinate
        energy_key: Column name for event energy
        telescope_phi_key: Column name for telescope azimuth
        telescope_theta_key: Column name for telescope zenith
        gpus: List of GPU IDs to use for training
        max_epochs: Maximum number of training epochs
        batch_size: Training batch size
        num_workers: Number of data loading workers
        wandb: Whether to use Weights & Biases logging
        wandb_project: W&B project name
        wandb_run_id: W&B run ID
        early_stopping_patience: Patience for early stopping
        seed: Random seed for reproducibility
        checkpoint_backbone_only: Only load backbone weights from checkpoint
        precision: Training precision
        fast_dev_run: Run single batch for debugging
        accumulate_grad_batches: Number of gradient accumulation steps
        checkpoint_path: Path to checkpoint
        resume: Resume training from checkpoint
        limit_train_batches: Proportion or count of training batches per epoch
        limit_val_batches: Proportion or count of validation batches per epoch
        val_check_interval: How frequently to run validation within an epoch
        gradient_clip_val: The value to clip gradients at
        use_swa: Whether to use Stochastic Weight Averaging
        swa_epoch_start: Epoch to start SWA from
        swa_lrs: Learning rate for SWA
        use_uncertainty: Whether to use uncertainty estimation task
        coord_range: Maximum coordinate range for clamping
        energy_scale: Energy scale parameter for loss function
        energy_weight_scale: Energy weight scale parameter for loss function
        energy_weight_offset: Energy weight offset parameter for loss function
    """
    # Set up logging
    logger = Logger()
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    
    # Load dataset configuration
    logger.info(f"Loading dataset configuration from {dataset_config_path}")
    dataset_config = DatasetConfig.load(dataset_config_path)
    
    # Set up output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save dataset configuration to output directory
    dataset_config_save_path = os.path.join(output_dir, "dataset_config.yml")
    dataset_config.dump(dataset_config_save_path)
    logger.info(f"Dataset configuration saved to {dataset_config_save_path}")
    
    # Set up Weights & Biases logging if requested
    wandb_logger = None
    if wandb:
        wandb_dir = os.path.join(output_dir, "wandb")
        os.makedirs(wandb_dir, exist_ok=True)
        if wandb_run_id is not None:
            logger.info(f"Resuming W&B run {wandb_run_id}")
        wandb_logger = WandbLogger(
            project=wandb_project,
            entity="max-planck",
            id=wandb_run_id,
            save_dir=wandb_dir,
            log_model=True,
            resume="allow",
        )

    logger.info("Loading datasets for camera plane reconstruction")
    
    # Create datasets
    assert isinstance(dataset_config, DatasetConfig)
    datasets = Dataset.from_config(dataset_config)
    
    # Construct datasets from multiple selections
    train_dataset = EnsembleDataset(
        [datasets[key] for key in datasets if key.startswith("train")]
    )
    valid_dataset = EnsembleDataset(
        [datasets[key] for key in datasets if "validation" in key]
    )
    test_dataset = EnsembleDataset(
        [datasets[key] for key in datasets if "test" in key]
    )

    # Debug: Print dataset information
    if isinstance(datasets, dict):
        logger.info(f"Available dataset keys: {list(datasets.keys())}")
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(valid_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")
    
    # Construct dataloaders
    dataloader_config = {
        "batch_size": batch_size,
        "num_workers": num_workers,
    }
    if num_workers > 0:
        dataloader_config["persistent_workers"] = True
        dataloader_config["prefetch_factor"] = 4
    else:
        dataloader_config["persistent_workers"] = False
    
    training_dataloader = DataLoader(
        train_dataset, shuffle=True, **dataloader_config
    )
    validation_dataloader = DataLoader(
        valid_dataset, shuffle=False, **dataloader_config
    )
    test_dataloader = DataLoader(
        test_dataset, shuffle=False, **dataloader_config
    )

    # Add this right after creating validation_dataloader:
    print("🔍 Testing validation dataloader...")
    val_iter = iter(validation_dataloader)
    batch1 = next(val_iter)
    batch2 = next(val_iter)

    # Check if we're getting different batches
    if isinstance(batch1, list):
        data1 = batch1[0]
        data2 = batch2[0]
    else:
        data1 = batch1
        data2 = batch2

    # These should be different if dataloader is working properly
    print(f"Batch 1 energy mean: {data1.true_energy.mean():.6f}")
    print(f"Batch 2 energy mean: {data2.true_energy.mean():.6f}")
    print(f"Same data? {torch.allclose(data1.true_energy, data2.true_energy)}")
    
    # Build or load model
    if model_config_path:
        logger.info(f"Loading model from config {model_config_path}")
        from graphnet.models import Model
        model = Model.from_config(model_config_path, trust=True)
    else:
        raise ValueError("--model-config must be provided")
    
    # Ensure the model has the correct task and loss function
    logger.info(f"Model tasks: {[type(task).__name__ for task in model._tasks]}")
    logger.info(f"Model loss functions: {[type(getattr(task, '_loss_function', None)).__name__ for task in model._tasks]}")
    
    # Load checkpoint if provided
    if checkpoint_path is not None:
        if Path(checkpoint_path).exists():
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            if isinstance(checkpoint, dict):
                if "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                elif "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            if checkpoint_backbone_only:
                state_dict = {
                    k: v for k, v in state_dict.items() if k.startswith("backbone.")
                }
                logger.info("✓ Loading backbone weights only from checkpoint")
            
            try:
                model.load_state_dict(state_dict, strict=True if not checkpoint_backbone_only else False)
                logger.info(f"✓ Checkpoint weights loaded successfully (strict={not checkpoint_backbone_only})")
            except RuntimeError as e:
                logger.warning(f"Strict loading failed: {e}")
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                if missing_keys:
                    logger.warning(f"Missing keys: {missing_keys}")
                if unexpected_keys:
                    logger.warning(f"Unexpected keys: {unexpected_keys}")
                logger.info("✓ Checkpoint weights loaded successfully (non-strict)")
        else:
            logger.warning(f"Checkpoint path {checkpoint_path} does not exist")
    
    # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Log configurations to W&B
    if wandb and wandb_logger:
        from pytorch_lightning.utilities import rank_zero_only
        if rank_zero_only.rank == 0 and not wandb_logger.experiment.resumed:
            config_dict = {
                "batch_size": batch_size,
                "max_epochs": max_epochs,
                "early_stopping_patience": early_stopping_patience,
                "seed": seed,
                "gpus": gpus,
                "precision": precision,
                "total_params": total_params,
                "trainable_params": trainable_params,
                "approach": "camera_plane_euclidean",
                "use_uncertainty": use_uncertainty,
                "coord_range": coord_range,
                "energy_scale": energy_scale,
                "energy_weight_scale": energy_weight_scale,
                "energy_weight_offset": energy_weight_offset,
                "model_config_path": model_config_path,
                "use_swa": use_swa,
                "swa_epoch_start": swa_epoch_start,
                "swa_lrs": swa_lrs,
            }
            wandb_logger.experiment.config.update(config_dict)
            wandb_logger.experiment.config.update(dataset_config.as_dict())
    
    # Prepare callbacks list
    callbacks = []
    
    # Add SWA if requested
    if use_swa:
        if swa_lrs is not None:
            logger.info(f"SWA: Starting epoch {swa_epoch_start}, LR={swa_lrs} (fixed)")
            swa_callback = StochasticWeightAveraging(swa_lrs=swa_lrs, swa_epoch_start=swa_epoch_start)
        else:
            logger.info(f"SWA: Starting epoch {swa_epoch_start}, using scheduler's final LR")
            swa_callback = SmartSWACallback(swa_epoch_start=swa_epoch_start, fallback_lr=1e-4)
        callbacks.append(swa_callback)
    
    # Add camera plane specific logging
    callbacks.append(CameraPlaneMetricsLogger(log_every_n_steps=50))
    
    # Add validation metrics logging
    telescope_keys = {
        'phi': telescope_phi_key,
        'theta': telescope_theta_key
    }
    callbacks.append(CameraPlaneValidationLogger(
        log_every_n_batches=1,
        accumulate_over_epoch=True,
        telescope_pointing_keys=telescope_keys
    ))
    
    # Add checkpoint callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=2,
        save_last=True,
    )
    epoch_checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,
        every_n_epochs=1,
        filename="epoch-{epoch:02d}",
    )
    callbacks.append(checkpoint_callback)
    callbacks.append(epoch_checkpoint_callback)
    
    # Training configuration
    fit_config = {
        "gpus": gpus,
        "max_epochs": max_epochs,
        "distribution_strategy": "ddp_find_unused_parameters_true",
        "early_stopping_patience": early_stopping_patience,
        "logger": wandb_logger,
        "precision": precision,
        "fast_dev_run": fast_dev_run,
        "accumulate_grad_batches": accumulate_grad_batches,
        "limit_train_batches": limit_train_batches,
        "limit_val_batches": limit_val_batches,
        "val_check_interval": val_check_interval,
        "gradient_clip_val": gradient_clip_val,
        "callbacks": callbacks,
    }
    
    # Train the model
    logger.info("Starting camera plane direction reconstruction training")
    logger.info(f"Coordinate range: [-{coord_range}, +{coord_range}] (±1.0 = ±2.5°)")
    logger.info(f"Energy weighting: {energy_scale} GeV scale, {energy_weight_scale:.3f} weight scale, {energy_weight_offset:.3f} offset")
    
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        model.fit(
            training_dataloader,
            validation_dataloader,
            ckpt_path=checkpoint_path if resume else None,
            **fit_config,
        )
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    
    # Save trained model
    model_save_path = os.path.join(output_dir, "trained_model.pth")
    logger.info(f"Saving trained model to {model_save_path}")
    model.save(model_save_path)
    
    # Save state dict
    state_dict_path = os.path.join(output_dir, "state_dict.pth")
    model.save_state_dict(state_dict_path)
    logger.info(f"State dict saved to {state_dict_path}")
    
    # Generate predictions
    logger.info("Generating predictions on test set")
    try:
        additional_attributes = [
            camera_x_key, camera_y_key, energy_key,
            telescope_phi_key, telescope_theta_key,
            "event_id"
        ]
        
        results = model.predict_as_dataframe(
            test_dataloader,
            additional_attributes=additional_attributes,
            gpus=gpus,
        )
        
        # Save predictions
        results_path = os.path.join(output_dir, "results.csv")
        results.to_csv(results_path, index=False)
        logger.info(f"Test predictions saved to {results_path}")
        
        # Calculate performance metrics
        if all(col in results.columns for col in [f"{camera_x_key}_pred", f"{camera_y_key}_pred"]):
            pred_coords = torch.tensor(results[[f"{camera_x_key}_pred", f"{camera_y_key}_pred"]].values)
            true_coords = torch.tensor(results[[camera_x_key, camera_y_key]].values)
            
            # Camera coordinate metrics
            camera_errors = torch.linalg.vector_norm(pred_coords - true_coords, dim=1)
            
            median_camera_error = torch.median(camera_errors).item()
            mean_camera_error = torch.mean(camera_errors).item()
            camera_error_68 = torch.quantile(camera_errors, 0.68).item()
            camera_error_95 = torch.quantile(camera_errors, 0.95).item()
            
            logger.info("Camera Coordinate Performance:")
            logger.info(f"  Median error: {median_camera_error:.4f} camera units")
            logger.info(f"  Mean error: {mean_camera_error:.4f} camera units")
            logger.info(f"  68% containment: {camera_error_68:.4f} camera units")
            logger.info(f"  95% containment: {camera_error_95:.4f} camera units")
            
            # Convert to angular errors if telescope pointing available
            if all(col in results.columns for col in [telescope_phi_key, telescope_theta_key]):
                try:
                    telescope_phi = torch.tensor(results[telescope_phi_key].values)
                    telescope_theta = torch.tensor(results[telescope_theta_key].values)
                    
                    # Convert to angular errors (reuse validation logic)
                    validation_logger = CameraPlaneValidationLogger()
                    angular_errors = validation_logger._convert_camera_to_angular_errors(
                        pred_coords, true_coords, telescope_phi, telescope_theta
                    )
                    
                    if angular_errors is not None:
                        median_angular_error = torch.median(angular_errors).item()
                        mean_angular_error = torch.mean(angular_errors).item()
                        angular_error_68 = torch.quantile(angular_errors, 0.68).item()
                        angular_error_95 = torch.quantile(angular_errors, 0.95).item()
                        
                        logger.info("Angular Performance:")
                        logger.info(f"  Median: {median_angular_error:.3f}°")
                        logger.info(f"  Mean: {mean_angular_error:.3f}°")
                        logger.info(f"  68% containment: {angular_error_68:.3f}°")
                        logger.info(f"  95% containment: {angular_error_95:.3f}°")
                        
                        # Quality fractions
                        for threshold in [0.1, 0.2, 0.5, 1.0]:
                            fraction = torch.mean((angular_errors < threshold).float()).item()
                            logger.info(f"  Sub-{threshold}° fraction: {fraction:.3%}")
                
                except Exception as e:
                    logger.warning(f"Could not compute angular errors: {e}")
            
            # Save metrics
            metrics = {
                "median_camera_error": median_camera_error,
                "mean_camera_error": mean_camera_error,
                "camera_error_68": camera_error_68,
                "camera_error_95": camera_error_95,
                "coord_range": coord_range,
                "approach": "camera_plane_euclidean",
                "use_uncertainty": use_uncertainty,
            }
            
            import json
            metrics_path = os.path.join(output_dir, "test_metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Test metrics saved to {metrics_path}")
            
            # Log to W&B
            if wandb and wandb_logger:
                wandb_logger.experiment.log(metrics)
        
    except Exception as e:
        logger.warning(f"Failed to generate predictions: {e}")
    
    logger.info("Camera plane training script completed successfully!")
    logger.info("🎯 Energy-weighted Euclidean loss with camera coordinate system!")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = ArgumentParser(
        description="Train MAGIC Camera Plane direction reconstruction model"
    )
    
    parser.add_argument(
        "--dataset-config",
        type=str,
        required=True,
        help="Path to dataset configuration YAML file",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save training results and model",
    )
    
    parser.add_argument(
        "--model-config",
        type=str,
        required=True,
        help="Path to model configuration YAML file",
    )
    
    parser.add_argument(
        "--camera-x-key",
        type=str,
        default="camera_x",
        help="Column name for camera X coordinate (default: %(default)s)",
    )
    
    parser.add_argument(
        "--camera-y-key",
        type=str,
        default="camera_y",
        help="Column name for camera Y coordinate (default: %(default)s)",
    )
    
    parser.add_argument(
        "--energy-key",
        type=str,
        default="energy",
        help="Column name for event energy (default: %(default)s)",
    )
    
    parser.add_argument(
        "--telescope-phi-key",
        type=str,
        default="telescope_phi",
        help="Column name for telescope azimuth (default: %(default)s)",
    )
    
    parser.add_argument(
        "--telescope-theta-key",
        type=str,
        default="telescope_theta",
        help="Column name for telescope zenith (default: %(default)s)",
    )
    
    # Training arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size (default: %(default)s)",
    )
    
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=10,
        help="Maximum number of training epochs (default: %(default)s)",
    )
    
    parser.add_argument(
        "--num-workers",
        type=int,
        default=10,
        help="Number of data loading workers (default: %(default)s)",
    )
    
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="*",
        default=[0],
        help="GPU IDs to use for training (default: [0])",
    )
    
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Use Weights & Biases for logging",
    )
    
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="magic-camera-plane",
        help="W&B project name (default: %(default)s)",
    )
    
    parser.add_argument(
        "--wandb-run-id",
        type=str,
        default=None,
        help="W&B run ID for resuming",
    )
    
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=5,
        help="Early stopping patience (default: %(default)s)",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: %(default)s)",
    )
    
    parser.add_argument(
        "--checkpoint-backbone-only",
        action="store_true",
        help="Only load the backbone weights from the checkpoint",
    )
    
    parser.add_argument(
        "--precision",
        type=str,
        default="32-true",
        help="Training precision (default: %(default)s)",
    )
    
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Run a single batch for debugging",
    )
    
    parser.add_argument(
        "--accumulate-grad-batches",
        type=int,
        default=1,
        help="Number of gradient accumulation steps (default: %(default)s)",
    )
    
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Path to checkpoint file",
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from checkpoint",
    )
    
    parser.add_argument(
        "--limit-train-batches",
        type=float,
        default=None,
        help="Fraction (0.0-1.0) or number of training batches per epoch",
    )
    
    parser.add_argument(
        "--limit-val-batches",
        type=float,
        default=None,
        help="Fraction (0.0-1.0) or number of validation batches per epoch",
    )
    
    parser.add_argument(
        "--val-check-interval",
        type=float,
        default=None,
        help="How often within one training epoch to check the validation set",
    )
    
    parser.add_argument(
        "--gradient-clip-val",
        type=float,
        default=0.0,
        help="Gradient clipping value (default: %(default)s)",
    )
    
    parser.add_argument(
        "--use-swa",
        action="store_true",
        help="Use Stochastic Weight Averaging",
    )
    
    parser.add_argument(
        "--swa-epoch-start",
        type=int,
        default=2,
        help="Epoch to start SWA from (default: %(default)s)",
    )
    
    parser.add_argument(
        "--swa-lrs",
        type=float,
        default=None,
        help="Learning rate for SWA (None uses scheduler's final LR)",
    )
    
    # Camera plane specific arguments
    parser.add_argument(
        "--use-uncertainty",
        action="store_true",
        help="Use uncertainty estimation task",
    )
    
    parser.add_argument(
        "--coord-range",
        type=float,
        default=1.5,
        help="Maximum coordinate range for clamping (default: %(default)s)",
    )
    
    parser.add_argument(
        "--energy-scale",
        type=float,
        default=100.0,
        help="Energy scale parameter for loss function (default: %(default)s)",
    )
    
    parser.add_argument(
        "--energy-weight-scale",
        type=float,
        default=0.575,
        help="Energy weight scale parameter for loss function (default: %(default)s)",
    )
    
    parser.add_argument(
        "--energy-weight-offset",
        type=float,
        default=1.075,
        help="Energy weight offset parameter for loss function (default: %(default)s)",
    )

    args = parser.parse_args()

    # Adjust argument names for function call
    args.dataset_config_path = args.dataset_config
    args.model_config_path = args.model_config
    del args.dataset_config, args.model_config
    
    # Run training
    main(**vars(args))
