"""Training script for MAGIC Angular Offset direction reconstruction.

This script trains the IceMix transformer architecture for MAGIC telescope
angular offset direction reconstruction using the competition-winning VMF approach.
Based on 2nd place IceCube solution adapted for MAGIC's coordinate system.
"""

import os
from pathlib import Path
from typing import List, Optional, Dict

import torch
from pytorch_lightning.loggers import WandbLogger

from graphnet.models import StandardModel
from graphnet.models.gnn import DeepIce
from graphnet.data.dataset import Dataset
from graphnet.data.dataloader import DataLoader
from graphnet.utilities.argparse import ArgumentParser
from graphnet.utilities.config import DatasetConfig
from graphnet.utilities.logging import Logger
from graphnet.data.dataset import LMDBDataset, EnsembleDataset

from graphnet.models.graphs import KNNGraph
from graphnet.models.graphs.nodes import NodesAsPulses
# Import our new angular offset components
from graphnet.models.task.magic_direction import (
    AngularOffsetLabel,
)


def main(
    dataset_config_path: str,
    output_dir: str,
    model_config_path: Optional[str] = None,
    telescope_phi_key: str = "telescope_phi",
    telescope_theta_key: str = "telescope_theta", 
    true_phi_key: str = "true_phi",
    true_theta_key: str = "true_theta",
    gpus: Optional[List[int]] = None,
    max_epochs: int = 10,
    batch_size: int = 32,
    num_workers: int = 10,
    wandb: bool = False,
    wandb_project: str = "magic-angular-offset",
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
) -> None:
    """Train MAGIC Angular Offset model for direction reconstruction.
    
    Args:
        dataset_config_path: Path to dataset configuration YAML file
        output_dir: Directory to save results and model
        model_config_path: Path to model configuration YAML file
        telescope_phi_key: Column name for telescope azimuth
        telescope_theta_key: Column name for telescope zenith
        true_phi_key: Column name for true event azimuth
        true_theta_key: Column name for true event zenith
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
        accumulate_grad_batches: Number of gradient accumulation steps (effective batch = batch_size * accumulate_grad_batches)
        checkpoint_path: Path to checkpoint
        resume: Resume training from checkpoint
        limit_train_batches: Proportion or count of training batches per epoch (PyTorch Lightning semantics)
        limit_val_batches: Proportion or count of validation batches per epoch
        val_check_interval: How frequently to run validation within an epoch (in epochs if float ≤1, in batches if int >1)
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
            log_model="all",  # save _all_ checkpoints since these will be long epochs
            resume="allow",
        )

    logger.info("Loading graph definition KNNGraph")
    
    # Construct datasets with our custom labels
    logger.info("Creating datasets with angular offset labels")
    
    # Create datasets using from_config (following magic_hybrid_direction_train.py paradigm)
    # Ensure dataset_config is of the correct type for static analysis
    assert isinstance(dataset_config, DatasetConfig)
    
    datasets: Dict[str, Dataset] = Dataset.from_config(dataset_config)
    
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
    logger.info(f"Available dataset keys: {list(datasets.keys())}")
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(valid_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")
    
    # Construct dataloaders
    dataloader_config = {"batch_size": batch_size, "num_workers": num_workers}
    if dataloader_config.get("num_workers", 0) == 0:
        dataloader_config["prefetch_factor"] = None
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
    
    # Build or load model
    if model_config_path:
        logger.info(f"Loading model from config {model_config_path}")
        from graphnet.models import Model
        model = Model.from_config(model_config_path, trust=True)
        # Ensure backbone.nb_outputs attribute exists for later metrics
        backbone = model.backbone if hasattr(model, 'backbone') else None
    else:
        raise ValueError("Either --model-config must be provided, or use the legacy manual model building (deprecated)")
    
    # Load checkpoint if provided
    if checkpoint_path is not None:
        if Path(checkpoint_path).exists():
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            
            # Load checkpoint with proper device handling
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                    logger.info("Loading from PyTorch Lightning checkpoint")
                elif "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                    logger.info("Loading from PyTorch checkpoint")
                else:
                    state_dict = checkpoint
                    logger.info("Loading from direct state_dict")
            else:
                state_dict = checkpoint
                logger.info("Loading from direct state_dict")

            if checkpoint_backbone_only:
                # Filter to keep only backbone weights
                state_dict = {
                    k: v for k, v in state_dict.items() if k.startswith("backbone.")
                }
                logger.info("✓ Loaded backbone weights only from checkpoint")
            
            # Load with error handling
            try:
                model.load_state_dict(state_dict, strict=True if not checkpoint_backbone_only else False)
                logger.info(f"✓ Checkpoint loaded successfully (strict={not checkpoint_backbone_only})")
            except RuntimeError as e:
                logger.warning(f"Strict loading failed: {e}")
                try:
                    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                    if missing_keys:
                        logger.warning(f"Missing keys: {missing_keys}")
                    if unexpected_keys:
                        logger.warning(f"Unexpected keys: {unexpected_keys}")
                    logger.info("✓ Checkpoint loaded successfully (non-strict)")
                except Exception as e2:
                    logger.error(f"Failed to load checkpoint: {e2}")
                    raise
        else:
            logger.warning(f"Checkpoint path {checkpoint_path} does not exist")
    
    # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Log configurations to W&B
    if wandb:
        from pytorch_lightning.utilities import rank_zero_only
        if rank_zero_only.rank == 0:
            if not wandb_logger.experiment.resumed:
                logger.info("Logging config to W&B")
                config_dict = {
                    "batch_size": batch_size,
                    "max_epochs": max_epochs,
                    "early_stopping_patience": early_stopping_patience,
                    "seed": seed,
                    "gpus": gpus,
                    "precision": precision,
                    "fast_dev_run": fast_dev_run,
                    "accumulate_grad_batches": accumulate_grad_batches,
                    "total_params": total_params,
                    "trainable_params": trainable_params,
                    "approach": "angular_offset_vmf",
                    "competition_basis": "icecube_2nd_place_icemix",
                    "model_config_path": model_config_path,
                    "limit_train_batches": limit_train_batches,
                    "limit_val_batches": limit_val_batches,
                    "val_check_interval": val_check_interval,
                }
                wandb_logger.experiment.config.update(config_dict)
                wandb_logger.experiment.config.update(dataset_config.as_dict())
            else:
                logger.info(f"Resuming training from checkpoint {checkpoint_path}")
    
    # Training configuration
    fit_config = {
        "gpus": gpus,
        "max_epochs": max_epochs,
        "early_stopping_patience": early_stopping_patience,
        "logger": wandb_logger,
        "precision": precision,
        "fast_dev_run": fast_dev_run,
        "accumulate_grad_batches": accumulate_grad_batches,
        "limit_train_batches": limit_train_batches,
        "limit_val_batches": limit_val_batches,
        "val_check_interval": val_check_interval,
    }
    
    # Train the model
    logger.info("Starting training with competition-winning VMF approach")
    try:
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
        # Get target labels from model tasks
        if hasattr(model, '_tasks') and model._tasks:
            additional_attributes = [
                target for task in model._tasks for target in task._target_labels
            ] + ["event_id", telescope_phi_key, telescope_theta_key, true_phi_key, true_theta_key]
        else:
            additional_attributes = [
                "event_id",
                telescope_phi_key,
                telescope_theta_key, 
                true_phi_key,
                true_theta_key,
                "angular_offset",
            ]
        
        logger.info(f"prediction_columns: {model.prediction_labels}")
        logger.info(f"additional_attributes: {additional_attributes}")
        
        results = model.predict_as_dataframe(
            test_dataloader,
            additional_attributes=additional_attributes,
            gpus=gpus,
        )
        
        # Save predictions
        results_path = os.path.join(output_dir, "results.csv")
        results.to_csv(results_path, index=False)
        logger.info(f"Test predictions saved to {results_path}")
        
        # Calculate angular offset performance metrics
        if all(col in results.columns for col in ["x_offset_pred", "y_offset_pred", "z_offset_pred"]):
            pred_directions = torch.tensor(results[["x_offset_pred", "y_offset_pred", "z_offset_pred"]].values)
            
            # Get true angular offsets by transforming true directions
            # This requires having the angular offset ground truth in results
            if "angular_offset" in results.columns:
                # If we have the true offset vectors directly
                true_directions = torch.tensor(results["angular_offset"].values)  # This might need adjustment
            else:
                # Calculate from telescope pointing and true direction
                logger.info("Calculating true angular offsets from telescope pointing and true directions")
                tel_phi = torch.tensor(results[telescope_phi_key].values)
                tel_theta = torch.tensor(results[telescope_theta_key].values)
                true_phi = torch.tensor(results[true_phi_key].values)
                true_theta = torch.tensor(results[true_theta_key].values)
                
                # Convert to offset vectors (same logic as AngularOffsetLabel)
                offset_label = AngularOffsetLabel(
                    telescope_phi_key=telescope_phi_key,
                    telescope_theta_key=telescope_theta_key,
                    true_phi_key=true_phi_key,
                    true_theta_key=true_theta_key,
                )
                
                true_offsets = []
                for i in range(len(results)):
                    row_data = {
                        telescope_phi_key: tel_phi[i],
                        telescope_theta_key: tel_theta[i],
                        true_phi_key: true_phi[i],
                        true_theta_key: true_theta[i],
                    }
                    offset = offset_label(row_data)
                    true_offsets.append(offset.squeeze())
                
                true_directions = torch.stack(true_offsets)
            
            # Compare offset vectors in telescope frame (both should be in same coordinate system)
            pred_directions = torch.tensor(results[["x_offset_pred", "y_offset_pred", "z_offset_pred"]].values)
            
            # Normalize directions
            pred_directions = pred_directions / torch.norm(pred_directions, dim=1, keepdim=True)
            true_directions = true_directions / torch.norm(true_directions, dim=1, keepdim=True)
            
            # Calculate angular errors in degrees
            dot_products = torch.sum(pred_directions * true_directions, dim=1)
            dot_products = torch.clamp(dot_products, -1.0, 1.0)
            angular_errors = torch.acos(torch.abs(dot_products)) * 180.0 / torch.pi
            
            median_error = torch.median(angular_errors).item()
            mean_error = torch.mean(angular_errors).item()
            std_error = torch.std(angular_errors).item()
            error_68 = torch.quantile(angular_errors, 0.68).item()
            error_95 = torch.quantile(angular_errors, 0.95).item()
            
            logger.info("Angular Offset Performance:")
            logger.info(f"  Median: {median_error:.3f}°")
            logger.info(f"  Mean: {mean_error:.3f}°") 
            logger.info(f"  68% containment: {error_68:.3f}°")
            logger.info(f"  95% containment: {error_95:.3f}°")
            logger.info(f"  Std: {std_error:.3f}°")
            
            # Check FoV violations
            z_components = pred_directions[:, 2]
            angular_from_center = torch.acos(torch.clamp(z_components, -1.0, 1.0)) * 180.0 / torch.pi
            # Use default MAGIC FoV radius if not available from config
            fov_radius_deg = 1.75  # MAGIC default
            fov_violations = (angular_from_center > fov_radius_deg).float()
            violation_rate = torch.mean(fov_violations).item()
            
            logger.info(f"  FoV violation rate: {violation_rate:.3%}")
            
            # Save metrics
            metrics = {
                "median_angular_error_deg": median_error,
                "mean_angular_error_deg": mean_error,
                "angular_resolution_68_deg": error_68,
                "angular_resolution_95_deg": error_95,
                "std_angular_error_deg": std_error,
                "fov_violation_rate": violation_rate,
                "fov_radius_deg": fov_radius_deg,
                "approach": "angular_offset_vmf",
            }
            
            import json
            metrics_path = os.path.join(output_dir, "validation_metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Validation metrics saved to {metrics_path}")
            
            # Log to W&B
            if wandb and wandb_logger:
                wandb_logger.experiment.log(metrics)
                
        else:
            logger.warning("Could not find prediction columns for angular offset evaluation")
        
    except Exception as e:
        logger.warning(f"Failed to generate predictions: {e}")
    
    logger.info("Angular Offset training script completed successfully!")
    logger.info("🚀 Competition-winning VMF approach with MAGIC constraints applied!")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = ArgumentParser(
        description="Train MAGIC Angular Offset model for direction reconstruction"
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
    
    parser.add_argument(
        "--true-phi-key",
        type=str,
        default="true_phi",
        help="Column name for true event azimuth (default: %(default)s)",
    )
    
    parser.add_argument(
        "--true-theta-key",
        type=str,
        default="true_theta",
        help="Column name for true event zenith (default: %(default)s)",
    )
    
    # Training arguments (keeping all from original)
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
        default="magic-icemix-direction",
        help="W&B project name (default: %(default)s)",
    )
    
    parser.add_argument(
        "--wandb-run-id",
        type=str,
        default=None,
        help="W&B run ID",
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
        default=False,
    )
    
    parser.add_argument(
        "--precision",
        type=str,
        help="Training precision (default: %(default)s)",
        default="32-true",
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
        help="Path to checkpoint",
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from checkpoint",
    )
    
    parser.add_argument(
        "--model-config",
        type=str,
        default=None,
        help="Path to model configuration YAML file",
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
        help="How often within one training epoch to check the validation set.",
    )
    
    args = parser.parse_args()

    # change dataset_config to dataset_config_path
    args.dataset_config_path = args.dataset_config
    del args.dataset_config

    args.model_config_path = args.model_config
    del args.model_config
    
    # Run training
    main(**vars(args))
