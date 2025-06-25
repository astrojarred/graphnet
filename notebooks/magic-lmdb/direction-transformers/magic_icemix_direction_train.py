"""Training script for MAGIC IceMix direction reconstruction.

This script trains the IceMix transformer architecture adapted for MAGIC
telescope direction reconstruction. Based on the 2nd place IceCube solution
but modified for MAGIC's dual-telescope stereo data.
"""

import os
from pathlib import Path
from typing import List, Optional

import torch
from pytorch_lightning.loggers import WandbLogger

from graphnet.models import StandardModel
from graphnet.data.datamodule import GraphNeTDataModule
from graphnet.data.dataset import Dataset
from graphnet.data.dataloader import DataLoader
from graphnet.utilities.argparse import ArgumentParser
from graphnet.utilities.config import ModelConfig, DatasetConfig
from graphnet.utilities.logging import Logger


def main(
    model_config_path: str,
    dataset_config_path: str,
    output_dir: str,
    gpus: Optional[List[int]] = None,
    max_epochs: int = 10,
    batch_size: int = 32,
    num_workers: int = 10,
    wandb: bool = False,
    wandb_project: str = "magic-icemix-direction",
    wandb_run_id: Optional[str] = None,
    early_stopping_patience: int = 5,
    seed: int = 42,
    checkpoint_backbone_only: bool = False,
    precision: str = "32-true",
    fast_dev_run: bool = False,
    checkpoint_path: Optional[str] = None,
    resume: bool = False,
) -> None:
    """Train MAGIC IceMix model for direction reconstruction.
    
    Args:
        model_config_path: Path to model configuration YAML file
        dataset_config_path: Path to dataset configuration YAML file
        output_dir: Directory to save results and model
        gpus: List of GPU IDs to use for training
        max_epochs: Maximum number of training epochs
        batch_size: Training batch size
        num_workers: Number of data loading workers
        wandb: Whether to use Weights & Biases logging
        wandb_project: W&B project name
        wandb_run_id: W&B run ID
        early_stopping_patience: Patience for early stopping
        validation_split: Fraction of data to use for validation
        seed: Random seed for reproducibility
        checkpoint_backbone_only: Only load the backbone weights from the checkpoint
        precision: Training precision
        fast_dev_run: Run a single batch for debugging
        checkpoint_path: Path to checkpoint
        resume: Resume training from checkpoint
    """
    # Set up logging
    logger = Logger()
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    
    # Load model configuration
    logger.info(f"Loading model configuration from {model_config_path}")
    model_config = ModelConfig.load(model_config_path)
    
    # Load dataset configuration
    logger.info(f"Loading dataset configuration from {dataset_config_path}")
    dataset_config = DatasetConfig.load(dataset_config_path)
    
    # Set up output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configurations to output directory
    model_config_save_path = os.path.join(output_dir, "model_config.yml")
    dataset_config_save_path = os.path.join(output_dir, "dataset_config.yml")
    model_config.dump(model_config_save_path)
    dataset_config.dump(dataset_config_save_path)
    logger.info(f"Model configuration saved to {model_config_save_path}")
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
        
        # Log configuration to W&B (after wandb_logger is initialized)
        # This will be done after model training starts
    
    # Construct dataloaders
    datasets: Dataset = Dataset.from_config(
        dataset_config,
    )

    # Construct datasets from multiple selections
    from graphnet.data.dataset import EnsembleDataset
    
    train_dataset = EnsembleDataset(
        [datasets[key] for key in datasets if key.startswith("train")]
    )
    valid_dataset = EnsembleDataset(
        [datasets[key] for key in datasets if key.startswith("validation")]
    )
    test_dataset = EnsembleDataset(
        [datasets[key] for key in datasets if key.startswith("test")]
    )

    # Debug: Print dataset information
    logger.info(f"Available dataset keys: {list(datasets.keys())}")
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(valid_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")
    
    # Debug: Check a sample from training data to understand particle_id values
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        logger.info(f"Sample particle_id value: {sample['particle_id']}")
        logger.info(f"Sample particle_id type: {type(sample['particle_id'])}")
        if hasattr(sample['particle_id'], 'shape'):
            logger.info(f"Sample particle_id shape: {sample['particle_id'].shape}")

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
    
    # Build model from configuration
    logger.info("Building MAGIC IceMix model")
    model = StandardModel.from_config(model_config, trust=True)

    # Load checkpoint if provided
    if checkpoint_path is not None:
        if Path(checkpoint_path).exists():
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            
            # Load checkpoint with proper device handling
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if "state_dict" in checkpoint:
                    # PyTorch Lightning checkpoint format
                    state_dict = checkpoint["state_dict"]
                    logger.info("Loading from PyTorch Lightning checkpoint")
                elif "model_state_dict" in checkpoint:
                    # Standard PyTorch checkpoint format
                    state_dict = checkpoint["model_state_dict"]
                    logger.info("Loading from PyTorch checkpoint")
                else:
                    # Assume the entire dict is the state_dict
                    state_dict = checkpoint
                    logger.info("Loading from direct state_dict")
            else:
                # Direct state_dict (not wrapped in dict)
                state_dict = checkpoint
                logger.info("Loading from direct state_dict")

            if checkpoint_backbone_only:
                # Filter the state_dict to keep only keys that start with "backbone."
                state_dict = {
                    k: v for k, v in state_dict.items() if k.startswith("backbone.")
                }
                logger.info("✓ Loaded backbone weights only from checkpoint")
            
            # Load with error handling
            try:
                # Try strict loading first
                model.load_state_dict(state_dict, strict=True if not checkpoint_backbone_only else False)
                logger.info(f"✓ Checkpoint loaded successfully (strict={not checkpoint_backbone_only})")
            except RuntimeError as e:
                logger.warning(f"Strict loading failed: {e}")
                try:
                    # Try non-strict loading
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
    
    # Print model summary
    # logger.info(f"Model: {model}")
    # logger.info(f"Backbone: {model.backbone}")
    # logger.info(f"Tasks: {model._tasks}")
    
    # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Log configurations to W&B (after model is created)
    if wandb:
        from pytorch_lightning.utilities import rank_zero_only
        if rank_zero_only.rank == 0:
            if not wandb_logger.experiment.resumed:
                logger.info("Logging config to W&B")
                # Create a config dict similar to magic_baseline_dir.py
                config_dict = {
                    "batch_size": batch_size,
                    "max_epochs": max_epochs,
                    "early_stopping_patience": early_stopping_patience,
                    "seed": seed,
                    "gpus": gpus,
                    "precision": precision,
                    "fast_dev_run": fast_dev_run,
                    "total_params": total_params,
                    "trainable_params": trainable_params,
                }
                wandb_logger.experiment.config.update(config_dict)
                wandb_logger.experiment.config.update(model_config.as_dict())
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
    }
    
    # Training configuration now uses the checkpoint_path and resume flags
    
    # Train the model
    logger.info("Starting training")
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
    
    # Save state dict (version-safe method)
    state_dict_path = os.path.join(output_dir, "state_dict.pth")
    model_config_save_path = os.path.join(output_dir, "final_model_config.yml")
    model.save_state_dict(state_dict_path)
    model.save_config(model_config_save_path)
    logger.info(f"State dict saved to {state_dict_path}")
    logger.info(f"Final model config saved to {model_config_save_path}")
    
    # Get predictions
    logger.info("Generating predictions on test set")
    try:
        # Get target labels from model tasks
        if hasattr(model, '_tasks') and model._tasks:
            additional_attributes = [
                target for task in model._tasks for target in task._target_labels
            ] + ["event_id"]
        else:
            # Fallback to common attributes
            additional_attributes = [
                "event_no",
                "mc_direction_x",
                "mc_direction_y", 
                "mc_direction_z",
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
        
        # Basic performance metrics
        # Calculate angular error
        pred_directions = torch.tensor(results[["direction_x_pred", "direction_y_pred", "direction_z_pred"]].values)
        true_directions = torch.tensor(results[["mc_direction_x", "mc_direction_y", "mc_direction_z"]].values)
        
        # Normalize directions
        pred_directions = pred_directions / torch.norm(pred_directions, dim=1, keepdim=True)
        true_directions = true_directions / torch.norm(true_directions, dim=1, keepdim=True)
        
        # Calculate angular errors in degrees
        dot_products = torch.sum(pred_directions * true_directions, dim=1)
        dot_products = torch.clamp(dot_products, -1.0, 1.0)  # Avoid numerical issues
        angular_errors = torch.acos(torch.abs(dot_products)) * 180.0 / torch.pi
        
        median_error = torch.median(angular_errors).item()
        mean_error = torch.mean(angular_errors).item()
        std_error = torch.std(angular_errors).item()
        
        logger.info(f"Validation angular error - Median: {median_error:.3f}°, Mean: {mean_error:.3f}°, Std: {std_error:.3f}°")
        
        # Save metrics
        metrics = {
            "median_angular_error_deg": median_error,
            "mean_angular_error_deg": mean_error,
            "std_angular_error_deg": std_error,
        }
        
        import json
        metrics_path = os.path.join(output_dir, "validation_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Validation metrics saved to {metrics_path}")
        
    except Exception as e:
        logger.warning(f"Failed to generate predictions: {e}")
    
    logger.info("Training script completed successfully")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = ArgumentParser(
        description="Train MAGIC IceMix model for direction reconstruction"
    )
    
    parser.add_argument(
        "--model-config",
        type=str,
        required=True,
        help="Path to model configuration YAML file",
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
    
    args = parser.parse_args()
    
    # Run training
    main(
        model_config_path=args.model_config,
        dataset_config_path=args.dataset_config,
        output_dir=args.output_dir,
        gpus=args.gpus,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_id=args.wandb_run_id,
        early_stopping_patience=args.early_stopping_patience,
        seed=args.seed,
        checkpoint_backbone_only=args.checkpoint_backbone_only,
        precision=args.precision,
        fast_dev_run=args.fast_dev_run,
        checkpoint_path=args.checkpoint_path,
        resume=args.resume,
    )
