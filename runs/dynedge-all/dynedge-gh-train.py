"""Multi-class classification using DynEdge from pre-defined config files.

This script trains models for multi-task learning including classification,
energy reconstruction, and direction reconstruction using DynEdge architecture.
"""

import os
import torch
from pathlib import Path
from typing import List, Optional, Dict, Any

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.callbacks import ModelCheckpoint

from graphnet.data.dataset.dataset import EnsembleDataset
from graphnet.data.dataloader import DataLoader
from graphnet.data.dataset import Dataset
from graphnet.models import StandardModel
from graphnet.utilities.argparse import ArgumentParser
from graphnet.utilities.config import (
    DatasetConfig,
    ModelConfig,
)
from graphnet.utilities.logging import Logger

# Set matmul precision for better performance
torch.set_float32_matmul_precision("high")


def main(
    dataset_config_path: str,
    output_dir: str,
    model_config_path: str,
    gpus: Optional[List[int]] = None,
    max_epochs: int = 10,
    early_stopping_patience: int = 5,
    batch_size: int = 32,
    num_workers: int = 10,
    wandb: bool = False,
    wandb_project: str = "dynedge-training",
    wandb_entity: str = "magic-graphnet-team",
    wandb_run_id: Optional[str] = None,
    seed: int = 42,
    checkpoint_path: Optional[str] = None,
    checkpoint_backbone_only: bool = False,
    resume: bool = False,
    precision: str = "32-true",
    fast_dev_run: bool = False,
    accumulate_grad_batches: int = 1,
    limit_train_batches: Optional[float] = None,
    limit_val_batches: Optional[float] = None,
    val_check_interval: Optional[float] = None,
    gradient_clip_val: float = 0.0,
    distribution_strategy: str = "ddp_find_unused_parameters_true",
    save_top_k: int = 2,
    save_every_n_epochs: int = 1,
) -> None:
    """Train DynEdge model for multi-task learning.

    Args:
        dataset_config_path: Path to dataset configuration YAML file
        output_dir: Directory to save results and model
        model_config_path: Path to model configuration YAML file
        gpus: List of GPU IDs to use for training
        max_epochs: Maximum number of training epochs
        early_stopping_patience: Patience for early stopping
        batch_size: Training batch size
        num_workers: Number of data loading workers
        wandb: Whether to use Weights & Biases logging
        wandb_project: W&B project name
        wandb_entity: W&B entity name
        wandb_run_id: W&B run ID for resuming
        seed: Random seed for reproducibility
        checkpoint_path: Path to checkpoint file
        checkpoint_backbone_only: Only load backbone weights from checkpoint
        resume: Resume training from checkpoint
        precision: Training precision
        fast_dev_run: Run single batch for debugging
        accumulate_grad_batches: Number of gradient accumulation steps
        limit_train_batches: Proportion or count of training batches per epoch
        limit_val_batches: Proportion or count of validation batches per epoch
        val_check_interval: How frequently to run validation within an epoch
        gradient_clip_val: The value to clip gradients at
        distribution_strategy: Strategy for distributed training
        save_top_k: Number of best models to save
        save_every_n_epochs: Save checkpoint every n epochs
    """
    # Construct Logger
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
            entity=wandb_entity,
            id=wandb_run_id,
            save_dir=wandb_dir,
            log_model=True,
            resume="allow",
        )

    # Build model
    logger.info(f"Loading model from config {model_config_path}")
    model_config = ModelConfig.load(model_config_path)
    model: StandardModel = StandardModel.from_config(model_config, trust=True)

    # Load checkpoint if provided
    if checkpoint_path is not None:
        if Path(checkpoint_path).exists():
            logger.info(f"Loading checkpoint from {checkpoint_path}")

            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )

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
                model.load_state_dict(
                    state_dict, strict=True if not checkpoint_backbone_only else False
                )
                logger.info(
                    f"✓ Checkpoint weights loaded successfully (strict={not checkpoint_backbone_only})"
                )
            except RuntimeError as e:
                logger.warning(f"Strict loading failed: {e}")
                missing_keys, unexpected_keys = model.load_state_dict(
                    state_dict, strict=False
                )
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

    # Extract target labels from model tasks
    target_labels = [target for task in model._tasks for target in task._target_labels]
    logger.info(f"Target labels: {target_labels}")

    # Construct datasets
    logger.info("Loading datasets...")
    datasets: Dict[str, Any] = Dataset.from_config(dataset_config)

    # Construct datasets from multiple selections
    train_dataset = EnsembleDataset(
        [datasets[key] for key in datasets if key.startswith("train")]
    )
    valid_dataset = EnsembleDataset(
        [datasets[key] for key in datasets if key.startswith("valid")]
    )
    test_dataset = EnsembleDataset(
        [datasets[key] for key in datasets if key.startswith("test")]
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

    train_dataloader = DataLoader(train_dataset, shuffle=True, **dataloader_config)
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, **dataloader_config)
    test_dataloader = DataLoader(test_dataset, shuffle=False, **dataloader_config)

    # Log configurations to W&B
    if wandb and wandb_logger:
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
                "target_labels": target_labels,
                "model_config_path": model_config_path,
            }
            wandb_logger.experiment.config.update(config_dict)
            wandb_logger.experiment.config.update(model_config.as_dict())
            wandb_logger.experiment.config.update(dataset_config.as_dict())

    # Prepare callbacks list
    callbacks = []

    # Add checkpoint callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=save_top_k,
        save_last=True,
    )
    epoch_checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,
        every_n_epochs=save_every_n_epochs,
        filename="epoch-{epoch:02d}",
    )
    callbacks.append(checkpoint_callback)
    callbacks.append(epoch_checkpoint_callback)

    # Training configuration
    fit_config = {
        "gpus": gpus,
        "max_epochs": max_epochs,
        "distribution_strategy": distribution_strategy,
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

    # Training model
    logger.info("Starting training...")
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        model.fit(
            train_dataloader,
            valid_dataloader,
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

    # Get predictions
    logger.info("Generating predictions on test set...")
    try:
        if isinstance(target_labels, str):
            additional_attributes = [target_labels]
        else:
            additional_attributes = target_labels

        logger.info(f"Target labels: {target_labels}")
        logger.info(f"Prediction columns: {model.prediction_labels}")

        results = model.predict_as_dataframe(
            test_dataloader,
            additional_attributes=additional_attributes + ["event_id"],
            gpus=gpus,
        )

        # Save predictions
        results_path = os.path.join(output_dir, "results.csv")
        results.to_csv(results_path, index=False)
        logger.info(f"Test predictions saved to {results_path}")
    except Exception as e:
        logger.warning(f"Failed to generate predictions: {e}")

    logger.info("DynEdge training script completed successfully!")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = ArgumentParser(description="Train DynEdge model for multi-task learning")

    # Required arguments
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
        default="dynedge-training",
        help="W&B project name (default: %(default)s)",
    )

    parser.add_argument(
        "--wandb-entity",
        type=str,
        default="magic-graphnet-team",
        help="W&B entity name (default: %(default)s)",
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
        "--checkpoint-path",
        type=str,
        default=None,
        help="Path to checkpoint file",
    )

    parser.add_argument(
        "--checkpoint-backbone-only",
        action="store_true",
        help="Only load the backbone weights from the checkpoint",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from checkpoint",
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
        "--distribution-strategy",
        type=str,
        default="ddp_find_unused_parameters_true",
        help="Strategy for distributed training (default: %(default)s)",
    )

    parser.add_argument(
        "--save-top-k",
        type=int,
        default=2,
        help="Number of best models to save (default: %(default)s)",
    )

    parser.add_argument(
        "--save-every-n-epochs",
        type=int,
        default=1,
        help="Save checkpoint every n epochs (default: %(default)s)",
    )

    args = parser.parse_args()

    # Adjust argument names for function call
    args.dataset_config_path = args.dataset_config
    args.model_config_path = args.model_config
    del args.dataset_config, args.model_config

    # Run training
    main(**vars(args))
