from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only

from graphnet.data.constants import FEATURES, TRUTH
from graphnet.data.dataloader import DataLoader
from graphnet.data.dataset import Dataset
from graphnet.data.dataset.dataset import EnsembleDataset
from graphnet.models import StandardModel
from graphnet.utilities.argparse import ArgumentParser
from graphnet.utilities.config import (
    DatasetConfig,
    ModelConfig,
    TrainingConfig,
)
from graphnet.utilities.logging import Logger

# Constants
features = FEATURES.MAGIC
truth = TRUTH.MAGIC
torch.set_float32_matmul_precision('high')

def main(
    dataset_config_path: str,
    model_config_path: str,
    output_dir: str,
    gpus: Optional[List[int]],
    max_epochs: int,
    early_stopping_patience: int,
    batch_size: int,
    num_workers: int,
    suffix: Optional[str] = None,
    wandb: bool = False,
    wandb_project: str = "gnn-direction-a",
    wandb_run_id: Optional[str] = None,
    precision: str = "32-true",
    checkpoint_path: Optional[str] = None,
    checkpoint_backbone_only: bool = False,
    resume: bool = False,
    fast_dev_run: bool = False,
) -> None:
    """Run example."""
    # Construct Logger
    logger = Logger()

    # Initialise Weights & Biases (W&B) run
    if wandb:
        # Make sure W&B output directory exists
        wandb_dir = "./wandb/"
        Path(wandb_dir).mkdir(parents=True, exist_ok=True)
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

    # Build model
    model_config = ModelConfig.load(model_config_path)
    model: StandardModel = StandardModel.from_config(model_config, trust=True)

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

    # Configuration
    config = TrainingConfig(
        target=[
            target for task in model._tasks for target in task._target_labels
        ],
        early_stopping_patience=early_stopping_patience,
        fit={
            "gpus": gpus,
            "max_epochs": max_epochs,
            "precision": precision,  # Enable AMP
        },
        dataloader={"batch_size": batch_size, "num_workers": num_workers},
    )

    if suffix is not None:
        archive = Path(output_dir) / f"train_model_{suffix}"
    else:
        archive = Path(output_dir) / "train_model"

    run_name = f"magic_baseline_direction_{config.target}"

    # Construct dataloaders
    dataset_config = DatasetConfig.load(dataset_config_path)
    datasets: Dataset = Dataset.from_config(
        dataset_config,
    )

    # Construct datasets from multiple selections
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
    dataloader_config = config.dataloader.copy()
    if dataloader_config.get("num_workers", 0) == 0:
        dataloader_config["prefetch_factor"] = None
        dataloader_config["persistent_workers"] = False
    
    train_dataloaders = DataLoader(
        train_dataset, shuffle=True, **dataloader_config
    )
    valid_dataloaders = DataLoader(
        valid_dataset, shuffle=False, **dataloader_config
    )
    test_dataloaders = DataLoader(
        test_dataset, shuffle=False, **dataloader_config
    )

    # Log configurations to W&B
    # NB: Only log to W&B on the rank-zero process in case of multi-GPU
    #     training.
    if wandb and rank_zero_only.rank == 0:
        if not wandb_logger.experiment.resumed:
            logger.info("Logging config to W&B")
            wandb_logger.experiment.config.update(config)
            wandb_logger.experiment.config.update(model_config.as_dict())
            wandb_logger.experiment.config.update(dataset_config.as_dict())
        else:
            logger.info(f"Resuming training from checkpoint {checkpoint_path}")

    # Training model
    model.fit(
        train_dataloaders,
        valid_dataloaders,
        early_stopping_patience=config.early_stopping_patience,
        logger=wandb_logger if wandb else None,
        gradient_clip_val=1.0,  # Add gradient clipping
        fast_dev_run=fast_dev_run,
        ckpt_path=checkpoint_path if resume else None,
        **config.fit,
    )

    # save model to file
    # Save model to file
    db_name = dataset_config.path.split("/")[-1].split(".")[0]
    path = archive / db_name / run_name
    path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing results to {path}")
    model.save_state_dict(str(path / "state_dict.pth"))
    model.save(str(path / "model.pth"))

    # Get predictions
    if isinstance(config.target, str):
        additional_attributes = [config.target]
    else:
        additional_attributes = config.target

    logger.info(f"config.target: {config.target}")
    logger.info(f"prediction_columns: {model.prediction_labels}")
    logger.info(f"additional_attributes: {additional_attributes}")

    results = model.predict_as_dataframe(
        test_dataloaders,
        additional_attributes=additional_attributes + ["event_id"],
        gpus=config.fit["gpus"],
    )
    results.to_csv(str(path / "results.csv"))


if __name__ == "__main__":
    # Parse command-line arguments
    parser = ArgumentParser(
        description="""
            Train GNN direction reconstruction model for MAGIC data.
            """
    )

    parser.with_standard_arguments(
        "dataset-config",
        "model-config",
        "gpus",
        ("max-epochs", 1),
        ("early-stopping-patience", 10),
        ("batch-size", 16),
        "num-workers",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory",
        default="./results",
    )

    parser.add_argument(
        "--suffix",
        type=str,
        help="Name addition to folder (default: %(default)s)",
        default=None,
    )

    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log to Weights & Biases",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        help="Weights & Biases project",
        default="gnn-direction-a",
    )

    parser.add_argument(
        "--wandb-run-id",
        type=str,
        help="Weights & Biases run ID",
        default=None,
    )
    
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        help="Path to checkpoint file to resume from",
        default=None,
    )

    parser.add_argument(
        "--checkpoint-backbone-only",
        action="store_true",
        help="Only load the backbone weights from the checkpoint",
        default=False,
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the checkpoint",
        default=False,
    )

    parser.add_argument(
        "--precision",
        type=str,
        help="Training precision",
        default="32-true",
    )

    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Run a single batch for debugging",
    )


    args, unknown = parser.parse_known_args()

    main(
        args.dataset_config,
        args.model_config,
        args.output_dir,
        args.gpus,
        args.max_epochs,
        args.early_stopping_patience,
        args.batch_size,
        args.num_workers,
        args.suffix,
        args.wandb,
        args.wandb_project,
        args.wandb_run_id,
        args.precision,
        args.checkpoint_path,
        args.checkpoint_backbone_only,
        args.resume,
        args.fast_dev_run,
    )
