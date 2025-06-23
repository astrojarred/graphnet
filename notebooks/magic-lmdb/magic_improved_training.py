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
    wandb_project: str = "gnn-classification-improved",
    precision: str = "16-mixed",  # UPGRADED: Enable AMP for speed
) -> None:
    """Run improved training with better early stopping and AMP."""
    # Construct Logger
    logger = Logger()

    # Initialise Weights & Biases (W&B) run
    if wandb:
        wandb_dir = "./wandb/"
        Path(wandb_dir).mkdir(parents=True, exist_ok=True)
        wandb_logger = WandbLogger(
            project=wandb_project,
            entity="max-planck",
            save_dir=wandb_dir,
            log_model=True,
        )

    # Build model
    model_config = ModelConfig.load(model_config_path)
    model: StandardModel = StandardModel.from_config(model_config, trust=True)

    # IMPROVED: Configuration with better early stopping
    config = TrainingConfig(
        target=[
            target for task in model._tasks for target in task._target_labels
        ],
        early_stopping_patience=early_stopping_patience,  # Now using 10 (recommended 8-10)
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

    run_name = f"magic_improved_classification_{config.target}"

    # Construct dataloaders
    dataset_config = DatasetConfig.load(dataset_config_path)
    datasets: Dataset = Dataset.from_config(dataset_config)

    train_dataset = EnsembleDataset(
        [datasets[key] for key in datasets if key.startswith("train")]
    )
    valid_dataset = EnsembleDataset(
        [datasets[key] for key in datasets if key.startswith("validation")]
    )
    test_dataset = EnsembleDataset(
        [datasets[key] for key in datasets if key.startswith("test")]
    )

    # Debug info
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(valid_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")
    
    # Estimate training steps for scheduler validation
    steps_per_epoch = len(train_dataset) // batch_size
    total_steps = steps_per_epoch * max_epochs
    logger.info(f"Estimated steps per epoch: {steps_per_epoch}")
    logger.info(f"Estimated total steps: {total_steps}")

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
    if wandb and rank_zero_only.rank == 0:
        logger.info("Logging config to W&B")
        wandb_logger.experiment.config.update(config)
        wandb_logger.experiment.config.update(model_config.as_dict())
        wandb_logger.experiment.config.update(dataset_config.as_dict())

    # IMPROVED: Training with better settings
    model.fit(
        train_dataloaders,
        valid_dataloaders,
        early_stopping_patience=config.early_stopping_patience,
        logger=wandb_logger if wandb else None,
        gradient_clip_val=1.0,
        **config.fit,
    )

    # Save model
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

    results = model.predict_as_dataframe(
        test_dataloaders,
        additional_attributes=additional_attributes + ["event_id"],
        gpus=config.fit["gpus"],
    )
    results.to_csv(str(path / "results.csv"))


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Train improved GNN classification model for MAGIC data."
    )

    parser.with_standard_arguments(
        "dataset-config",
        "model-config", 
        "gpus",
        ("max-epochs", 50),  # INCREASED: Allow for longer training
        ("early-stopping-patience", 10),  # IMPROVED: Better patience
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
        help="Name addition to folder",
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
        default="gnn-classification-improved",
    )

    parser.add_argument(
        "--precision",
        type=str,
        help="Training precision (16-mixed for AMP)",
        default="16-mixed",
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
        args.precision,
    )
