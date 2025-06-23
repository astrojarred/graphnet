from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
import atexit
import signal
import sys
import gc

from graphnet.data.constants import FEATURES, TRUTH
from graphnet.data.dataloader import DataLoader
from graphnet.data.dataset import Dataset
from graphnet.data.dataset.dataset import EnsembleDataset
from graphnet.models import StandardModel
from graphnet.models.graphs import KNNGraph
from graphnet.models.detector.magic import MAGICDetectorFixed
from graphnet.models.graphs.nodes import NodesAsPulses
from graphnet.utilities.argparse import ArgumentParser
from graphnet.utilities.config import (
    DatasetConfig,
    ModelConfig,
    TrainingConfig,
)
from graphnet.utilities.logging import Logger

# Import MAGIC-specific components
# You'll need to ensure these are importable
# from graphnet.models.detector import MAGICDetectorFixed  # Assuming this is in magic.py
from graphnet.models.gnn import (
    MAGICTransformer,
    MAGICDirectionClassifier, 
    MAGICHybridModel,
    # MAGICDirectionReconstructionVMF,
    # VMFLoss,
    # FocalLoss,
    # CombinedVMFClassificationLoss,
    # angles_to_direction_vector
)

from graphnet.models.task.magic_reconstruction import (
    MAGICDirectionReconstructionVMF,
    MAGICDirectionClassification,
    MAGICHybridDirectionTask,
    MAGICAngularResolution,
)

from graphnet.training.loss_functions import (
    MAGICFocalLoss,
    MAGICVMFLoss,
    CombinedVMFClassificationLoss,
)

# Constants
features = FEATURES.MAGIC
truth = TRUTH.MAGIC
torch.set_float32_matmul_precision('high')


def cleanup_cuda_memory():
    """Comprehensive CUDA memory cleanup."""
    try:
        if torch.cuda.is_available():
            print("🧹 Cleaning up CUDA memory...")
            
            # Clear PyTorch cache
            torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
            # Reset CUDA memory stats
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
            
            # Synchronize CUDA operations
            torch.cuda.synchronize()
            
            # Get memory info
            if torch.cuda.device_count() > 0:
                for i in range(torch.cuda.device_count()):
                    memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
                    memory_reserved = torch.cuda.memory_reserved(i) / 1024**3   # GB
                    print(f"  GPU {i}: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
            
            print("✅ CUDA memory cleanup completed")
    except Exception as e:
        print(f"⚠️  CUDA cleanup warning: {e}")


def signal_handler(signum, frame):
    """Handle interrupt signals and clean up memory."""
    print(f"\n🛑 Received signal {signum}, cleaning up...")
    cleanup_cuda_memory()
    print("👋 Exiting gracefully")
    sys.exit(0)


def setup_cleanup_handlers():
    """Set up automatic cleanup on exit and interrupt."""
    # Register cleanup function to run on normal exit
    atexit.register(cleanup_cuda_memory)
    
    # Register signal handlers for interrupts
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination
    
    print("🛡️  CUDA memory cleanup handlers registered")


def create_model(
    model_type: str,
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
    # Training hyperparameters (for optimizer/scheduler)
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-4,
    scheduler_t0: int = 5000,
) -> StandardModel:
    """Create MAGIC direction reconstruction model directly.
    
    Args:
        model_type: One of "transformer", "classifier", or "hybrid"
        hidden_dim: Hidden dimension size for model layers
        num_layers: Number of transformer/model layers
        Other args: Model-specific parameters
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for regularization
        scheduler_t0: T0 parameter for cosine annealing scheduler
    
    Returns:
        StandardModel instance with configured optimizer and scheduler
    """
    print(f"DEBUG: Creating model with type '{model_type}'")
    
    # Create graph definition
    graph_definition = KNNGraph(
        detector=MAGICDetectorFixed(),
        nb_nearest_neighbours=nb_nearest_neighbours,
        node_definition=NodesAsPulses(),
        columns=[0, 1, 2],  # x_cam, y_cam, t for k-NN construction
    )
    
    # Create backbone based on model type
    if model_type == "transformer":
        print("DEBUG: Creating transformer model")
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
        
        # Single task for transformer
        vmf_loss = MAGICVMFLoss(prediction_kappa_index=3)
        print(f"DEBUG: Using loss function: {type(vmf_loss).__name__}")
        task = MAGICDirectionReconstructionVMF(
            hidden_size=backbone.nb_outputs,
            loss_function=vmf_loss,
        )
        print(f"DEBUG: Task loss function: {type(task._loss_function).__name__}")
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
        
        # Multiple tasks for classifier
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
            transformer_layers=num_layers // 2,  # Fewer layers for hybrid
            transformer_heads=num_heads,
            num_fine_bins=num_fine_bins,
            roi_radius=roi_radius,
            ensemble_method=ensemble_method,
        )
        
        # Combined task for hybrid - use simple VMF loss for now
        task = MAGICDirectionReconstructionVMF(
            hidden_size=backbone.nb_outputs,
            loss_function=MAGICVMFLoss(prediction_kappa_index=3),
        )
        tasks = [task]
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create optimizer and scheduler with configurable hyperparameters
    # CRITICAL: Much lower learning rates after fixing VMF loss
    optimizer_class = torch.optim.AdamW
    optimizer_kwargs = {
        "lr": learning_rate,
        "weight_decay": weight_decay,
        "eps": 1e-08,
        "betas": (0.9, 0.999),
    }
    
    # Use cosine annealing with warm restarts for all model types (most stable)
    scheduler_class = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
    scheduler_kwargs = {
        "T_0": scheduler_t0,
        "T_mult": 1,    # Fixed cycle length
        "eta_min": learning_rate / 100,  # Minimum LR = 1% of initial LR
    }
    scheduler_config = {"interval": "step"}
    
    # Create StandardModel
    model = StandardModel(
        graph_definition=graph_definition,
        backbone=backbone,
        tasks=tasks,
        optimizer_class=optimizer_class,
        optimizer_kwargs=optimizer_kwargs,
        scheduler_class=scheduler_class,
        scheduler_kwargs=scheduler_kwargs,
        scheduler_config=scheduler_config,
    )
    
    return model


def main(
    dataset_config_path: str,
    model_type: str,
    output_dir: str,
    gpus: Optional[List[int]],
    max_epochs: int,
    early_stopping_patience: int,
    batch_size: int,
    num_workers: int,
    # Model parameters
    hidden_dim: int = 256,
    num_layers: int = 8,
    num_heads: int = 8,
    mlp_ratio: int = 4,
    dropout: float = 0.1,
    num_fine_bins: int = 128,
    roi_radius: float = 0.5,
    ensemble_method: str = "attention",
    nb_nearest_neighbours: int = 16,
    # Training hyperparameters - CRITICAL: Conservative defaults after VMF fix
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-4,
    scheduler_t0: int = 5000,
    gradient_clip_val: float = 1.0,
    # Training parameters
    suffix: Optional[str] = None,
    wandb: bool = False,
    wandb_project: str = "gnn-direction-b",
    wandb_run_id: Optional[str] = None,
    precision: str = "32-true",
    checkpoint_path: Optional[str] = None,
    checkpoint_backbone_only: bool = False,
    resume: bool = False,
    fast_dev_run: bool = False,
    gradient_accumulation_steps: int = 1,
) -> None:
    """Run MAGIC direction reconstruction training."""
    # Set up CUDA memory cleanup handlers
    setup_cleanup_handlers()
    
    # Construct Logger
    logger = Logger()
    logger.info(f"Training {model_type} model for MAGIC direction reconstruction")

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
            name=f"magic_{model_type}_direction",
        )

    # Build model directly
    logger.info("Building model directly (not from config)")
    model = create_model(
        model_type=model_type,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        num_fine_bins=num_fine_bins,
        roi_radius=roi_radius,
        ensemble_method=ensemble_method,
        nb_nearest_neighbours=nb_nearest_neighbours,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        scheduler_t0=scheduler_t0,
    )
    
    # Log model info
    logger.info(f"Model type: {model_type}")
    logger.info(f"Backbone: {model.backbone.__class__.__name__}")
    logger.info(f"Tasks: {[task.__class__.__name__ for task in model._tasks]}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

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

    # Configuration
    config = TrainingConfig(
        target=[
            target for task in model._tasks for target in task._target_labels
        ],
        early_stopping_patience=early_stopping_patience,
        fit={
            "gpus": gpus,
            "max_epochs": max_epochs,
            "precision": precision,
            "accumulate_grad_batches": gradient_accumulation_steps,
        },
        dataloader={"batch_size": batch_size, "num_workers": num_workers},
    )

    if suffix is not None:
        archive = Path(output_dir) / f"train_{model_type}_{suffix}"
    else:
        archive = Path(output_dir) / f"train_{model_type}"

    run_name = f"magic_{model_type}_direction_{config.target}"

    # Construct datasets
    dataset_config = DatasetConfig.load(dataset_config_path)
    datasets: Dataset = Dataset.from_config(dataset_config)

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
    
    # Update scheduler total_steps for OneCycle if needed
    if model_type == "transformer" and "total_steps" in model._scheduler_kwargs:
        steps_per_epoch = len(train_dataset) // (batch_size * gradient_accumulation_steps)
        total_steps = steps_per_epoch * max_epochs
        model._scheduler_kwargs["total_steps"] = total_steps
        logger.info(f"Updated OneCycle scheduler total_steps to {total_steps}")

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
        if not wandb_logger.experiment.resumed:
            logger.info("Logging config to W&B")
            wandb_logger.experiment.config.update({
                "model_type": model_type,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "num_heads": num_heads,
                "mlp_ratio": mlp_ratio,
                "dropout": dropout,
                "num_fine_bins": num_fine_bins,
                "roi_radius": roi_radius,
                "ensemble_method": ensemble_method,
                "nb_nearest_neighbours": nb_nearest_neighbours,
                "batch_size": batch_size,
                "effective_batch_size": batch_size * gradient_accumulation_steps,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "precision": precision,
                "max_epochs": max_epochs,
                # CRITICAL: Log the fixed hyperparameters
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "scheduler_t0": scheduler_t0,
                "gradient_clip_val": gradient_clip_val,
                "vmf_loss_fixed": True,  # Flag indicating we fixed the VMF loss
            })
            wandb_logger.experiment.config.update(config)
            wandb_logger.experiment.config.update(dataset_config.as_dict())
        else:
            logger.info(f"Resuming training from checkpoint {checkpoint_path}")

    # Handle distributed training strategy for unused parameters
    distribution_strategy = "ddp"
    if len(gpus or []) > 1:
        distribution_strategy = "ddp_find_unused_parameters_true"
    
    # Training model
    model.fit(
        train_dataloaders,
        valid_dataloaders,
        early_stopping_patience=config.early_stopping_patience,
        logger=wandb_logger if wandb else None,
        gradient_clip_val=gradient_clip_val,  # Use configurable gradient clipping
        fast_dev_run=fast_dev_run,
        ckpt_path=checkpoint_path if resume else None,
        distribution_strategy=distribution_strategy,
        **config.fit,
    )

    # Save model to file
    db_name = dataset_config.path.split("/")[-1].split(".")[0]
    path = archive / db_name / run_name
    path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing results to {path}")
    model.save_state_dict(str(path / "state_dict.pth"))
    model.save(str(path / "model.pth"))
    
    # Save model configuration for reference
    model_info = {
        "model_type": model_type,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "mlp_ratio": mlp_ratio,
        "dropout": dropout,
        "num_fine_bins": num_fine_bins,
        "roi_radius": roi_radius,
        "ensemble_method": ensemble_method,
        "nb_nearest_neighbours": nb_nearest_neighbours,
        "total_parameters": sum(p.numel() for p in model.parameters()),
    }
    import json
    with open(path / "model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)

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
    
    # Final cleanup
    logger.info("Training completed, performing final CUDA cleanup...")
    cleanup_cuda_memory()


if __name__ == "__main__":
    # Parse command-line arguments
    parser = ArgumentParser(
        description="""
            Train MAGIC direction reconstruction model with IceCube-inspired architectures.
            """
    )

    parser.with_standard_arguments(
        "dataset-config",
        "gpus",
        ("max-epochs", 50),
        ("early-stopping-patience", 10),
        ("batch-size", 256),
        "num-workers",
    )

    # Model selection
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["transformer", "classifier", "hybrid"],
        required=True,
        help="Type of model to train",
    )

    # Model parameters
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="Hidden dimension size (default: %(default)s)",
    )
    
    parser.add_argument(
        "--num-layers",
        type=int,
        default=8,
        help="Number of transformer/model layers (default: %(default)s)",
    )
    
    parser.add_argument(
        "--num-heads",
        type=int,
        default=8,
        help="Number of attention heads for transformer (default: %(default)s)",
    )
    
    parser.add_argument(
        "--mlp-ratio",
        type=int,
        default=4,
        help="MLP expansion ratio for transformer (default: %(default)s)",
    )
    
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate (default: %(default)s)",
    )
    
    parser.add_argument(
        "--num-fine-bins",
        type=int,
        default=128,
        help="Number of fine angular bins for classifier (default: %(default)s)",
    )
    
    parser.add_argument(
        "--roi-radius",
        type=float,
        default=0.5,
        help="Region of interest radius in degrees (default: %(default)s)",
    )
    
    parser.add_argument(
        "--ensemble-method",
        type=str,
        choices=["attention", "learned", "average"],
        default="attention",
        help="Ensemble method for hybrid model (default: %(default)s)",
    )
    
    parser.add_argument(
        "--nb-nearest-neighbours",
        type=int,
        default=16,
        help="Number of nearest neighbors for graph construction (default: %(default)s)",
    )

    # Training parameters
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
        default="gnn-direction-b",
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
        help="Training precision (16-mixed, 32-true, etc.)",
        default="32-true",
    )

    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Run a single batch for debugging",
    )
    
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps (default: %(default)s)",
    )

    # Training hyperparameters - CRITICAL after VMF loss fix
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate (MUCH LOWER after VMF fix) (default: %(default)s)",
    )
    
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for regularization (default: %(default)s)",
    )
    
    parser.add_argument(
        "--scheduler-t0",
        type=int,
        default=5000,
        help="T0 parameter for cosine annealing scheduler (default: %(default)s)",
    )
    
    parser.add_argument(
        "--gradient-clip-val",
        type=float,
        default=1.0,
        help="Gradient clipping value (default: %(default)s)",
    )

    args, unknown = parser.parse_known_args()

    try:
        main(
            args.dataset_config,
            args.model_type,
            args.output_dir,
            args.gpus,
            args.max_epochs,
            args.early_stopping_patience,
            args.batch_size,
            args.num_workers,
            # Model parameters
            args.hidden_dim,
            args.num_layers,
            args.num_heads,
            args.mlp_ratio,
            args.dropout,
            args.num_fine_bins,
            args.roi_radius,
            args.ensemble_method,
            args.nb_nearest_neighbours,
            # Training hyperparameters
            args.learning_rate,
            args.weight_decay,
            args.scheduler_t0,
            args.gradient_clip_val,
            # Training parameters
            args.suffix,
            args.wandb,
            args.wandb_project,
            args.wandb_run_id,
            args.precision,
            args.checkpoint_path,
            args.checkpoint_backbone_only,
            args.resume,
            args.fast_dev_run,
            args.gradient_accumulation_steps,
        )
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        print("🧹 Performing emergency CUDA cleanup...")
        cleanup_cuda_memory()
        raise  # Re-raise the exception after cleanup
