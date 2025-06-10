#!/usr/bin/env python3
"""Simple standalone multi-task GNN for MAGIC telescope data.

This script trains a baseline GNN for gamma-ray astronomy that can:
1. Classify gamma rays vs protons  
2. Reconstruct energy of gamma rays
3. Reconstruct arrival direction (theta, phi) of gamma rays

Example usage:
    python magic_baseline.py --path /path/to/data.lmdb --max-epochs 20
    
Resume from checkpoint:
    python magic_baseline.py --path /path/to/data.lmdb --resume-from-checkpoint /path/to/checkpoint.ckpt

Evaluate only (no training):
    python magic_baseline.py --path /path/to/data.lmdb --eval-only --checkpoint /path/to/model.pth
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import torch
from torch import Tensor
from torch.optim.adam import Adam
from pytorch_lightning.loggers import WandbLogger

from graphnet.data.constants import FEATURES, TRUTH
from graphnet.models import StandardModel
from graphnet.models.detector.magic import MAGICDetector
from graphnet.models.gnn import DynEdge
from graphnet.models.graphs import KNNGraph
from graphnet.models.task.classification import BinaryClassificationTask
from graphnet.models.task.reconstruction import (
    EnergyReconstruction, ZenithReconstruction, AzimuthReconstruction
)
from graphnet.models.task import StandardLearnedTask
from graphnet.training.loss_functions import (
    BinaryCrossEntropyLoss, LogCoshLoss, VonMisesFisher2DLoss
)
from graphnet.utilities.argparse import ArgumentParser
from graphnet.utilities.logging import Logger
from graphnet.data import GraphNeTDataModule
from graphnet.data.dataset.lmdb import MAGICLMDBDataset

# Import our evaluation utilities
from magic_evaluation_utils import (
    evaluate_magic_results_comprehensive,
    print_magic_summary
)

# Constants
features = FEATURES.MAGIC
truth = TRUTH.MAGIC
torch.set_float32_matmul_precision('high')

# NOTE: Your data preparation step should create a 'particle_id' column:
# particle_id = (particle_id == 0).float()  # 1.0 for gamma, 0.0 for proton
# This ensures reconstruction losses only apply to gamma events.


class MAGICEnergyReconstruction(StandardLearnedTask):
    """Energy reconstruction for MAGIC gamma-ray telescopes.
    
    Adapted for primary gamma-ray energy reconstruction (not deposited energy).
    Typically handles energies from ~10 GeV to ~100 TeV.
    """
    
    default_target_labels = ["true_energy"]
    default_prediction_labels = ["energy_pred"]
    nb_inputs = 1

    def _forward(self, x: Tensor) -> Tensor:
        # For gamma-ray energy: ensure positive domain
        # Using softplus with smaller beta for smoother gradients at low energies
        return torch.nn.functional.softplus(x, beta=0.1) + 1e-6


class MAGICDirectionReconstruction(StandardLearnedTask):
    """Direction reconstruction for MAGIC telescopes (zenith and azimuth).

    Outputs two values for zenith and azimuth, intended for use with 
    VonMisesFisher2DLoss which operates on 2D direction vectors on a sphere.
    
    MAGIC zenith angles are typically 5-35° from vertical.
    Azimuth covers full 360° range.
    """

    default_target_labels = ["true_theta", "true_phi"]
    default_prediction_labels = ["theta_pred", "phi_pred"]
    nb_inputs = 2  # Output both theta and phi

    def _forward(self, x: Tensor) -> Tensor:
        # Don't apply activation functions here - let VonMisesFisher2DLoss
        # handle the conversion to proper direction vectors
        # It works with raw network outputs (logits) to compute direction
        return x


def create_model():
    """Create the multi-task model."""
    # Graph definition
    graph_definition = KNNGraph(
        detector=MAGICDetector(),
        nb_nearest_neighbours=8,
        columns=[0, 1, 2],  # x_cam, y_cam, t
    )
    
    # Backbone GNN
    backbone = DynEdge(
        nb_inputs=graph_definition.nb_outputs,
        global_pooling_schemes=["min", "max", "mean", "sum"],
    )
    
    # Tasks
    tasks = [
        # 1. Binary classification (gamma vs proton)
        BinaryClassificationTask(
            hidden_size=backbone.nb_outputs,
            target_labels=["particle_id"],
            prediction_labels=["gamma_prob"],
            loss_function=BinaryCrossEntropyLoss(),
        ),
        # 2. Energy reconstruction (only for gamma events)
        MAGICEnergyReconstruction(
            hidden_size=backbone.nb_outputs,
            target_labels=["true_energy"],
            prediction_labels=["energy_pred"],
            loss_function=LogCoshLoss(),
            loss_weight="particle_id",  # Only compute loss for gamma events
        ),
        # 3. Direction reconstruction (zenith and azimuth, only for gamma events)
        MAGICDirectionReconstruction(
            hidden_size=backbone.nb_outputs,
            target_labels=["true_theta", "true_phi"],
            prediction_labels=["theta_pred", "phi_pred"],
            loss_function=VonMisesFisher2DLoss(),
            loss_weight="particle_id",  # Only compute loss for gamma events
        ),
    ]
    
    # Complete model
    model = StandardModel(
        graph_definition=graph_definition,
        backbone=backbone,
        tasks=tasks,
        optimizer_class=Adam,
        optimizer_kwargs={"lr": 1e-3, "eps": 1e-08},
    )
    
    return model


def setup_data(data_path, batch_size, num_workers):
    """Setup data loaders."""
    graph_definition = KNNGraph(
        detector=MAGICDetector(),
        nb_nearest_neighbours=8,
        columns=[0, 1, 2],
    )
    
    dataset_args = {
        "path": data_path,
        "features": features,
        "truth": truth + ["particle_id"],  # Add gamma mask to truth
        "graph_definition": graph_definition,
    }

    dm = GraphNeTDataModule(
        dataset_reference=MAGICLMDBDataset,
        dataset_args=dataset_args,
        train_dataloader_kwargs={
            "batch_size": batch_size,
            "num_workers": num_workers,
            "shuffle": True,
        },
        test_dataloader_kwargs={
            "batch_size": batch_size,
            "num_workers": num_workers,
        },
    )
    
    return dm



def evaluate_results(results_df, output_dir, logger, data_path, limits_file=None):
    """Evaluate results using comprehensive evaluation from magic_evaluation_utils."""
    return evaluate_magic_results_comprehensive(
        results_df, data_path, output_dir, logger, limits_file
    )


def main():
    """Main function."""
    parser = ArgumentParser(description="MAGIC baseline multi-task GNN")
    
    parser.add_argument("--path", required=True, help="Path to LMDB dataset")
    parser.add_argument("--output-dir", default="./magic_baseline_results", help="Output directory")
    parser.add_argument("--limits-file", type=str, help="Path to preprocessing limits file (optional - will auto-search in data directory)")
    parser.add_argument("--resume-from-checkpoint", type=str, help="Path to checkpoint to resume from")
    # parser.add_argument("--auto-resume", action="store_true", help="Automatically find and resume from latest checkpoint")
    parser.add_argument("--wandb-run-id", type=str, help="W&B run ID to resume (if resuming a W&B logged run)")
    
    # Evaluation-only mode
    parser.add_argument("--eval-only", action="store_true", help="Skip training and only run evaluation")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint for evaluation (required with --eval-only)")
    parser.add_argument("--use-test-data", action="store_true", help="Use test split instead of validation for evaluation")
    
    # Standard arguments
    parser.with_standard_arguments(
        "gpus", ("max-epochs", 50), ("early-stopping-patience", 10),
        ("batch-size", 128), "num-workers"
    )
    
    # W&B
    parser.add_argument("--wandb", action="store_true", help="Use W&B logging")
    parser.add_argument("--wandb-project", default="magic-gnn", help="W&B project")
    parser.add_argument("--wandb-entity", default="max-planck", help="W&B entity")
    
    args, unknown = parser.parse_known_args()
    
    logger = Logger()
    
    # Validation for eval-only mode
    if args.eval_only and not args.checkpoint:
        logger.error("❌ --checkpoint is required when using --eval-only")
        return
    
    if args.eval_only and not os.path.exists(args.checkpoint):
        logger.error(f"❌ Checkpoint file not found: {args.checkpoint}")
        return
    
    # Check limits file and auto-search info
    if args.limits_file and not os.path.exists(args.limits_file):
        logger.warning(f"⚠️  Specified limits file not found: {args.limits_file}")
        logger.info("Will auto-search in data directory instead...")
    elif args.limits_file:
        logger.info(f"📄 Using explicit limits file: {args.limits_file}")
    else:
        logger.info(f"🔍 Will auto-search for preprocessing limits in: {args.path}")
        logger.info("   (Looking for: preprocessing_limits.pkl, limits.pkl, etc.)")
    
    # Handle checkpoint resumption
    checkpoint_path = None
    wandb_run_id = args.wandb_run_id
    
    if args.resume_from_checkpoint:
        checkpoint_path = args.resume_from_checkpoint
        logger.info(f"🔄 Resuming from checkpoint: {checkpoint_path}")
    
    # W&B setup
    wandb_logger = None
    if args.wandb and not args.eval_only:  # Skip W&B for eval-only mode
        wandb_dir = os.path.join(args.output_dir, "wandb")
        os.makedirs(wandb_dir, exist_ok=True)
        
        wandb_kwargs = {
            "project": args.wandb_project,
            "entity": args.wandb_entity,
            "save_dir": wandb_dir,
            "log_model": True,
        }
        
        # If resuming, add the run ID and resume mode
        if wandb_run_id:
            wandb_kwargs["id"] = wandb_run_id
            wandb_kwargs["resume"] = "must"  # "must" ensures it fails if run doesn't exist
            logger.info(f"🔄 Resuming W&B run: {wandb_run_id}")
        
        wandb_logger = WandbLogger(**wandb_kwargs)
    
    logger.info("🚀 MAGIC Baseline Multi-task GNN")
    logger.info(f"📁 Data: {args.path}")
    logger.info(f"📁 Output: {args.output_dir}")
    
    if args.eval_only:
        logger.info(f"🔍 Mode: EVALUATION ONLY")
        logger.info(f"📄 Checkpoint: {args.checkpoint}")
        logger.info(f"📊 Data split: {'Test' if args.use_test_data else 'Validation'}")
    else:
        logger.info(f"🎯 Tasks: Classification + Energy + Direction (3 tasks total)")
    
    try:
        # Setup
        dm = setup_data(args.path, args.batch_size, args.num_workers)
        
        if args.eval_only:
            # Create model and load checkpoint for evaluation only
            logger.info("Creating model...")
            model = create_model()
            logger.info("Loading weights from checkpoint...")
            
            # Load checkpoint
            checkpoint = torch.load(args.checkpoint, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("✅ Model loaded successfully")
            
            # Select dataloader
            if args.use_test_data:
                eval_dataloader = dm.test_dataloader
                split_name = "test"
            else:
                eval_dataloader = dm.val_dataloader
                split_name = "validation"
            
            # Create output directory
            os.makedirs(args.output_dir, exist_ok=True)
            
            # Predict
            logger.info(f"Getting predictions on {split_name} data...")
            predictions = model.predict_as_dataframe(
                eval_dataloader,
                gpus=args.gpus,
            )
            
            # Save predictions
            predictions.to_csv(f"{args.output_dir}/predictions_{split_name}.csv", index=False)
            logger.info(f"💾 Predictions saved to: predictions_{split_name}.csv")
            
            # For evaluation, we need to manually iterate through the data to get both predictions and truth
            logger.info("Generating detailed evaluation data...")
            eval_results = []
            model.eval()
            
            # Move model to GPU if available
            device = torch.device(f"cuda:{args.gpus[0]}" if args.gpus else "cpu")
            model = model.to(device)
            
            with torch.no_grad():
                for i, batch in enumerate(eval_dataloader):
                    # Move batch to device
                    batch = batch.to(device)
                    
                    # Get predictions
                    preds = model(batch)
                    
                    # Extract predictions (preds is a list of tensors, one per task)
                    # Order: [classification, energy, direction]
                    gamma_prob = torch.sigmoid(preds[0]).cpu().numpy()  # Classification task
                    energy_pred = preds[1].cpu().numpy()  # Energy task
                    direction_pred = preds[2].cpu().numpy()  # Direction task (theta, phi)
                    
                    # Split direction into theta and phi
                    theta_pred = direction_pred[:, 0]  # First column is theta
                    phi_pred = direction_pred[:, 1]    # Second column is phi
                    
                    # Extract truth values from batch
                    particle_id = batch.particle_id.cpu().numpy()
                    true_energy = batch.true_energy.cpu().numpy()
                    true_theta = batch.true_theta.cpu().numpy()
                    true_phi = batch.true_phi.cpu().numpy()
                    
                    # Combine results
                    for j in range(len(particle_id)):
                        eval_results.append({
                            'gamma_prob': gamma_prob[j],
                            'energy_pred': energy_pred[j],
                            'theta_pred': theta_pred[j],
                            'phi_pred': phi_pred[j],
                            'particle_id': particle_id[j],
                            'true_energy': true_energy[j],
                            'true_theta': true_theta[j],
                            'true_phi': true_phi[j],
                        })
            
            # Convert to DataFrame
            results = pd.DataFrame(eval_results)
            results.to_csv(f"{args.output_dir}/results_{split_name}.csv", index=False)
            logger.info(f"💾 Evaluation results saved to: results_{split_name}.csv")
            
            # Evaluate with transforms
            evaluate_results(results, args.output_dir, logger, args.path, args.limits_file)
            
            logger.info(f"✅ Evaluation complete! Results in: {args.output_dir}")
            
        else:
            # Normal training mode
            model = create_model()
            
            # Train
            if checkpoint_path:
                logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
            else:
                logger.info("Starting fresh training...")
                
            model.fit(
                dm.train_dataloader,
                dm.val_dataloader,
                early_stopping_patience=args.early_stopping_patience,
                logger=wandb_logger,
                gradient_clip_val=1.0,
                gpus=args.gpus,
                max_epochs=args.max_epochs,
                ckpt_path=checkpoint_path,  # This is the key parameter for resumption
            )
            
            # Save
            os.makedirs(args.output_dir, exist_ok=True)
            model.save(f"{args.output_dir}/model.pth")
            model.save_state_dict(f"{args.output_dir}/state_dict.pth")
            model.save_config(f"{args.output_dir}/model_config.yml")
            
            # Predict
            logger.info("Getting predictions...")
            results = model.predict_as_dataframe(
                dm.val_dataloader,
                additional_attributes=["particle_id", "true_energy", "true_theta", "true_phi"],
                gpus=args.gpus,
            )
            results.to_csv(f"{args.output_dir}/results.csv", index=False)
            
            # Evaluate with transforms
            evaluate_results(results, args.output_dir, logger, args.path, args.limits_file)
            
            logger.info(f"✅ Done! Results in: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"❌ Failed: {e}")
        raise


if __name__ == "__main__":
    main() 
 