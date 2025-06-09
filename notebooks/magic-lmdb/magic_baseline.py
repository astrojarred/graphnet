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
    
Auto-resume from latest checkpoint:
    python magic_baseline.py --path /path/to/data.lmdb --auto-resume
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

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

def evaluate_results(results_df, output_dir, logger):
    """Quick evaluation of results."""
    logger.info("Evaluating results...")
    
    # Classification metrics
    y_true = results_df['particle_id'].values
    y_prob = results_df['gamma_prob'].values  # Use named prediction
    y_pred = (y_prob > 0.5).astype(int)
    
    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    
    # Energy metrics (gamma events only)
    gamma_mask = results_df['particle_id'] == 0
    if gamma_mask.sum() > 0:
        df_gamma = results_df[gamma_mask]
        y_true_energy = df_gamma['true_energy'].values
        y_pred_energy = df_gamma['energy_pred'].values  # Use named prediction
        
        energy_mae = np.mean(np.abs(y_pred_energy - y_true_energy))
        energy_bias = np.mean((y_pred_energy - y_true_energy) / (y_true_energy + 1e-8))
    else:
        energy_mae = energy_bias = 0
        
    # Direction metrics (gamma events only)
    if gamma_mask.sum() > 0:
        theta_true = df_gamma['true_theta'].values
        phi_true = df_gamma['true_phi'].values
        theta_pred = df_gamma['theta_pred'].values  # Use named prediction
        phi_pred = df_gamma['phi_pred'].values      # Use named prediction
        
        # Angular distance calculation
        x1 = np.sin(theta_true) * np.cos(phi_true)
        y1 = np.sin(theta_true) * np.sin(phi_true)
        z1 = np.cos(theta_true)
        
        x2 = np.sin(theta_pred) * np.cos(phi_pred)
        y2 = np.sin(theta_pred) * np.sin(phi_pred)
        z2 = np.cos(theta_pred)
        
        dot_product = np.clip(x1*x2 + y1*y2 + z1*z2, -1, 1)
        angular_dist_deg = np.degrees(np.arccos(dot_product))
        
        mean_angular_error = np.mean(angular_dist_deg)
        angular_resolution_68 = np.percentile(angular_dist_deg, 68)
    else:
        mean_angular_error = angular_resolution_68 = 0
    
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
    plt.savefig(f'{output_dir}/roc_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Energy correlation (if gamma events exist)
    if gamma_mask.sum() > 0:
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true_energy, y_pred_energy, alpha=0.6, s=20)
        plt.plot([y_true_energy.min(), y_true_energy.max()], 
                 [y_true_energy.min(), y_true_energy.max()], 'r--')
        plt.xlabel('True Energy')
        plt.ylabel('Predicted Energy')
        plt.title('Energy Reconstruction')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/energy_correlation.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # Print summary
    print("\n" + "="*50)
    print("MAGIC BASELINE GNN RESULTS")
    print("="*50)
    print(f"📊 CLASSIFICATION:")
    print(f"   • AUC: {auc:.4f}")
    print(f"   • Accuracy: {accuracy:.4f}")
    print(f"   • Total events: {len(results_df)}")
    
    if gamma_mask.sum() > 0:
        print(f"\n⚡ ENERGY (Gamma events only):")
        print(f"   • MAE: {energy_mae:.4f}")
        print(f"   • Relative bias: {energy_bias*100:.2f}%")
        print(f"   • Gamma events: {gamma_mask.sum()}")
        
        print(f"\n🎯 DIRECTION (Gamma events only):")
        print(f"   • Mean angular error: {mean_angular_error:.2f}°")
        print(f"   • 68% containment: {angular_resolution_68:.2f}°")
    
    print(f"\n📁 Plots saved: roc_curve.png, energy_correlation.png")
    print("="*50)


def main():
    """Main function."""
    parser = ArgumentParser(description="MAGIC baseline multi-task GNN")
    
    parser.add_argument("--path", required=True, help="Path to LMDB dataset")
    parser.add_argument("--output-dir", default="./magic_baseline_results", help="Output directory")
    parser.add_argument("--resume-from-checkpoint", type=str, help="Path to checkpoint to resume from")
    # parser.add_argument("--auto-resume", action="store_true", help="Automatically find and resume from latest checkpoint")
    parser.add_argument("--wandb-run-id", type=str, help="W&B run ID to resume (if resuming a W&B logged run)")
    
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
    
    # Handle checkpoint resumption
    checkpoint_path = None
    wandb_run_id = args.wandb_run_id
    
    if args.resume_from_checkpoint:
        checkpoint_path = args.resume_from_checkpoint
        logger.info(f"🔄 Resuming from checkpoint: {checkpoint_path}")
    
    # W&B setup
    wandb_logger = None
    if args.wandb:
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
    logger.info(f"🎯 Tasks: Classification + Energy + Direction (3 tasks total)")
    
    try:
        # Setup
        dm = setup_data(args.path, args.batch_size, args.num_workers)
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
        
        # Evaluate
        evaluate_results(results, args.output_dir, logger)
        
        logger.info(f"✅ Done! Results in: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"❌ Failed: {e}")
        raise


if __name__ == "__main__":
    main() 
