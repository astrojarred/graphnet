#!/usr/bin/env python3
"""Simple config-based binary classification GNN for MAGIC telescope data.

This script trains a baseline GNN for gamma-ray astronomy that classifies:
- Gamma rays vs protons (binary classification only)

Uses configuration files for model definition and includes comprehensive W&B logging.

Example usage:
    python magic_baseline_classification.py --model-config configs/magic_binary_classifier.yml --path /path/to/data.lmdb
    
With W&B tracking:
    python magic_baseline_classification.py --model-config configs/magic_binary_classifier.yml --path /path/to/data.lmdb --wandb --wandb-project my-magic-project
    
Resume from checkpoint:
    python magic_baseline_classification.py --model-config configs/magic_binary_classifier.yml --path /path/to/data.lmdb --resume-from-checkpoint /path/to/checkpoint.ckpt

Evaluate only (no training):
    python magic_baseline_classification.py --model-config configs/magic_binary_classifier.yml --path /path/to/data.lmdb --eval-only --checkpoint /path/to/model.pth
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import torch
from typing import Optional, List, Dict, Any
import yaml

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only

from graphnet.data.constants import FEATURES, TRUTH
from graphnet.models import StandardModel
from graphnet.utilities.argparse import ArgumentParser
from graphnet.utilities.logging import Logger
from graphnet.utilities.config import ModelConfig
from graphnet.data import GraphNeTDataModule
from graphnet.data.dataset.lmdb import MAGICLMDBDataset
from graphnet.models.detector.magic import MAGICDetector
from graphnet.models.graphs import KNNGraph

# Import our evaluation utilities (assuming they exist)
try:
    from magic_evaluation_utils import (
        evaluate_magic_classification_results,
        print_magic_classification_summary
    )
    EVALUATION_AVAILABLE = True
except ImportError:
    print("Warning: magic_evaluation_utils not found. Basic evaluation will be used.")
    EVALUATION_AVAILABLE = False

# Constants
features = FEATURES.MAGIC
truth = TRUTH.MAGIC
torch.set_float32_matmul_precision('high')


def setup_data(data_path: str, num_workers: int, model_config: ModelConfig, training_config: Dict = None):
    """Setup data loaders based on model configuration."""
    
    # Extract graph definition from model config arguments
    if not hasattr(model_config, 'arguments') or 'graph_definition' not in model_config.arguments:
        raise ValueError("Model config must have 'graph_definition' in arguments")
    
    graph_config = model_config.arguments['graph_definition']
    
    # Get batch size from training config (default to 128 if not specified)
    batch_size = 128  # Default value
    if training_config and 'training' in training_config:
        batch_size = training_config['training'].get('batch_size', batch_size)
    
    # Create graph definition - we'll reconstruct it to ensure consistency
    graph_definition = KNNGraph(
        detector=MAGICDetector(),
        nb_nearest_neighbours=graph_config.arguments.get('nb_nearest_neighbours', 8),
        columns=graph_config.arguments.get('columns', [0, 1, 2]),  # x_cam, y_cam, t
    )
    
    dataset_args = {
        "path": data_path,
        "features": features,
        "truth": truth + ["particle_id"],  # Add binary gamma/proton label
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


def log_hyperparameters_to_wandb(wandb_logger: Optional[WandbLogger], args, model_config: ModelConfig, training_config: Dict, logger):
    """Log all relevant hyperparameters to W&B."""
    if wandb_logger is None:
        return
        
    logger.info("📊 Logging hyperparameters to W&B...")
    
    # Operational and model hyperparameters
    hyperparams = {
        # Execution mode
        "mode": "eval_only" if args.eval_only else "train",
        "use_test_data": args.use_test_data if hasattr(args, 'use_test_data') else False,
        
        # Operational config (not hyperparameters)
        "num_workers": args.num_workers,
        "gpus": args.gpus,
        
        # Data
        "dataset_path": args.path,
        "dataset_type": "LMDB",
        "features": len(features),
        "truth_labels": len(truth) + 1,  # +1 for particle_id
        "detector": "MAGIC",
        
        # Task type
        "task_type": "binary_classification",
        "target": "gamma_vs_proton",
        
        # Model config (flatten the nested structure)
        "model_config_path": args.model_config,
    }
    
    # Add model configuration details
    if hasattr(model_config, 'arguments'):
        model_args = model_config.arguments
        
        # Backbone details
        if 'backbone' in model_args:
            backbone_config = model_args['backbone']
            if hasattr(backbone_config, 'arguments'):
                backbone_args = backbone_config.arguments
                hyperparams.update({
                    "backbone_class": backbone_config.class_name,
                    "backbone_nb_inputs": backbone_args.get('nb_inputs', None),
                    "backbone_global_pooling": backbone_args.get('global_pooling_schemes', None),
                    "backbone_nb_neighbours": backbone_args.get('nb_neighbours', None),
                })
        
        # Graph definition details
        if 'graph_definition' in model_args:
            graph_config = model_args['graph_definition']
            if hasattr(graph_config, 'arguments'):
                graph_args = graph_config.arguments
                hyperparams.update({
                    "graph_class": graph_config.class_name,
                    "graph_columns": graph_args.get('columns', None),
                    "graph_nb_nearest_neighbours": graph_args.get('nb_nearest_neighbours', None),
                })
        
        # Training hyperparameters from separate config
        if training_config and 'training' in training_config:
            train_config = training_config['training']
            hyperparams.update({
                "max_epochs": train_config.get('max_epochs', None),
                "batch_size": train_config.get('batch_size', None),
                "early_stopping_patience": train_config.get('early_stopping_patience', None),
                "gradient_clip_val": train_config.get('gradient_clip_val', None),
            })
        
        # Optimizer details
        if 'optimizer_class' in model_args:
            hyperparams["optimizer_class"] = str(model_args['optimizer_class'])
        if 'optimizer_kwargs' in model_args:
            for key, value in model_args['optimizer_kwargs'].items():
                hyperparams[f"optimizer_{key}"] = value
        
        # Task details
        if 'tasks' in model_args and model_args['tasks']:
            task_config = model_args['tasks'][0]  # Assuming single task
            if hasattr(task_config, 'arguments'):
                task_args = task_config.arguments
                hyperparams.update({
                    "task_class": task_config.class_name,
                    "task_hidden_size": task_args.get('hidden_size', None),
                    "task_prediction_labels": task_args.get('prediction_labels', None),
                    "task_target_labels": task_args.get('target_labels', None),
                })
    
    # Log to W&B
    wandb_logger.log_hyperparams(hyperparams)
    logger.info(f"✅ Logged {len(hyperparams)} hyperparameters to W&B")


def evaluate_classification_results(results_df: pd.DataFrame, output_dir: str, logger, wandb_logger: Optional[WandbLogger] = None):
    """Evaluate binary classification results."""
    logger.info("📊 Evaluating classification results...")
    
    if EVALUATION_AVAILABLE:
        # Use comprehensive evaluation if available
        metrics = evaluate_magic_classification_results(results_df, output_dir, logger)
    else:
        # Basic evaluation
        from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
        
        # Ensure we have the required columns
        if 'gamma_prob' not in results_df.columns or 'particle_id' not in results_df.columns:
            logger.error("Required columns 'gamma_prob' and 'particle_id' not found in results")
            return {}
        
        # Calculate basic metrics
        y_true = results_df['particle_id'].values
        y_prob = results_df['gamma_prob'].values
        y_pred = (y_prob > 0.5).astype(int)
        
        auc = roc_auc_score(y_true, y_prob)
        accuracy = accuracy_score(y_true, y_pred)
        
        metrics = {
            'auc': auc,
            'accuracy': accuracy,
            'total_events': len(results_df),
            'gamma_events': int(y_true.sum()),
            'proton_events': int((1 - y_true).sum()),
        }
        
        logger.info(f"🎯 Classification Results:")
        logger.info(f"   AUC: {auc:.4f}")
        logger.info(f"   Accuracy: {accuracy:.4f}")
        logger.info(f"   Total events: {len(results_df)}")
        logger.info(f"   Gamma events: {int(y_true.sum())}")
        logger.info(f"   Proton events: {int((1-y_true).sum())}")
        
        # Save classification report
        report = classification_report(y_true, y_pred, 
                                     target_names=['Proton', 'Gamma'], 
                                     output_dict=True)
        
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(f"{output_dir}/classification_report.csv")
        logger.info(f"💾 Classification report saved to: classification_report.csv")
    
    # Log to W&B if available
    if wandb_logger is not None:
        log_metrics_to_wandb(metrics, wandb_logger, logger)
    
    return metrics


def log_metrics_to_wandb(metrics: Dict[str, Any], wandb_logger: WandbLogger, logger):
    """Log evaluation metrics to W&B."""
    logger.info("📊 Logging metrics to W&B...")
    
    # Prepare metrics for W&B
    wandb_metrics = {}
    
    if 'auc' in metrics:
        wandb_metrics['classification/auc'] = metrics['auc']
    if 'accuracy' in metrics:
        wandb_metrics['classification/accuracy'] = metrics['accuracy']
    if 'total_events' in metrics:
        wandb_metrics['classification/total_events'] = metrics['total_events']
    if 'gamma_events' in metrics:
        wandb_metrics['classification/gamma_events'] = metrics['gamma_events']
    if 'proton_events' in metrics:
        wandb_metrics['classification/proton_events'] = metrics['proton_events']
    
    # Log metrics
    wandb_logger.log_metrics(wandb_metrics)
    
    # Log any plots if they exist
    import glob
    import wandb
    
    plot_files = glob.glob(f"{os.path.dirname(wandb_logger.save_dir)}/*.png")
    for plot_file in plot_files:
        plot_name = os.path.basename(plot_file).replace('.png', '')
        wandb_logger.experiment.log({
            f"plots/classification/{plot_name}": wandb.Image(plot_file)
        })
    
    logger.info(f"✅ Logged {len(wandb_metrics)} metrics to W&B")


def main():
    """Main function."""
    parser = ArgumentParser(description="MAGIC binary classification GNN from config")
    
    # Required arguments
    parser.add_argument("--model-config", required=True, help="Path to model configuration file")
    parser.add_argument("--path", required=True, help="Path to LMDB dataset")
    
    # Optional training config
    parser.add_argument("--training-config", type=str, help="Path to training configuration file (optional)")
    
    # Optional arguments
    parser.add_argument("--output-dir", default="./magic_classification_results", help="Output directory")
    parser.add_argument("--resume-from-checkpoint", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--wandb-run-id", type=str, help="W&B run ID to resume")
    
    # Evaluation-only mode
    parser.add_argument("--eval-only", action="store_true", help="Skip training and only run evaluation")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint for evaluation")
    parser.add_argument("--use-test-data", action="store_true", help="Use test split for evaluation")
    
    # Operational arguments (non-hyperparameters)
    parser.with_standard_arguments("gpus", "num-workers")
    
    # W&B arguments
    parser.add_argument("--wandb", action="store_true", help="Use W&B logging")
    parser.add_argument("--wandb-project", default="magic-classification", help="W&B project")
    parser.add_argument("--wandb-entity", default="max-planck", help="W&B entity")
    
    # Additional arguments
    parser.add_argument("--suffix", type=str, help="Suffix for output directory")
    
    args, unknown = parser.parse_known_args()
    
    logger = Logger()
    
    # Validation
    if not os.path.exists(args.model_config):
        logger.error(f"❌ Model config file not found: {args.model_config}")
        return
    
    if args.eval_only and not args.checkpoint:
        logger.error("❌ --checkpoint is required when using --eval-only")
        return
    
    if args.eval_only and not os.path.exists(args.checkpoint):
        logger.error(f"❌ Checkpoint file not found: {args.checkpoint}")
        return
    
    # Setup output directory
    if args.suffix:
        args.output_dir = f"{args.output_dir}_{args.suffix}"
    os.makedirs(args.output_dir, exist_ok=True)
    
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
        
        if args.wandb_run_id:
            wandb_kwargs["id"] = args.wandb_run_id
            wandb_kwargs["resume"] = "must"
            logger.info(f"🔄 Resuming W&B run: {args.wandb_run_id}")
        
        wandb_logger = WandbLogger(**wandb_kwargs)
    
    logger.info("🚀 MAGIC Binary Classification GNN")
    logger.info(f"📁 Config: {args.model_config}")
    logger.info(f"📁 Data: {args.path}")
    logger.info(f"📁 Output: {args.output_dir}")
    
    if args.eval_only:
        logger.info("🔍 Mode: EVALUATION ONLY")
        logger.info(f"📄 Checkpoint: {args.checkpoint}")
        logger.info(f"📊 Data split: {'Test' if args.use_test_data else 'Validation'}")
    else:
        logger.info("🎯 Task: Binary Classification (Gamma vs Proton)")
    
    try:
        # Load model configuration
        logger.info("📄 Loading model configuration...")
        model_config = ModelConfig.load(args.model_config)
        logger.info(f"✅ Model config loaded: {model_config.class_name}")
        
        # Load training configuration
        training_config = {}
        if args.training_config:
            if os.path.exists(args.training_config):
                logger.info(f"📄 Loading training configuration from: {args.training_config}")
                with open(args.training_config, 'r') as f:
                    training_config = yaml.safe_load(f)
                logger.info("✅ Training config loaded")
            else:
                logger.warning(f"⚠️  Training config file not found: {args.training_config}")
        else:
            logger.info("📄 Using default training configuration")
        
        # Setup data
        logger.info("📊 Setting up data loaders...")
        dm = setup_data(args.path, args.num_workers, model_config, training_config)
        
        if args.eval_only:
            # Evaluation only mode
            logger.info("🔍 Creating model for evaluation...")
            model: StandardModel = StandardModel.from_config(model_config, trust=True)
            
            # Log hyperparameters
            log_hyperparameters_to_wandb(wandb_logger, args, model_config, training_config, logger)
            
            # Load checkpoint
            logger.info("Loading weights from checkpoint...")
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
            
            # Get predictions
            logger.info(f"Getting predictions on {split_name} data...")
            results = model.predict_as_dataframe(
                eval_dataloader,
                additional_attributes=["particle_id"],
                gpus=args.gpus,
            )
            
            # Save results
            results.to_csv(f"{args.output_dir}/results_{split_name}.csv", index=False)
            logger.info(f"💾 Results saved to: results_{split_name}.csv")
            
            # Evaluate
            evaluate_classification_results(results, args.output_dir, logger, wandb_logger)
            
            logger.info(f"✅ Evaluation complete! Results in: {args.output_dir}")
            
        else:
            # Training mode
            logger.info("🏗️  Creating model for training...")
            model: StandardModel = StandardModel.from_config(model_config, trust=True)
            
            # Log hyperparameters and config to W&B
            log_hyperparameters_to_wandb(wandb_logger, args, model_config, training_config, logger)
            
            if wandb_logger and rank_zero_only == 0:
                # Log the full model config to W&B
                wandb_logger.experiment.config.update(model_config.as_dict())
            
            # Train
            checkpoint_path = args.resume_from_checkpoint
            if checkpoint_path:
                logger.info(f"🔄 Resuming training from: {checkpoint_path}")
            else:
                logger.info("🚀 Starting fresh training...")
            
            # Get training hyperparameters from training config
            max_epochs = 50  # Default value
            early_stopping_patience = 10  # Default value
            gradient_clip_val = 1.0  # Default value
            
            # Try to extract from training config if available
            if training_config and 'training' in training_config:
                train_config = training_config['training']
                max_epochs = train_config.get('max_epochs', max_epochs)
                early_stopping_patience = train_config.get('early_stopping_patience', early_stopping_patience)
                gradient_clip_val = train_config.get('gradient_clip_val', gradient_clip_val)
            
            logger.info(f"🎯 Training config: max_epochs={max_epochs}, early_stopping_patience={early_stopping_patience}, gradient_clip_val={gradient_clip_val}")
            
            model.fit(
                dm.train_dataloader,
                dm.val_dataloader,
                early_stopping_patience=early_stopping_patience,
                logger=wandb_logger,
                gradient_clip_val=gradient_clip_val,
                gpus=args.gpus,
                max_epochs=max_epochs,
                ckpt_path=checkpoint_path,
            )
            
            # Save model
            model.save(f"{args.output_dir}/model.pth")
            model.save_state_dict(f"{args.output_dir}/state_dict.pth")
            model.save_config(f"{args.output_dir}/model_config.yml")
            logger.info("💾 Model saved")
            
            # Get predictions on validation set
            logger.info("Getting predictions on validation data...")
            results = model.predict_as_dataframe(
                dm.val_dataloader,
                additional_attributes=["particle_id"],
                gpus=args.gpus,
            )
            
            # Save results
            results.to_csv(f"{args.output_dir}/results.csv", index=False)
            logger.info("💾 Results saved to: results.csv")
            
            # Evaluate
            evaluate_classification_results(results, args.output_dir, logger, wandb_logger)
            
            logger.info(f"✅ Training complete! Results in: {args.output_dir}")
    
    except Exception as e:
        logger.error(f"❌ Failed: {e}")
        raise


if __name__ == "__main__":
    main() 
 