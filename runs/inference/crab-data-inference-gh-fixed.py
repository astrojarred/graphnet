#!/usr/bin/env python3
"""
Standalone script for Crab data inference - Fixed version for EnsembleDataset.
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import Subset
from tqdm import tqdm
import pandas as pd
import dotenv

from graphnet.data.dataset import Dataset
from graphnet.data.dataset.dataset import EnsembleDataset
from graphnet.data.dataloader import DataLoader
from graphnet.models import StandardModel

def parse_args():
    parser = argparse.ArgumentParser(description="Crab data inference script")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model config YAML")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to model checkpoint (.ckpt)")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to write the output Parquet file")
    parser.add_argument("--gpus", type=int, nargs='+', required=True, help="GPU device IDs to use (e.g., 0 or 0 1)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for data loader")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--prefix", type=str, default='gh', help="Prefix to add to the output file name")
    parser.add_argument("--use-single-dataset", action='store_true', 
                       help="Use single dataset instead of EnsembleDataset (safer option)")
    return parser.parse_args()


def main():
    args = parse_args()
    dotenv.load_dotenv(override=True)

    print("Running gamma/hadron separation inference...")

    assert Path(args.model_path).exists(), f"Model config file {args.model_path} does not exist"
    assert Path(args.checkpoint_path).exists(), f"Checkpoint file {args.checkpoint_path} does not exist"

    # Collect dataset configuration files
    dataset_configs = list(Path('.').glob('dataset*.yml'))
    if not dataset_configs:
        raise FileNotFoundError("No dataset*.yml files found in current directory")
    
    print(f"Found {len(dataset_configs)} dataset config files: {[p.name for p in dataset_configs]}")
    
    # Load datasets
    datasets = [Dataset.from_config(str(p)) for p in dataset_configs]
    
    # Choose between single dataset and EnsembleDataset
    if args.use_single_dataset or len(datasets) == 1:
        if len(datasets) > 1:
            print(f"Warning: Found {len(datasets)} datasets but using single dataset mode")
            print("Using first dataset only. Use --use-single-dataset to suppress this warning.")
        crab_dataset = datasets[0]
        print(f"Using single dataset: {type(crab_dataset).__name__}")
    else:
        print(f"Creating EnsembleDataset from {len(datasets)} datasets")
        crab_dataset = EnsembleDataset(datasets)
        print(f"EnsembleDataset created: {type(crab_dataset).__name__}")

    print(f"Dataset length: {len(crab_dataset)}")

    # Filter out mono events (keep only events with exactly two telescopes)
    tel_id_idx = 3
    n_excluded = 0
    good_idxs = []
    
    print("Filtering events...")
    for i in tqdm(range(len(crab_dataset)), desc='Filtering events'):
        try:
            data = crab_dataset[i]
            tel_ids = torch.unique(data.x[:, tel_id_idx])
            if len(tel_ids) == 2:
                good_idxs.append(i)
            else:
                n_excluded += 1
        except Exception as e:
            print(f"Error accessing event {i}: {e}")
            n_excluded += 1
            continue

    print(f"Excluded {n_excluded} events")
    print(f"Kept {len(good_idxs)} events")
    
    if len(good_idxs) == 0:
        raise ValueError("No events passed the filtering criteria!")
    
    filtered = Subset(crab_dataset, good_idxs)
    dataloader = DataLoader(
        filtered,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Load model and checkpoint
    print("Loading model...")
    model = StandardModel.from_config(args.model_path, trust=True)
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=True)

    # Run inference without additional attributes first
    print("Running inference...")
    results = model.predict_as_dataframe(
        dataloader,
        gpus=args.gpus
    )

    print("Saving preliminary results")
    temp_filename = f'{args.prefix}_preliminary_results.parquet' if args.prefix else 'preliminary_results.parquet'
    temp_output_path = Path(args.output_dir) / temp_filename
    temp_output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(str(temp_output_path))
    print(f"Saved {len(results)} preliminary predictions to {temp_output_path}")
    
    # Manually collect attributes from the filtered dataset
    print(f"Collecting attributes for {len(results)} predictions...")
    event_ids = []
    n_pulses = []
    phis = []
    thetas = []
    
    for i in tqdm(range(len(filtered)), desc='Collecting attributes'):
        try:
            data = filtered[i]
            event_ids.append(int(data.event_id))
            n_pulses.append(int(data.n_pulses))  # Convert tensor to int
            phis.append(float(data.telescope_phi[0]))  # Convert tensor to float
            thetas.append(float(data.telescope_theta[0]))  # Convert tensor to float
        except Exception as e:
            print(f"Error collecting attributes for event {i}: {e}")
            # Use default values if attributes are missing
            event_ids.append(-1)
            n_pulses.append(0)
            phis.append(0.0)
            thetas.append(0.0)
    
    # Add attributes to results
    results['event_id'] = event_ids
    results['n_pulses'] = n_pulses
    results['telescope_phi'] = phis
    results['telescope_theta'] = thetas

    # Save results to Parquet
    filename = f'{args.prefix}_results.parquet' if args.prefix else 'results.parquet'
    output_path = Path(args.output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # if output_path already exists, append new version number
    if output_path.exists():
        version_num = 1
        while output_path.exists():
            output_path = Path(args.output_dir) / f'{filename}_v{version_num}.parquet'
            version_num += 1

    print(f"Saving {len(results)} predictions to {output_path}")
    results.to_parquet(str(output_path))
    print(f"Excluded {n_excluded} events")
    print(f"Saved {len(results)} predictions")
    print(f"Saved predictions to {output_path}")


if __name__ == '__main__':
    main() 
