#!/usr/bin/env python3
"""
Standalone script for Crab data inference.
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import Subset
from tqdm import tqdm


from graphnet.data.dataset import Dataset
from graphnet.data.dataset.dataset import EnsembleDataset
from graphnet.data.dataloader import DataLoader
from graphnet.models import StandardModel

torch.set_float32_matmul_precision('high')

def parse_args():
    parser = argparse.ArgumentParser(description="Crab data inference script")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model config YAML")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to model checkpoint (.ckpt)")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to write the output Parquet file")
    parser.add_argument("--gpus", type=int, nargs='+', required=True, help="GPU device IDs to use (e.g., 0 or 0 1)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for data loader")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--prefix", type=str, default='gh', help="Prefix to add to the output file name")
    parser.add_argument("--filter-mono", action='store_true', help="Filter out mono events")
    return parser.parse_args()


def main():
    args = parse_args()

    assert Path(args.model_path).exists(), f"Model config file {args.model_path} does not exist"
    assert Path(args.checkpoint_path).exists(), f"Checkpoint file {args.checkpoint_path} does not exist"

    # Collect dataset configuration files
    dataset_configs = list(Path('.').glob('.yml'))
    yaml_dataset_configs = list(Path('.').glob('*.yaml'))
    # combine yaml and yml files
    dataset_configs = dataset_configs + yaml_dataset_configs
    if not dataset_configs:
        raise FileNotFoundError("No dataset*.yml files found in current directory")
    datasets = [Dataset.from_config(str(p)) for p in dataset_configs]
    crab_dataset = EnsembleDataset(datasets)

    # Filter out mono events (keep only events with exactly two telescopes)
    if args.filter_mono:
        print("Filtering out mono events")
        tel_id_idx = 3
        n_excluded = 0
        good_idxs = []
        for i in tqdm(range(len(crab_dataset)), desc='Filtering events'):
            data = crab_dataset[i]
            tel_ids = torch.unique(data.x[:, tel_id_idx])
            if len(tel_ids) == 2:
                good_idxs.append(i)
            else:
                n_excluded += 1
    
        print(f"Excluded {n_excluded} events")
        
        filtered = Subset(crab_dataset, good_idxs)

    else:
        print("Not filtering out mono events")
        good_idxs = list(range(len(crab_dataset)))
        filtered = Subset(crab_dataset, good_idxs)

    dataloader = DataLoader(
        filtered,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Load model and checkpoint
    model = StandardModel.from_config(args.model_path, trust=True)
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=True)

    # Run inference without additional attributes first
    additional_attributes = ["event_id", "n_pulses", "telescope_phi", "telescope_theta", "signal", "tel_id"]
    results = model.predict_as_dataframe(
        dataloader,
        gpus=args.gpus,
        additional_attributes=additional_attributes
    )

    # Rename signal to size if it exists
    if 'signal' in results.columns:
        results.rename(columns={'signal': 'size'}, inplace=True)

    if "tel_id" in results.columns:
        results.rename(columns={'tel_id': 'stereo'}, inplace=True)
        # convert column to boolean
        results['stereo'] = results['stereo'].astype(bool)
    
    if "event_id" in results.columns:
        # convert to int
        results['event_id'] = results['event_id'].astype(int)
    
    if "n_pulses" in results.columns:
        # convert to int
        results['n_pulses'] = results['n_pulses'].astype(int)

    # add db_indices
    results['db_index'] = good_idxs

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
    print(f"Saved {len(results)} predictions")
    print(f"Saved predictions to {output_path}")

    return

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
    sizes = []
    db_indices = []
    
    for i in tqdm(good_idxs, desc='Collecting attributes'):
        data = crab_dataset[i]
        event_ids.append(int(data.event_id))
        n_pulses.append(int(data.n_pulses))  # Convert tensor to int
        phis.append(float(data.telescope_phi[0]))  # Convert tensor to float
        thetas.append(float(data.telescope_theta[0]))  # Convert tensor to float
        sizes.append(float(data.signal.sum()))  # Convert tensor to float
        db_indices.append(i)

    # Add attributes to results
    results['event_id'] = event_ids
    results['n_pulses'] = n_pulses
    results['telescope_phi'] = phis
    results['telescope_theta'] = thetas
    results['size'] = sizes
    results['db_index'] = db_indices

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
    print(f"Saved {len(results)} predictions")
    print(f"Saved predictions to {output_path}")


if __name__ == '__main__':
    main() 
