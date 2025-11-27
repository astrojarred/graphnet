#!/usr/bin/env python3
"""
Standalone script for Crab data inference.
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import Subset
from tqdm import tqdm
import pandas as pd
import numpy as np

from graphnet.data.dataset import Dataset
from graphnet.data.dataset.dataset import EnsembleDataset
from graphnet.data.dataloader import DataLoader
from graphnet.models import StandardModel
from graphnet.models.task.magic_direction_cam import Loc0CamToLoc

torch.set_float32_matmul_precision('high')

def parse_args():
    parser = argparse.ArgumentParser(description="Crab data inference script")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model config YAML")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to model checkpoint (.ckpt)")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to write the output Parquet file")
    parser.add_argument("--gpus", type=int, nargs='+', required=True, help="GPU device IDs to use (e.g., 0 or 0 1)")
    parser.add_argument("--gh-results-path", type=str, required=True, help="Path to gamma/hadron separation results")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for data loader")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--prefix", type=str, default='dir', help="Prefix to add to the output file name")
    parser.add_argument("--min-gammaness", type=float, default=0.9, help="Minimum gammaness for event to be included")
    return parser.parse_args()


def main():
    args = parse_args()

    assert Path(args.model_path).exists(), f"Model config file {args.model_path} does not exist"
    assert Path(args.checkpoint_path).exists(), f"Checkpoint file {args.checkpoint_path} does not exist"

    # Collect dataset configuration files
    dataset_configs = list(Path('.').glob('dataset*dir.yml'))
    if not dataset_configs:
        raise FileNotFoundError("No dataset*.yml files found in current directory")
    datasets = [Dataset.from_config(str(p)) for p in dataset_configs]
    if len(datasets) > 1:
        raise ValueError("Multiple dataset configuration files not supported yet")
    crab_dataset = datasets[0]

    gh_results = pd.read_parquet(args.gh_results_path)
    print(f"Loaded {len(gh_results)} gamma/hadron separation results")

    # Filter out events with gammaness < min_gammaness
    gh_results = gh_results[gh_results['gamma_prob'] >= args.min_gammaness]
    print(f"Filtered to {len(gh_results)} events with gammaness >= {args.min_gammaness}")

    # Filter out mono events (keep only events with exactly two telescopes)
    good_db_indices = gh_results['db_index'].tolist()
    # good_db_indices = gh_results[gh_results['stereo'].astype(bool)]['db_index'].tolist()

    print(f"Excluded {len(crab_dataset) - len(good_db_indices)} events")
    print(f"Keeping {len(good_db_indices)} events")
    
    filtered = Subset(crab_dataset, good_db_indices)
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
    additional_attributes = ['event_id', 'telescope_theta', 'telescope_phi', 'signal']
    # additional_attributes = ['event_id', 'signal']
    results = model.predict_as_dataframe(
        dataloader,
        gpus=args.gpus,
        additional_attributes=additional_attributes
    )

    print(f"Results columns: {results.columns}")
    for col in results.columns:
        print(f"{col}: {results[col].dtype}")

    if "event_id" in results.columns:
        # convert to int
        print("Converting event_id to int")
        results['event_id'] = results['event_id'].astype(int)
    if "signal" in results.columns:
        # rename to size
        results.rename(columns={'signal': 'size'}, inplace=True)

    print("Saving preliminary results")
    temp_filename = f'{args.prefix}_preliminary_results.parquet' if args.prefix else 'preliminary_results.parquet'
    temp_output_path = Path(args.output_dir) / temp_filename
    temp_output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(str(temp_output_path))
    print(f"Saved {len(results)} preliminary predictions to {temp_output_path}")
    
    # Manually collect attributes from the filtered dataset
    print(f"Collecting attributes for {len(results)} predictions...")
    # event_ids = []
    # phis = []
    # thetas = []
    # arrival_thetas = []
    # arrival_phis = []
    # db_indices = []
    # tel_phi_deg = []
    # tel_theta_deg = []
    # arrival_phi_deg = []
    # arrival_theta_deg = []

    results['db_index'] = good_db_indices

    DIST_CAM = 22.9038  # same focal-length constant used in the task module

    def get_row_attributes(row):
        tel_theta = torch.as_tensor(row['telescope_theta'])
        tel_phi   = torch.as_tensor(row['telescope_phi'])
        x_cam     = torch.as_tensor(row['camera_x_pred'])
        y_cam     = torch.as_tensor(row['camera_y_pred'])

        theta_rad, phi_rad = Loc0CamToLoc(
            tel_theta,
            tel_phi,              # real data → no CORSIKA-to-DRIVE flip
            x_cam,
            y_cam,
            torch.as_tensor(DIST_CAM, dtype=tel_theta.dtype),
        )
        return theta_rad.item(), phi_rad.item()


    print("Calculating arrival theta and phi")
    results[['arrival_theta', 'arrival_phi']] = results.apply(get_row_attributes, axis=1, result_type='expand')

    # for i in tqdm(range(len(filtered)), desc='Collecting attributes'):
    #     data = filtered[i]
    #     event_ids.append(int(data.event_id))
    #     telescope_theta = data.telescope_theta[0]
    #     telescope_phi = data.telescope_phi[0]
    #     pred_theta_rad, pred_phi_rad = camera_to_sky_wrapper(
    #         torch.as_tensor(telescope_theta),
    #         torch.as_tensor(telescope_phi),
    #         torch.as_tensor(results['camera_x_pred'][i]),
    #         torch.as_tensor(results['camera_y_pred'][i]),
    #         use_monte_carlo=False
    #     )

    #     arrival_thetas.append(pred_theta_rad.item())
    #     arrival_phis.append(pred_phi_rad.item())
    #     arrival_theta_deg.append(torch.rad2deg(pred_theta_rad).item())
    #     arrival_phi_deg.append(torch.rad2deg(pred_phi_rad).item())
    #     tel_theta_deg.append(torch.rad2deg(telescope_theta).item())
    #     tel_phi_deg.append(torch.rad2deg(telescope_phi).item())
    #     db_indices.append(i)

    # # Add attributes to results
    # results['event_id'] = event_ids
    # results['arrival_theta'] = arrival_thetas
    # results['arrival_phi'] = arrival_phis
    # results['telescope_theta_deg'] = tel_theta_deg
    # results['telescope_phi_deg'] = tel_phi_deg
    # results['arrival_theta_deg'] = arrival_theta_deg
    # results['arrival_phi_deg'] = arrival_phi_deg
    # results['angular_distance'] = angular_distances
    # results['db_index'] = db_indices

    results['telescope_theta_deg'] = np.rad2deg(results['telescope_theta'])
    results['telescope_phi_deg'] = np.rad2deg(results['telescope_phi'])
    results['arrival_theta_deg'] = np.rad2deg(results['arrival_theta'])
    results['arrival_phi_deg'] = np.rad2deg(results['arrival_phi'])

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
