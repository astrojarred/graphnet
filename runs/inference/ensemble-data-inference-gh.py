#!/usr/bin/env python3
"""
Standalone script for Crab data inference.
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import Subset
from tqdm import tqdm


from graphnet.data.dataset.dataset import EnsembleDataset
from graphnet.data.dataloader import DataLoader
from graphnet.models import StandardModel
from graphnet.data.dataset.lmdb import LMDBDataset
from graphnet.models.graphs import KNNGraph
from graphnet.models.detector.magic import MAGICDetector
from graphnet.models.graphs.nodes import NodesAsPulses
from graphnet.models.task.magic_direction_improved import TrueTelescopeLabel

torch.set_float32_matmul_precision('high')

def parse_args():
    parser = argparse.ArgumentParser(description="Crab data inference script")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model config YAML")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing LMDB files")
    parser.add_argument("--filter-string", type=str, required=True, help="Glob filter for LMDB files (e.g., 'Mrk421-5072011-*.lmdb')")
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
    assert Path(args.data_dir).exists(), f"Data directory {args.data_dir} does not exist"

    # Build LMDB datasets from directory and filter pattern
    data_dir = Path(args.data_dir)
    lmdb_files = list(data_dir.glob(args.filter_string))
    if not lmdb_files:
        raise FileNotFoundError(f"No files found in {data_dir} matching pattern '{args.filter_string}'")

    # Sort files deterministically; try numeric sort on trailing token if present
    try:
        lmdb_files = sorted(lmdb_files, key=lambda p: int(p.stem.rsplit("-", 1)[-1]))
    except ValueError:
        lmdb_files = sorted(lmdb_files)

    # Define dataset construction parameters as in the notebook example
    features = ['x_cam', 'y_cam', 't', 'tel_id', 'signal', 'telescope_phi', 'telescope_theta']
    pulsemaps = ['total']
    truth = ["telescope_theta", "telescope_phi"]
    graph_definition = KNNGraph(
        columns=[0, 1, 2],
        detector=MAGICDetector(),
        dtype=torch.float32,
        nb_nearest_neighbours=8,
        node_definition=NodesAsPulses(),
        input_feature_names=features,
    )

    # Create individual datasets and add labels
    datasets = []
    true_telescope = TrueTelescopeLabel()
    for file in lmdb_files:
        dataset = LMDBDataset(
            path=str(file),
            features=features,
            truth=truth,
            pulsemaps=pulsemaps,
            graph_definition=graph_definition,
            selection=None,
            index_column='event_id',
        )
        dataset.add_label(true_telescope)
        datasets.append(dataset)

    # Create ensemble dataset
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

    # Run inference
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
    
    # Manually collect attributes from the filtered dataset, skipping bad/empty events
    print(f"Collecting attributes for {len(results)} predictions (skipping missing/empty events)...")
    event_ids = []
    n_pulses = []
    phis = []
    thetas = []
    sizes = []
    db_indices = []

    for i in tqdm(good_idxs, desc='Collecting attributes'):
        try:
            data = crab_dataset[i]
        except Exception:
            # Skip events not present in LMDB after cleaning
            continue
        # Mimic safe-predict filter: require >1 pulses
        npulses = int(getattr(data, 'n_pulses', 0))
        if npulses <= 1:
            continue
        event_ids.append(int(data.event_id))
        n_pulses.append(npulses)
        phis.append(float(data.telescope_phi[0]))
        thetas.append(float(data.telescope_theta[0]))
        sizes.append(float(data.signal.sum()))
        db_indices.append(i)

    # Align results length with collected attributes if needed
    if len(results) != len(event_ids):
        # Keep the leading rows to match attribute order (good_idxs order)
        results = results.iloc[: len(event_ids)].reset_index(drop=True)

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
