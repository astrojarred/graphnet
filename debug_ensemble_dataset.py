#!/usr/bin/env python3
"""
Diagnostic script to debug EnsembleDataset indexing issues.
"""
import argparse
from pathlib import Path
import sys

import torch
from tqdm import tqdm

from graphnet.data.dataset import Dataset
from graphnet.data.dataset.dataset import EnsembleDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Debug EnsembleDataset indexing")
    parser.add_argument("--dataset-pattern", type=str, default='dataset*.yml', 
                       help="Pattern to match dataset config files")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("=== EnsembleDataset Debugging ===")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Collect dataset configuration files
    dataset_configs = list(Path('.').glob(args.dataset_pattern))
    print(f"Found {len(dataset_configs)} dataset config files: {[p.name for p in dataset_configs]}")
    
    if not dataset_configs:
        print("No dataset config files found!")
        return
    
    try:
        # Load individual datasets
        print("\n--- Loading individual datasets ---")
        datasets = []
        for i, config_path in enumerate(dataset_configs):
            print(f"Loading dataset {i+1}: {config_path.name}")
            try:
                dataset = Dataset.from_config(str(config_path))
                print(f"  Type: {type(dataset)}")
                print(f"  Length: {len(dataset)}")
                if hasattr(dataset, '_indices'):
                    print(f"  Has _indices: True")
                    print(f"  First few indices: {dataset._indices[:5] if len(dataset._indices) > 0 else '[]'}")
                else:
                    print(f"  Has _indices: False")
                
                # Test accessing first element
                try:
                    first_data = dataset[0]
                    print(f"  Successfully accessed dataset[0]")
                    print(f"  Data type: {type(first_data)}")
                    if hasattr(first_data, 'x'):
                        print(f"  Has x attribute: True, shape: {first_data.x.shape}")
                    else:
                        print(f"  Has x attribute: False")
                except Exception as e:
                    print(f"  Failed to access dataset[0]: {e}")
                
                datasets.append(dataset)
                
            except Exception as e:
                print(f"  Failed to load dataset: {e}")
                continue
        
        if not datasets:
            print("No datasets loaded successfully!")
            return
        
        # Create EnsembleDataset
        print(f"\n--- Creating EnsembleDataset from {len(datasets)} datasets ---")
        try:
            ensemble_dataset = EnsembleDataset(datasets)
            print(f"EnsembleDataset type: {type(ensemble_dataset)}")
            print(f"EnsembleDataset length: {len(ensemble_dataset)}")
            
            # Check if it has the expected attributes
            print(f"Has _indices: {hasattr(ensemble_dataset, '_indices')}")
            if hasattr(ensemble_dataset, '_indices'):
                print(f"First few indices: {ensemble_dataset._indices[:5] if len(ensemble_dataset._indices) > 0 else '[]'}")
            
            # Test accessing first element
            print(f"\n--- Testing EnsembleDataset indexing ---")
            try:
                first_data = ensemble_dataset[0]
                print(f"Successfully accessed ensemble_dataset[0]")
                print(f"Data type: {type(first_data)}")
                if hasattr(first_data, 'x'):
                    print(f"Has x attribute: True, shape: {first_data.x.shape}")
                else:
                    print(f"Has x attribute: False")
            except Exception as e:
                print(f"Failed to access ensemble_dataset[0]: {e}")
                print(f"Error type: {type(e)}")
                
                # Try to understand the error better
                if "No data found for event" in str(e):
                    print("This suggests an LMDB indexing issue")
                elif "Index" in str(e):
                    print("This suggests a general indexing issue")
                
                # Try accessing with different indices
                print(f"\n--- Testing different indices ---")
                for test_idx in [0, 1, 2, 10, 100]:
                    if test_idx < len(ensemble_dataset):
                        try:
                            test_data = ensemble_dataset[test_idx]
                            print(f"Successfully accessed ensemble_dataset[{test_idx}]")
                            break
                        except Exception as e2:
                            print(f"Failed to access ensemble_dataset[{test_idx}]: {e2}")
            
            # Test the filtering loop
            print(f"\n--- Testing filtering loop ---")
            try:
                tel_id_idx = 3
                n_excluded = 0
                good_idxs = []
                
                # Test with a small range first
                test_range = min(10, len(ensemble_dataset))
                print(f"Testing with first {test_range} events...")
                
                for i in tqdm(range(test_range), desc='Testing filtering'):
                    try:
                        data = ensemble_dataset[i]
                        tel_ids = torch.unique(data.x[:, tel_id_idx])
                        if len(tel_ids) == 2:
                            good_idxs.append(i)
                        else:
                            n_excluded += 1
                    except Exception as e:
                        print(f"Failed at index {i}: {e}")
                        break
                
                print(f"Filtering test completed: {len(good_idxs)} good, {n_excluded} excluded")
                
            except Exception as e:
                print(f"Filtering test failed: {e}")
                
        except Exception as e:
            print(f"Failed to create EnsembleDataset: {e}")
            
    except Exception as e:
        print(f"General error: {e}")

if __name__ == '__main__':
    main() 
