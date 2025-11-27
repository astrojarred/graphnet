import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append("..")

from evaluate_camera_plane import (
    evaluate_camera_plane_model
)

# a4kqdtod
checkpoint_path = "/afs/ipp-garching.mpg.de/home/j/jagre/graphnet-i/graphnet/runs/direction-cam/results/magic-direction-cam-v5/wandb/magic-direction-cam/1xqedhd9/checkpoints/epoch=2-step=42145.ckpt"
config_file = "/afs/ipp-garching.mpg.de/home/j/jagre/graphnet-i/graphnet/runs/direction-cam/config/magic-direction-cam-v5.yml"
output_dir = "./output-v5-b/eval"
data_config_path = "/afs/ipp-garching.mpg.de/home/j/jagre/graphnet-i/graphnet/runs/datasets/gpu01-2M-gammas_nocoords_config.yaml"

# find newest checkpoint
print("Configuration files:")
print(f"Model config: {config_file}")
print(f"Dataset config: {data_config_path}")
print(f"Checkpoint: {Path(checkpoint_path).absolute()}")

# Verify that files exist
for name, path in [("Model config", config_file), ("Dataset config", data_config_path), ("Checkpoint", checkpoint_path)]:
    if Path(path).exists():
        print(f"✓ {name} exists")
    else:
        print(f"✗ {name} NOT FOUND: {path}")

print("Configuration files:")
print(f"Model config: {config_file}")
print(f"Dataset config: {data_config_path}")
print(f"Checkpoint: {Path(checkpoint_path).absolute()}")

_df, metrics = evaluate_camera_plane_model(
    model_config_path=config_file,
    dataset_config_path=data_config_path,
    checkpoint_path=checkpoint_path,
    dataset_split="test",
    dataset_fraction=0.1,
    batch_size=4,
    plot=False,
    gpus=[3],
    save_results=True,
    output_dir=output_dir,
)

print(f"Saving results to {output_dir}")
output_dir = Path(output_dir)
if not output_dir.exists():
    output_dir.mkdir(parents=True)

_df.to_parquet(output_dir / "dir_results.parquet")

print(f"Saved results to {output_dir / 'dir_results.parquet'}")


