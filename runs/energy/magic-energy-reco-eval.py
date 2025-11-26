import sys
from pathlib import Path

sys.path.append("..")

from evaluate_energy_reco import (
    evaluate_energy_reco_model
)

# Configuration
checkpoint_path = "/afs/ipp-garching.mpg.de/home/j/jagre/graphnet-i/graphnet/runs/energy/results/magic-energy-reco-v1/wandb/magic-energy-reco/1xqedhd9/checkpoints/epoch=2-step=42145.ckpt"
config_file = "/afs/ipp-garching.mpg.de/home/j/jagre/graphnet-i/graphnet/runs/energy/config/magic-energy-reco-v1.yml"
output_dir = "./output-v1/eval"
data_config_path = "/afs/ipp-garching.mpg.de/home/j/jagre/graphnet-i/graphnet/runs/datasets/gpu01-2M-gammas_nocoords_config.yaml"

# Print configuration
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

_df, metrics = evaluate_energy_reco_model(
    model_config_path=config_file,
    dataset_config_path=data_config_path,
    checkpoint_path=checkpoint_path,
    dataset_split="test",
    dataset_fraction=0.1,
    batch_size=4,
    plot=True,
    gpus=[0],
    save_results=True,
    output_dir=output_dir,
    energy_in_log10=True,
    energy_unit="GeV",
)

print(f"Saving results to {output_dir}")
output_dir = Path(output_dir)
if not output_dir.exists():
    output_dir.mkdir(parents=True)

_df.to_parquet(output_dir / "energy_results.parquet")

print(f"Saved results to {output_dir / 'energy_results.parquet'}")

