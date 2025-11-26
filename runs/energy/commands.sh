#!/usr/bin/env bash

# v1: baseline MAGIC energy reconstruction run (log10 target, systematic weights)
python magic-energy-reco-train.py \
  --dataset-config        /afs/ipp-garching.mpg.de/home/j/jagre/graphnet-i/graphnet/runs/datasets/gpu01-2M-gammas_config.yaml \
  --model-config          config/magic-energy-reco-v1.yml \
  --output-dir            results/magic-energy-reco-v1 \
  --max-epochs            10 \
  --batch-size            4 \
  --gpus                  6 7 \
  --accumulate-grad-batches 16 \
  --num-workers           32 \
  --early-stopping-patience 8 \
  --precision             "32-true" \
  --gradient-clip-val     1.0 \
  --limit-val-batches     0.1 \
  --val-check-interval    0.1 \
  --wandb \
  --wandb-project         "magic-energy-reco"
  # --checkpoint-path       /path/to/checkpoint.ckpt \
  # --checkpoint-backbone-only

# v1 + SWA: start SWA late to stabilise energy bins
python magic-energy-reco-train.py \
  --dataset-config        /afs/ipp-garching.mpg.de/home/j/jagre/graphnet-i/graphnet/runs/datasets/gpu01-2M-gammas_config.yaml \
  --model-config          config/magic-energy-reco-v1.yml \
  --output-dir            results/magic-energy-reco-v1-swa \
  --max-epochs            15 \
  --batch-size            4 \
  --gpus                  2 3 \
  --accumulate-grad-batches 16 \
  --num-workers           32 \
  --early-stopping-patience 10 \
  --use-swa \
  --swa-epoch-start       3 \
  --precision             "32-true" \
  --gradient-clip-val     0.5 \
  --limit-val-batches     0.1 \
  --val-check-interval    0.1 \
  --wandb \
  --wandb-project         "magic-energy-reco"

# v1 fast-dev sanity check (single GPU / quick debug)
python magic-energy-reco-train.py \
  --dataset-config        /afs/ipp-garching.mpg.de/home/j/jagre/graphnet-i/graphnet/runs/datasets/gpu01-2M-gammas_config.yaml \
  --model-config          config/magic-energy-reco-v1.yml \
  --output-dir            results/magic-energy-reco-debug \
  --max-epochs            1 \
  --batch-size            2 \
  --gpus                  0 \
  --num-workers           8 \
  --fast-dev-run \
  --precision             "32-true"

