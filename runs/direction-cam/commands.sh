# v5: no telescope coordinates, weighted
python magic-camera-plane-train.py \
  --dataset-config        /afs/ipp-garching.mpg.de/home/j/jagre/graphnet-i/graphnet/runs/datasets/gpu01-2M-gammas_nocoords_config.yaml \
  --model-config          config/magic-direction-cam-v5.yml \
  --output-dir            results/magic-direction-cam-v5 \
  --max-epochs            10 \
  --use-swa \
  --swa-epoch-start       2 \
  --batch-size            2 \
  --gpus                  0 2 \
  --accumulate-grad-batches 32 \
  --num-workers           32 \
  --early-stopping-patience 8 \
  --precision             "32-true" \
  --gradient-clip-val     1.0 \
  --limit-val-batches     0.1 \
  --val-check-interval    0.1 \
  --wandb \
  --wandb-project         "magic-direction-cam"
  # --checkpoint-path       /afs/ipp-garching.mpg.de/home/j/jagre/graphnet-i/graphnet/runs/inference/checkpoints/DirectionCam-finetune-epoch=2-step=23503.ckpt \
  # --checkpoint-backbone-only

# v4: no telescope coordinates
python magic-camera-plane-train.py \
  --dataset-config        /afs/ipp-garching.mpg.de/home/j/jagre/graphnet-i/graphnet/runs/datasets/gpu01-2M-gammas_nocoords_config.yaml \
  --model-config          config/magic-direction-cam-v4-nocoords.yml \
  --output-dir            results/magic-direction-cam-v4 \
  --max-epochs            10 \
  --use-swa \
  --swa-epoch-start       2 \
  --batch-size            4 \
  --gpus                  3 5 6 7 \
  --accumulate-grad-batches 16 \
  --num-workers           32 \
  --early-stopping-patience 8 \
  --precision             "32-true" \
  --gradient-clip-val     1.0 \
  --limit-val-batches     0.1 \
  --val-check-interval    0.1 \
  --wandb \
  --wandb-project         "magic-direction-cam"
  # --checkpoint-path       /afs/ipp-garching.mpg.de/home/j/jagre/graphnet-i/graphnet/runs/inference/checkpoints/DirectionCam-finetune-epoch=2-step=23503.ckpt \
  # --checkpoint-backbone-only

# v4 checkpointed: direct mars function
python magic-camera-plane-train.py \
  --dataset-config        /afs/ipp-garching.mpg.de/home/j/jagre/graphnet-i/graphnet/runs/datasets/gpu01-2M-gammas_cam_config.yaml \
  --model-config          config/magic-direction-cam-v4.yml \
  --output-dir            results/magic-direction-cam-v4 \
  --max-epochs            5 \
  --batch-size            4 \
  --gpus                  3 5 6 7 \
  --accumulate-grad-batches 16 \
  --num-workers           32 \
  --early-stopping-patience 8 \
  --precision             "32-true" \
  --gradient-clip-val     1.0 \
  --limit-val-batches     0.1 \
  --val-check-interval    0.1 \
  --wandb \
  --wandb-project         "magic-direction-cam" \
  --checkpoint-path       /afs/ipp-garching.mpg.de/home/j/jagre/graphnet-i/graphnet/runs/inference/checkpoints/DirectionCam-finetune-epoch=2-step=23503.ckpt \
  --checkpoint-backbone-only

# v3 checkpointed
python magic-camera-plane-train.py \
  --dataset-config        /afs/ipp-garching.mpg.de/home/j/jagre/graphnet-i/graphnet/runs/datasets/gpu01-2M-gammas_cam_config.yaml \
  --model-config          config/magic-direction-cam-v3.yml \
  --output-dir            results/magic-direction-cam-v3 \
  --max-epochs            20 \
  --batch-size            4 \
  --gpus                  0 1 2 3 \
  --accumulate-grad-batches 16 \
  --num-workers           32 \
  --early-stopping-patience 8 \
  --use-swa \
  --swa-epoch-start       2 \
  --precision             "32-true" \
  --gradient-clip-val     1.0 \
  --limit-val-batches     0.1 \
  --val-check-interval    0.1 \
  --wandb \
  --wandb-project         "magic-direction-cam" \
  --checkpoint-path       /afs/ipp-garching.mpg.de/home/j/jagre/graphnet-i/graphnet/runs/direction-cam/results/magic-direction-cam-v1/wandb/magic-direction-cam/ilaurmfb/checkpoints/epoch=3-step=43227.ckpt \
  --checkpoint-backbone-only

# v2 checkpointed
python magic-camera-plane-train.py \
  --dataset-config        /afs/ipp-garching.mpg.de/home/j/jagre/graphnet-i/graphnet/runs/datasets/gpu01-2M-gammas_cam_config.yaml \
  --model-config          config/magic-direction-cam-v2.yml \
  --output-dir            results/magic-direction-cam-v2 \
  --max-epochs            20 \
  --batch-size            4 \
  --gpus                  5 6 7 \
  --accumulate-grad-batches 16 \
  --num-workers           32 \
  --early-stopping-patience 8 \
  --use-swa \
  --swa-epoch-start       2 \
  --precision             "32-true" \
  --gradient-clip-val     1.0 \
  --limit-val-batches     0.1 \
  --val-check-interval    0.1 \
  --wandb \
  --wandb-project         "magic-direction-cam" \
  --checkpoint-path       /afs/ipp-garching.mpg.de/home/j/jagre/graphnet-i/graphnet/runs/direction-cam/results/magic-direction-cam-v1/wandb/magic-direction-cam/ilaurmfb/checkpoints/epoch=3-step=43227.ckpt \
  --checkpoint-backbone-only

# v1 checkpointed
python magic-camera-plane-train.py \
  --dataset-config        /afs/ipp-garching.mpg.de/home/j/jagre/graphnet-i/graphnet/runs/datasets/gpu01-2M-gammas_cam_config.yaml \
  --model-config          config/magic-direction-cam-v1.yml \
  --output-dir            results/magic-direction-cam-v1 \
  --max-epochs            20 \
  --batch-size            4 \
  --gpus                  5 6 7 \
  --accumulate-grad-batches 16 \
  --num-workers           32 \
  --early-stopping-patience 8 \
  --use-swa \
  --swa-epoch-start       2 \
  --precision             "32-true" \
  --gradient-clip-val     1.0 \
  --limit-val-batches     0.1 \
  --val-check-interval    0.1 \
  --wandb \
  --wandb-project         "magic-direction-cam" \
  --checkpoint-path       /afs/ipp-garching.mpg.de/home/j/jagre/graphnet-i/graphnet/runs/direction-cam/results/magic-direction-cam-v1/wandb/magic-direction-cam/oawzjglr/checkpoints/epoch=2-step=743.ckpt \
  --checkpoint-backbone-only

# v1
python magic-camera-plane-train.py \
  --dataset-config        /afs/ipp-garching.mpg.de/home/j/jagre/graphnet-i/graphnet/runs/datasets/gpu01-2M-gammas_cam_config.yaml \
  --model-config          config/magic-direction-cam-v1.yml \
  --output-dir            results/magic-direction-cam-v1 \
  --max-epochs            20 \
  --batch-size            4 \
  --gpus                  1 2 3 5 \
  --accumulate-grad-batches 512 \
  --num-workers           32 \
  --early-stopping-patience 8 \
  --use-swa \
  --swa-epoch-start       3 \
  --precision             "32-true" \
  --gradient-clip-val     1.0 \
  --limit-val-batches     0.1 \
  --val-check-interval    0.1 \
  --wandb \
  --wandb-project         "magic-direction-cam"
