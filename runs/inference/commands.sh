# gh model
GH_MODEL="/afs/ipp-garching.mpg.de/home/j/jagre/graphnet-i/graphnet/runs/inference/configs/magic-stereo-config-b-sleek-star-58.yml"

# gh checkpoint
GH_CHECKPOINT="/afs/ipp-garching.mpg.de/home/j/jagre/graphnet-i/graphnet/runs/inference/checkpoints/DynEdgeStereo-epoch=9-val_loss=0.08-train_loss=0.05.ckpt"

# direction model
DIRECTION_MODEL="/afs/ipp-garching.mpg.de/home/j/jagre/graphnet-i/graphnet/runs/inference/configs/magic-direction-cam-v5.yml"

# direction checkpoint
# DIRECTION_CHECKPOINT="/afs/ipp-garching.mpg.de/home/j/jagre/graphnet-i/graphnet/runs/direction-cam/results/magic-direction-cam-v4/wandb/magic-direction-cam/zxdds1ve/checkpoints/epoch=0-step=2431.ckpt"
# DIRECTION_CHECKPOINT="/afs/ipp-garching.mpg.de/home/j/jagre/graphnet-i/graphnet/runs/direction-cam/results/magic-direction-cam-v4/wandb/magic-direction-cam/irsz4ksy/checkpoints/epoch=1-step=8915.ckpt"
DIRECTION_CHECKPOINT="/afs/ipp-garching.mpg.de/home/j/jagre/graphnet-i/graphnet/runs/direction-cam/results/magic-direction-cam-v5/wandb/magic-direction-cam/1xqedhd9/checkpoints/epoch=2-step=42145.ckpt"
# script path

SCRIPT_PATH="/afs/ipp-garching.mpg.de/home/j/jagre/graphnet-i/graphnet/runs/inference/crab-data-inference-gh.py"
DIR_SCRIPT_PATH="/afs/ipp-garching.mpg.de/home/j/jagre/graphnet-i/graphnet/runs/inference/crab-data-inference-direction.py"

# gh command
python $SCRIPT_PATH \
--model-path $GH_MODEL \
--checkpoint-path $GH_CHECKPOINT \
--output-dir ./gh \
--batch-size 16 \
--num-workers 16 \
--gpus 5

# direction command
python $DIR_SCRIPT_PATH \
--model-path $DIRECTION_MODEL \
--checkpoint-path $DIRECTION_CHECKPOINT \
--output-dir ./mrk-output-v1/dir \
--batch-size 2 \
--num-workers 16 \
--gpus 5 \
--min-gammaness 0.2 \
--gh-results-path ./mrk-output-v1/gh/gh_results.parquet


