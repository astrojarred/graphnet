# source info
SOURCE_NAME="Mrk421"
RUN_NUMBER="5072014"
DATA_DIR="/run/user/54802/output/Mrk421"

FILTER_STRING="${SOURCE_NAME}-${RUN_NUMBER}-*.lmdb"

# gh model
GH_MODEL="/afs/ipp-garching.mpg.de/home/j/jagre/graphnet-i/graphnet/runs/inference/configs/magic-stereo-config-b-sleek-star-58.yml"

# gh checkpoint
GH_CHECKPOINT="/afs/ipp-garching.mpg.de/home/j/jagre/graphnet-i/graphnet/runs/inference/checkpoints/DynEdgeStereo-epoch=9-val_loss=0.08-train_loss=0.05.ckpt"

# direction model
DIRECTION_MODEL="/afs/ipp-garching.mpg.de/home/j/jagre/graphnet-i/graphnet/runs/inference/configs/magic-direction-cam-v5.yml"

# direction checkpoint
# DIRECTION_CHECKPOINT="/afs/ipp-garching.mpg.de/home/j/jagre/graphnet-i/graphnet/runs/direction-cam/results/magic-direction-cam-v4/wandb/magic-direction-cam/zxdds1ve/checkpoints/epoch=0-step=2431.ckpt"
# DIRECTION_CHECKPOINT="/afs/ipp-garching.mpg.de/home/j/jagre/graphnet-i/graphnet/runs/direction-cam/results/magic-direction-cam-v4/wandb/magic-direction-cam/irsz4ksy/checkpoints/epoch=1-step=8915.ckpt"
DIRECTION_CHECKPOINT="/afs/ipp-garching.mpg.de/home/j/jagre/ptmp/runs/direction-cam/results/magic-direction-cam-v5/wandb/magic-direction-cam/1xqedhd9/checkpoints/epoch=2-step=42145.ckpt"
# script path

SCRIPT_PATH="/afs/ipp-garching.mpg.de/home/j/jagre/graphnet-i/graphnet/runs/inference/ensemble-data-inference-gh.py"
DIR_SCRIPT_PATH="/afs/ipp-garching.mpg.de/home/j/jagre/graphnet-i/graphnet/runs/inference/ensemble-data-inference-direction.py"

# gh command
python $SCRIPT_PATH \
--model-path $GH_MODEL \
--data-dir $DATA_DIR \
--filter-string $FILTER_STRING \
--checkpoint-path $GH_CHECKPOINT \
--output-dir ./output-v1-$RUN_NUMBER/gh \
--batch-size 16 \
--num-workers 16 \
--gpus 5

# direction command
python $DIR_SCRIPT_PATH \
--model-path $DIRECTION_MODEL \
--data-dir $DATA_DIR \
--filter-string $FILTER_STRING \
--checkpoint-path $DIRECTION_CHECKPOINT \
--output-dir ./output-v1-$RUN_NUMBER/dir \
--batch-size 2 \
--num-workers 16 \
--gpus 5 \
--min-gammaness 0.2 \
--gh-results-path ./output-v1-$RUN_NUMBER/gh/gh_results.parquet


