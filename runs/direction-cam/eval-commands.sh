EVAL_SCRIPT_PATH="/afs/ipp-garching.mpg.de/home/j/jagre/graphnet-i/graphnet/runs/direction-cam/evaluate_camera_plane.py"
CHECKPOINT_PATH="/afs/ipp-garching.mpg.de/home/j/jagre/graphnet-i/graphnet/runs/direction-cam/results/magic-direction-cam-v5/wandb/magic-direction-cam/1xqedhd9/checkpoints/epoch=2-step=42145.ckpt"
CONFIG_FILE="/afs/ipp-garching.mpg.de/home/j/jagre/graphnet-i/graphnet/runs/direction-cam/config/magic-direction-cam-v5.yml"
DATA_CONFIG_PATH="/afs/ipp-garching.mpg.de/home/j/jagre/graphnet-i/graphnet/runs/datasets/gpu01-2M-gammas_nocoords_config.yaml"

python $EVAL_SCRIPT_PATH \
--model-config $CONFIG_FILE \
--dataset-config $DATA_CONFIG_PATH \
--checkpoint $CHECKPOINT_PATH \
--split test \
--fraction 1.0 \
--batch-size 2 \
--output-dir ./output-v5-b/eval \
--no-plot \
--gpus 0 3 6
