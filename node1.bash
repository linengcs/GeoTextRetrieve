GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost # your master address
MASTER_PORT=29500 # your master port
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=/mnt/shared-storage-user/tanxin/lilinfeng/new_track4/track4/data/new_split_merged_train.json # the directory of query_train.jsonl
EVAL_DATA_PATH=/mnt/shared-storage-user/tanxin/lilinfeng/new_track4/track4/data/new_merged_test.json
SAVE_PATH=/mnt/shared-storage-user/tanxin/lilinfeng/new_track4/GeoTextRetrieve/train # your saving path
IMAGE_PATH=/mnt/shared-storage-user/tanxin/lilinfeng/robosense_track4/track4-cross-modal-drone-navigation/images # the training image directory
EPOCH=50
RESUME_PATH=/mnt/shared-storage-user/tanxin/lilinfeng/new_track4/GeoTextRetrieve/train/checkpoint-8000 # pre-trained visualized bge weights
SAVE_STEPS=500
GROUP_SIZE=5 # = one (positive sample) + number (of hard negative samples)
BSZ_PERGPU=100
LR=1e-5

Training_Dir=/mnt/shared-storage-user/tanxin/lilinfeng/new_track4 #your training dir
DeepSpeedConfig=/mnt/shared-storage-user/tanxin/lilinfeng/new_track4/GeoTextRetrieve/ds_config_zero2.json #your deepspeed config file
cd $Training_Dir
# Data and model


mkdir $SAVE_PATH
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

export LAUNCHER="torchrun \
    $DISTRIBUTED_ARGS \
    "

full_options="
  --output_dir $SAVE_PATH \
  --bge_model_name_or_path  /mnt/shared-storage-user/tanxin/lilinfeng/robosense_track4/BAAI/bge-m3 \
  --visual_model_name_or_path  /mnt/shared-storage-user/tanxin/lilinfeng/robosense_track4/BAAI/EVA02_CLIP_L_psz14_s4B.pt \
  --dataloader_num_workers 1  \
  --train_data $DATA_PATH \
  --eval_data $EVAL_DATA_PATH \
  --train_data_image $IMAGE_PATH \
  --train_group_size $GROUP_SIZE \
  --learning_rate $LR \
  --weight_decay 0.05 \
  --warmup_steps 1000 \
  --bf16 \
  --per_device_train_batch_size $BSZ_PERGPU \
  --per_device_eval_batch_size $BSZ_PERGPU \
  --dataloader_drop_last True \
  --normlized True \
  --temperature 0.1 \
  --logging_steps 100 \
  --num_train_epochs $EPOCH \
  --negatives_cross_device \
  --train_text_tower True  \
  --train_vision_tower True \
  --custom_train_vision_tower 4 \
  --custom_train_text_tower 2 \
  --save_steps $SAVE_STEPS \
  --do_eval True \
  --eval_strategy "steps" \
  --eval_steps $SAVE_STEPS \
  --load_best_model_at_end True \
  --metric_for_best_model "eval_loss" \
  --greater_is_better False \
  --deepspeed $DeepSpeedConfig \
  --gradient_checkpointing \
  "

run_cmd="$LAUNCHER -m GeoTextRetrieve.run_ds_cirr ${full_options}"
echo ${run_cmd}
eval ${run_cmd} 2>&1 | tee $SAVE_PATH/output_$NODE_RANK.log



set +x

