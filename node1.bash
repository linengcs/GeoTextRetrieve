GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=localhost # your master address
MASTER_PORT=29500 # your master port
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/xieyuan-24039/llf/track4/data/train_output.jsonl # the directory of query_train.jsonl
SAVE_PATH=/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/xieyuan-24039/llf/VISTA_Evaluation_FineTuning/downstream_finetune_example/new_output # your saving path 
IMAGE_PATH=/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/xieyuan-24039/llf/track4/track4-cross-modal-drone-navigation/images # the training image directory
EPOCH=50
RESUME_PATH=/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/xieyuan-24039/llf/VISTA_Evaluation_FineTuning/downstream_finetune_example/output/checkpoint-9400/BGE_EVA_Token.pth # pre-trained visualized bge weights
SAVE_STEPS=1000
GROUP_SIZE=5 # = one (positive sample) + number (of hard negative samples)
BSZ_PERGPU=55
LR=2e-5

Training_Dir=/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/xieyuan-24039/llf/VISTA_Evaluation_FineTuning #your training dir
DeepSpeedConfig=/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/xieyuan-24039/llf/VISTA_Evaluation_FineTuning/downstream_finetune_example/ds_config_zero2.json #your deepspeed config file
cd $Training_Dir
# Data and model


mkdir $SAVE_PATH
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

export LAUNCHER="torchrun \
    $DISTRIBUTED_ARGS \
    "

full_options="
  --output_dir $SAVE_PATH \
  --bge_model_name_or_path  /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/xieyuan-24039/llf/track4/BAAI/bge-m3 \
  --visual_model_name_or_path  /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/xieyuan-24039/llf/track4/EVA02_CLIP_L_psz14_s4B.pt \
  --dataloader_num_workers 1  \
  --train_data $DATA_PATH \
  --train_data_image $IMAGE_PATH \
  --train_group_size $GROUP_SIZE
  --learning_rate $LR \
  --bf16 \
  --per_device_train_batch_size $BSZ_PERGPU \
  --dataloader_drop_last True \
  --normlized True \
  --temperature 0.02 \
  --logging_steps 10 \
  --num_train_epochs $EPOCH \
  --negatives_cross_device \
  --train_text_tower True  \
  --train_vision_tower True \
  --save_steps $SAVE_STEPS \
  --deepspeed $DeepSpeedConfig \
  --gradient_checkpointing \
  "

run_cmd="$LAUNCHER -m downstream_finetune_example.run_ds_cirr ${full_options}"
echo ${run_cmd}
eval ${run_cmd} 2>&1 | tee $SAVE_PATH/output_$NODE_RANK.log



set +x

