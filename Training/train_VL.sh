#!/bin/bash

RUN="LCO_VL_test"

BASE_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
DATA_PATH="data/demo_data.jsonl"
OUTPUT_DIR="checkpoints/${RUN}"

BATCH_SIZE=1056
PER_DEVICE_BATCH_SIZE=33
EPOCH=2
LR=4e-4
MAX_LENGTH=650
LORA_RANK=64
LORA_ALPHA=128

NNODES=4
NPROC_PER_NODE=8

echo "Starting run: ${RUN}"
echo "Base Model: ${BASE_MODEL}"
echo "Data Path: ${DATA_PATH}"
echo "Per-device batch size: ${PER_DEVICE_BATCH_SIZE}, Total Batch size: ${BATCH_SIZE}"

export WANDB_MODE=disabled

export NCCL_DEBUG=INFO
torchrun \
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes $NNODES \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    train_VL_any2any.py \
    --base_model $BASE_MODEL \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size $BATCH_SIZE \
    --per_device_batch_size $PER_DEVICE_BATCH_SIZE \
    --num_epochs $EPOCH \
    --learning_rate $LR \
    --max_length ${MAX_LENGTH} \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout 0.05 \
    --save_steps 200 \
    --logging_steps 5 \
    --bf16 \
    --grad_checkpoint \
    --deepspeed deepspeed_config/ds.config
