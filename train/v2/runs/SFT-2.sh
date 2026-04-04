#!/bin/bash
# Single GPU reproduction for SFT Phase 2: Multi-turn Trajectory Tuning
# Global Batch Size = 64
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model /root/autodl-tmp/output/qwen2.5-vl-poirot-v1/v1-20260206-011519/checkpoint-921 \
    --dataset /root/autodl-tmp/train2.jsonl \
    --train_type full \
    --freeze_vit true \
    --freeze_aligner true \
    --freeze_llm false \
    --output_dir output/qwen2.5-vl-poirot-v2 \
    --max_length 24576 \
    --max_pixels 501760 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.03 \
    --num_train_epochs 3 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 3 \
    --logging_steps 10 \
    --gradient_checkpointing true