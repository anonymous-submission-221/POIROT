# Single GPU reproduction provided for open source
# Global Batch Size = per_device_train_batch_size * gradient_accumulation_steps * num_gpus = 1 * 64 * 1 = 64
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --dataset /root/autodl-tmp/train1.jsonl \
    --train_type full \
    --freeze_vit true \
    --freeze_aligner false \
    --output_dir output/qwen2.5-vl-poirot-v1 \
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