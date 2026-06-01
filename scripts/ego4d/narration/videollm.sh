#!/bin/bash
#SBATCH --job-name=training
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --partition=accelerated-h100
#SBATCH --cpus-per-task=32

source ~/miniconda3/etc/profile.d/conda.sh
conda activate flownar

deepspeed train.py --deepspeed configs/deepspeed/zero2.json \
    --live_version live1+ \
    --train_datasets ego4d_refined_narration_stream_train \
    --eval_datasets ego4d_refined_narration_stream_val \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing True \
    --evaluation_strategy no \
    --prediction_loss_only False \
    --save_strategy no \
    --learning_rate 0.0002 \
    --optim adamw_torch \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --logging_steps 1 \
    --dataloader_num_workers 16 \
    --bf16 True \
    --tf32 True \
    --report_to tensorboard \
    --max_num_frames 1200 \
    --data_root /hkfs/work/workspace_haic/scratch/on3546-Dataset_Long_new/FlowNar-Data/ego4d \
    --output_dir outputs/ego4d_refined_narration_stream_train_1B/live1+ \
    --llm_pretrained meta-llama/Llama-3.2-1B-Instruct
