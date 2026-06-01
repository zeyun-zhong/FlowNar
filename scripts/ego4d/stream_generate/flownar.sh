#!/bin/bash
#SBATCH --job-name=evaluation
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --time=10:00:00
#SBATCH --partition=accelerated-h100
#SBATCH --cpus-per-task=32

source ~/miniconda3/etc/profile.d/conda.sh
conda activate flownar

deepspeed stream_generate.py \
    --live_version live1+ \
    --eval_datasets ego4d_segment_summary_val \
    --per_device_eval_batch_size 1 \
    --evaluation_strategy no \
    --prediction_loss_only False \
    --save_strategy no \
    --logging_steps 1 \
    --dataloader_num_workers 16 \
    --report_to tensorboard \
    --max_num_frames 1200 \
    --attn_implementation sdpa \
    --bf16 True \
    --data_root /hkfs/work/workspace_haic/scratch/on3546-Dataset_Long_new/FlowNar-Data/ego4d \
    --output_dir outputs/1B_visionmemory_narrations \
    --resume_from_checkpoint zeyun-zhong/flownar-1B-ego4d \
    --llm_pretrained meta-llama/Llama-3.2-1B-Instruct \
    --vision_mask True \
    --enable_vision_memory True \
    --num_m_tokens 20 \
    --finetune_modules connector clustering \
    --output_attentions False
