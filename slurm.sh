#!/bin/bash
#SBATCH --job-name=matclip_lora
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

export CUDA_VISIBLE_DEVICES=1


# Load your shell configuration
source ~/.bashrc

# Activate conda environment
source activate matclip

# Go to project directory
cd ~/wk-ganesh/multi-modal



# Start training
python sd_3_lora.py

