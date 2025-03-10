#!/bin/bash
#SBATCH --job-name="gemma-2-2b profs"
#SBATCH --output="a.out.%j.%N.out"
#SBATCH --mem=60G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=eng-research-gpu
#SBATCH --account=hs1-cs-eng
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wtyyeh@gmail.com
#SBATCH -t 23:59:59
#SBATCH -e ~/slurm_logs/slurm-%j.err
#SBATCH -o ~/slurm_logs/slurm-%j.out

rm -rf ~/.cache/huggingface/hub/* # clear cache
nvidia-smi --gpu-reset  # Reset GPU state
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=1

srun python3 baselines/detox_edit.py --config_file gemma-2-2b.ini