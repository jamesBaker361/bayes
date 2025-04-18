#!/bin/bash

#SBATCH --partition=gpu        # Partition (job queue)

#SBATCH --requeue                 # Return job to the queue if preempted

#SBATCH --nodes=1                 # Number of nodes you require

#SBATCH --ntasks=1                # Total # of tasks across all nodes

#SBATCH --cpus-per-task=1         # Cores per task (>1 if multithread tasks)

#SBATCH --gres=gpu:1

#SBATCH --mem=32000                # Real memory (RAM) required (MB)

#SBATCH --time=3-00:00:00           # Total run time limit (D-HH:MM:SS)

#SBATCH --output=slurm/out/%j.out  # STDOUT output file

#SBATCH --error=slurm/err/%j.err   # STDERR output file (optional)

day=$(date +'%m/%d/%Y %R')
echo "gpu"  ${day} $SLURM_JOBID "node_list" $SLURM_NODELIST $@  "\n" >> jobs.txt
module purge
module load shared
module load slurm/ada-slurm/23.02.1
module load CUDA/11.7.0
source venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
export TRANSFORMERS_CACHE="/umbc/ada/donengel/common/trans_cache"
export HF_HOME="/umbc/ada/donengel/common/trans_cache"
export HF_HUB_CACHE="/umbc/ada/donengel/common/trans_cache"
export TORCH_CACHE="/umbc/ada/donengel/common/torch_cache/"
export WANDB_DIR="/umbc/ada/donengel/common/wandb"
export WANDB_CACHE_DIR="/umbc/ada/donengel/common/wandb_cache"
export HPS_ROOT="/umbc/ada/donengel/common/hps-cache"
export IMAGE_REWARD_PATH="/umbc/ada/donengel/common/reward-blob"
export IMAGE_REWARD_CONFIG="/umbc/ada/donengel/common/ImageReward/med_config.json"
srun python -u $@