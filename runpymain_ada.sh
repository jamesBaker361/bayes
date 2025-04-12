#!/bin/bash

#SBATCH --partition=main        # Partition (job queue)

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
eval "$(conda shell.bash hook)"
conda activate scenescape
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
export TRANSFORMERS_CACHE="donengel_ada/common/trans_cache"
export HF_HOME="donengel_ada/common/trans_cache"
export HF_HUB_CACHE="donengel_ada/common/trans_cache"
export TORCH_CACHE="donengel_ada/common/torch_cache/"
export WANDB_DIR="donengel_ada/common/wandb"
export WANDB_CACHE_DIR="donengel_ada/common/wandb_cache"
export HPS_ROOT="donengel_ada/common/hps-cache"
export IMAGE_REWARD_PATH="donengel_ada/common/reward-blob"
export IMAGE_REWARD_CONFIG="donengel_ada/common/ImageReward/med_config.json"
srun python3 -u $@