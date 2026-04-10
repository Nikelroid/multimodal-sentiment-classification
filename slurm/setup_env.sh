#!/bin/bash
# setup_env.sh
# This script sets up the environment and executes the pipeline on SLURM nodes.

echo "Setting up environment variables..."
export GITHUB_TOKEN=${GITHUB_TOKEN}
export USER=${USER}
export GITHUB_USER=${GITHUB_USER}
export WANDB_API_KEY=${WANDB_API_KEY}
export HF_TOKEN=${HF_TOKEN}
export KAGGLE_USERNAME=${KAGGLE_USERNAME}
export KAGGLE_KEY=${KAGGLE_KEY}

echo "Submitting SLURM job..."
sbatch slurm/run_pipeline.sbatch
