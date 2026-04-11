#!/bin/bash
# setup_env.sh
# Automates repository syncing and Conda environment building for USC CARC partitions.

set -e

echo "=========================================="
echo " 1.Starting CARC Environment Setup"
echo "=========================================="

# 1. GitHub Token Validation safely prompted if not in session memory
if [ -z "$GITHUB_TOKEN" ]; then
    echo "GitHub PAT not found in environment. Using default fallback..."
    export GITHUB_TOKEN=""
fi

# 2. WandB Configuration for automated tracking
if [ -z "$WANDB_API_KEY" ]; then
    echo "Warning: WANDB_API_KEY not found."
fi

REPO_URL="https://${GITHUB_USER}:${GITHUB_TOKEN}@github.com/nikelroid/multimodal-sentiment-classification.git"
REPO_NAME="multimodal-sentiment-classification"
WORK_DIR="/home1/$USER"

echo "Navigating to root workstation: $WORK_DIR"
cd $WORK_DIR

# 2. Seamless clone or update via git
if [ ! -d "$REPO_NAME" ]; then
    echo "Repository not found locally. Cloning..."
    git clone "$REPO_URL" "$REPO_NAME"
    echo "Successfully cloned!"
else
    echo "Repository found. Executing aggressive pull for latest changes..."
    cd $REPO_NAME
    git remote set-url origin "$REPO_URL"
    git pull
    cd ..
fi

echo "=========================================="
echo " 2.Preparing Miniconda"
echo "=========================================="

export PATH="$HOME/miniconda3/bin:$HOME/miniconda/bin:$PATH"

# 3. Bypass Conda license prompts that freeze standard bash sessions.
echo "Preemptively accepting Conda Terms of Service..."
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

cd $REPO_NAME

# 4. Environment Syncing
ENV_NAME="multimodal_env"
echo "Analyzing target environment footprint: $ENV_NAME"

if conda env list | grep -q "$ENV_NAME"; then
    echo "$ENV_NAME payload detected! Synchronizing missing packages..."
    conda env update -f environment.yml --prune
else
    echo "$ENV_NAME payload not found. Constructing fresh conda environment..."
    conda env create -f environment.yml
fi



echo "=========================================="
echo " 3. Pre-Caching Models"
echo "=========================================="
echo "Downloading models on login node to prevent Slurm timeouts..."
conda run -n $ENV_NAME python -c "
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor, AutoConfig
models_txt = ['roberta-base']
models_img = ['google/vit-base-patch16-224-in21k']
models_aud = ['facebook/wav2vec2-base']

for m in models_txt:
    print(f'Caching text model {m}...')
    AutoConfig.from_pretrained(m)
    AutoTokenizer.from_pretrained(m)
    AutoModel.from_pretrained(m, use_safetensors=True)

for m in models_img:
    print(f'Caching vision model {m}...')
    AutoConfig.from_pretrained(m)
    AutoImageProcessor.from_pretrained(m)
    AutoModel.from_pretrained(m, use_safetensors=True)

for m in models_aud:
    print(f'Caching audio model {m}...')
    AutoConfig.from_pretrained(m)
    AutoModel.from_pretrained(m, use_safetensors=True)
"

echo "=========================================="
echo " 4. Queueing Deployment"
echo "=========================================="

echo "Initializing cluster background job scheduler..."
# Sanitize script line-endings to avoid Slurm parsing errors
sed -i 's/\r$//' slurm/run_pipeline.sbatch
# Explicitly pass all key parameters to bypass potential header parsing issues
sbatch --partition=gpu slurm/run_pipeline.sbatch

echo "=========================================="
echo " Initialization & Queue Complete!"
echo " Track your job's footprint via 'squeue -u $USER'"
echo "=========================================="
