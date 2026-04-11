#!/bin/bash
#SBATCH --job-name=multimodal_sentiment
#SBATCH --output=slurm/multimodal_sentiment_%j.out
#SBATCH --error=slurm/multimodal_sentiment_%j.err
#SBATCH --time=6:00:00
#SBATCH --partition=gpu
#SBATCH --account=robinjia_1822
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=4

echo "Starting pipeline..."

# Enforce strict Conda isolation to prevent ~/.local dependency interference!
export PYTHONNOUSERSITE=1

# Load conda environment properly across different initializations
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda/etc/profile.d/conda.sh"
else
    echo "Warning: Conda initialization not found. Ensure miniconda is installed."
fi

conda activate multimodal_env
echo "Active environment:"
conda info --envs | grep '*'

echo "Flawlessly synchronizing entire PyPI ecosystem (Bypassing stalled Conda Resolver)..."
conda run -n multimodal_env pip install --no-cache-dir --upgrade transformers datasets wandb httpx platformdirs click regex tqdm pyyaml librosa numpy==1.26.4

DATA_PATH="/scratch1/kelidari/multimodal_data"
DATASET_NAME="MSCTD"
MODEL="roberta-base"
VISION_MODEL="google/vit-base-patch16-224-in21k"
BATCH_SIZE=32
TASK="classification"

# Step 1: Data Ingestion
echo "Running data ingestion..."
conda run -n multimodal_env python -u src/data/ingestion.py --data_dir ${DATA_PATH}

# Step 2: Preprocessing
echo "Running data preprocessing..."
conda run -n multimodal_env python -u src/data/preprocess.py

# Step 3: Pretraining Modalities (Face, Scene, Text, Audio)
echo "Running modality pretraining..."
conda run -n multimodal_env python -u src/pipelines/pretrain.py

# Step 4: Final Multimodal Training
echo "Running main training loop..."
conda run -n multimodal_env python -u src/pipelines/train.py --task ${TASK} --data_dir ${DATA_PATH} --dataset_name ${DATASET_NAME} --model_name ${MODEL} --vision_model ${VISION_MODEL} --batch_size ${BATCH_SIZE}

# Step 5: Post-training analysis
echo "Running post-training analysis..."
conda run -n multimodal_env python -u src/pipelines/evaluate.py --task ${TASK} --data_dir ${DATA_PATH} --dataset_name ${DATASET_NAME} --model_name ${MODEL} --vision_model ${VISION_MODEL} --batch_size ${BATCH_SIZE}

echo "Pipeline completed successfully."
