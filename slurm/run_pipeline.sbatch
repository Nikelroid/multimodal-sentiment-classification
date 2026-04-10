#!/bin/bash
#SBATCH --job-name=multimodal_sentiment
#SBATCH --output=slurm/multimodal_sentiment_%j.out
#SBATCH --error=slurm/multimodal_sentiment_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

echo "Starting pipeline..."

# Load conda if applicable or navigate to virtual env
# source /home/$USER/miniconda3/etc/profile.d/conda.sh
# conda activate ml_env

# Install requirements if not present
echo "Installing dependencies..."
pip install -r requirements.txt

# Step 1: Data Ingestion
echo "Running data ingestion..."
python src/data/ingestion.py

# Step 2: Preprocessing
echo "Running data preprocessing..."
python src/data/preprocess.py

# Step 3: Pretraining Modalities (Face, Scene, Text, Audio)
echo "Running modality pretraining..."
python src/pipelines/pretrain.py

# Step 4: Final Multimodal Training
echo "Running main training loop..."
python src/pipelines/train.py

# Step 5: Post-training analysis
echo "Running post-training analysis..."
python src/pipelines/post_train.py

echo "Pipeline completed successfully."
