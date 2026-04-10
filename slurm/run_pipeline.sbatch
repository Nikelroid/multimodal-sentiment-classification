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
