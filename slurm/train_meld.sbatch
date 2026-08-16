#!/bin/bash
#SBATCH --job-name=meld_train
#SBATCH --output=slurm/meld_train_%j.out
#SBATCH --error=slurm/meld_train_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --account=msoleyma_946
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=48GB
#SBATCH --cpus-per-task=8

set -u
cd "$SLURM_SUBMIT_DIR"

module purge
export PYTHONNOUSERSITE=1
export HF_HOME=/scratch1/kelidari/hf_cache
export TMPDIR=/scratch1/kelidari/tmp
mkdir -p "$TMPDIR"
PY=/home1/kelidari/miniconda/envs/ser_env/bin/python

PROC=/scratch1/kelidari/meld/processed

echo "=== 1/4 text (RoBERTa-large + dialogue context) ==="
$PY -u src/pipelines/train_meld_text.py --processed "$PROC" || exit 1

echo "=== 2/4 face (FER ViT on MELD 7-emotion) ==="
$PY -u src/pipelines/train_meld_face.py --processed "$PROC" || exit 1

echo "=== 3/4 audio (Whisper layer-mix, warm-started) ==="
$PY -u src/pipelines/train_meld_audio.py --processed "$PROC" || exit 1

echo "=== 4/4 fusion + test-set report ==="
$PY -u src/pipelines/train_meld_fusion.py --processed "$PROC" || exit 1

echo "MELD training complete."
cat models/meld_metrics.json
