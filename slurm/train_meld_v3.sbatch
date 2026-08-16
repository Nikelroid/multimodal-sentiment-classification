#!/bin/bash
#SBATCH --job-name=meld_v3
#SBATCH --output=slurm/meld_v3_%j.out
#SBATCH --error=slurm/meld_v3_%j.err
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --account=msoleyma_946
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=48GB
#SBATCH --cpus-per-task=16

set -u
cd "$SLURM_SUBMIT_DIR"

module purge
module load gcc/13.3.0

export PYTHONNOUSERSITE=1
export HF_HOME=/scratch1/kelidari/hf_cache
export TMPDIR=/scratch1/kelidari/tmp
mkdir -p "$TMPDIR"
PY=/home1/kelidari/miniconda/envs/ser_env/bin/python

PROC=/scratch1/kelidari/meld/processed
EXT=/scratch1/kelidari/meld/extracted

cp models/meld_metrics.json models/meld_metrics_v2.json 2>/dev/null || true

echo "=== 0/4 multi-frame extraction (25/50/75%) + MTCNN crops ==="
for SPLIT in train dev test; do
  $PY -u src/data/extract_frames_multi.py --extracted "$EXT" \
      --processed "$PROC" --split "$SPLIT" --workers "$SLURM_CPUS_PER_TASK" || exit 1
  $PY -u src/data/extract_faces.py \
      --images_dir "$PROC/frames3/$SPLIT" --out_dir "$PROC/faces3/$SPLIT" || exit 1
done

echo "=== 1/4 text B (ModernBERT-large, 8-turn context) ==="
$PY -u src/pipelines/train_meld_text.py --processed "$PROC" \
    --model_name answerdotai/ModernBERT-large --context 8 --max_len 320 \
    --lr 3e-5 --epochs 8 --out models/meld_text_mb.pt || exit 1

echo "=== 2/4 face (multi-frame crops, class-weighted) ==="
$PY -u src/pipelines/train_meld_face.py --processed "$PROC" \
    --multi --out models/meld_face_v3.pt || exit 1

echo "=== 3/4 audio (whisper-small, warm-started from RAVDESS 90%) ==="
$PY -u src/pipelines/train_meld_audio.py --processed "$PROC" \
    --model_name openai/whisper-small --init models/audio_sentiment_small_ft.pt \
    --out models/meld_audio_small.pt || exit 1

echo "=== 4/4 fusion v3 (RoBERTa + ModernBERT ensemble, feature-level) ==="
$PY -u src/pipelines/train_meld_fusion.py --processed "$PROC" \
    --text2_ckpt models/meld_text_mb.pt \
    --audio_ckpt models/meld_audio_small.pt --audio_model_name openai/whisper-small \
    --face_ckpt models/meld_face_v3.pt --multi_face \
    --cache_tag v3 --out models/meld_fusion_v3.pt || exit 1

echo "MELD v3 complete."
cat models/meld_metrics.json
