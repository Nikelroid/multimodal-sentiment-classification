#!/bin/bash
#SBATCH --job-name=meld_prep
#SBATCH --output=slurm/meld_prep_%j.out
#SBATCH --error=slurm/meld_prep_%j.err
#SBATCH --time=5:00:00
#SBATCH --partition=gpu
#SBATCH --account=msoleyma_946
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=16

set -eu
cd "$SLURM_SUBMIT_DIR"

# gcc provides the libstdc++ the conda env's wheels need; ffmpeg comes from
# imageio-ffmpeg's bundled static binary (module ffmpeg won't load off the
# default compiler stack on compute nodes).
module purge
module load gcc/13.3.0

export PYTHONNOUSERSITE=1
export HF_HOME=/scratch1/kelidari/hf_cache
export TMPDIR=/scratch1/kelidari/tmp
mkdir -p "$TMPDIR"
PY=/home1/kelidari/miniconda/envs/ser_env/bin/python

RAW=/scratch1/kelidari/meld/MELD.Raw
EXT=/scratch1/kelidari/meld/extracted
OUT=/scratch1/kelidari/meld/processed
mkdir -p "$EXT" "$OUT"

echo "=== 1/3 extract tars ==="
[ -d "$EXT/train_splits" ]                || tar -xzf "$RAW/train.tar.gz" -C "$EXT"
[ -d "$EXT/dev_splits_complete" ]         || tar -xzf "$RAW/dev.tar.gz" -C "$EXT"
[ -d "$EXT/output_repeated_splits_test" ] || tar -xzf "$RAW/test.tar.gz" -C "$EXT"
ls "$EXT"

echo "=== 2/3 audio + mid-frames + manifests ==="
for SPLIT in train dev test; do
  $PY -u src/data/preprocess_meld.py --raw "$RAW" --extracted "$EXT" \
      --out "$OUT" --split "$SPLIT" --workers "$SLURM_CPUS_PER_TASK"
done

echo "=== 3/3 MTCNN face crops ==="
for SPLIT in train dev test; do
  $PY -u src/data/extract_faces.py \
      --images_dir "$OUT/frames/$SPLIT" --out_dir "$OUT/faces/$SPLIT"
done

echo "MELD preprocessing complete."
