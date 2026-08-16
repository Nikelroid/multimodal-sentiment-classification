# Multimodal Sentiment Classification

[![CI](https://github.com/Nikelroid/multimodal-sentiment-classification/actions/workflows/ci.yml/badge.svg)](https://github.com/Nikelroid/multimodal-sentiment-classification/actions/workflows/ci.yml)
[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen)](https://kelidari.com/multimodal-sentiment-classification/)

End-to-end MLOps repository for predicting sentiment (Negative / Neutral / Positive)
from multimodal inputs — text, image, and optional audio — trained on the MSCTD
dialogue dataset. Covers the full lifecycle: data ingestion, GPU training with a
modern fine-tuning recipe, tested pipelines under CI, containerized serving, and a
live browser playground with webcam and voice capture.

**Live demo:** <https://kelidari.com/multimodal-sentiment-classification/>

## 📊 Results (MSCTD test split, n=5,067)

| Model | Text encoder | Vision encoder | Test acc | Macro-F1 | Checkpoint |
|---|---|---|---|---|---|
| Baseline (legacy recipe) | RoBERTa-base | ViT-B/16 | 62.6% | 0.607 | 0.8 GB |
| **Fusion-Large (default)** | **RoBERTa-large** (mean-pooled) | **DINOv2-large** | **62.2%** | **0.601** | 1.3 GB (fp16) |
| Fusion-Base (modern recipe) | RoBERTa-base | DINOv2-base | 60.9% | 0.594 | 0.8 GB |

The top two are statistically tied at this test size (±1.4% at 95% CI) —
MSCTD's text+image sentiment signal saturates around ~62%, consistent with
published results on this dataset. Fusion-Large ships as the default: it pairs
the tied-best test score with the soundest selection methodology (validation
macro-F1, early stopping) and is the released checkpoint.

Training recipe for the modern runs: discriminative learning rates (backbones
2e-5, fusion head 5e-4), cosine schedule with warmup, label smoothing 0.1,
gradient clipping, bf16 autocast, and best-model selection on validation
macro-F1 with early stopping. The released checkpoint stores weights in fp16
(1.3 GB) and casts to fp32 on load.

### 🎙 Speech-tone model (voice sentiment)

MSCTD ships no audio, so the voice pathway is trained on real emotional speech
— **RAVDESS + CREMA-D (9,758 clips)** mapped to the same sentiment ids — and
fused with the multimodal model at decision level:

| Component | Val accuracy (speaker-independent) | Checkpoint |
|---|---|---|
| Whisper-base encoder (frozen) + learned layer mix + MLP head | **82.1%** | 40 MB (fp16) |

Two techniques worth noting:

- **Learnable layer weighting** (SUPERB-style) over all frozen Whisper encoder
  layers. Training assigned the highest weight (0.21) to **layer 3 — the exact
  middle of the encoder** — confirming that intermediate layers carry the most
  prosodic/emotional information.
- **Confidence-modulated late fusion**: each component's modality prior is
  scaled by its prediction certainty (1 − normalized entropy), so a confident
  angry voice outweighs an unsure text read, and voice takes the lead
  automatically when no text is provided.

## 🚀 Key Features

* **Multi-Modal Fusion**: text (`RoBERTa-large`), image (`DINOv2-large`), optional audio (`wav2vec2`) — concatenated and classified by a LayerNorm/GELU head. Backbones are config-swappable; embedding sizes are derived automatically.
* **Modern training pipeline**: AMP (bf16), discriminative LRs, cosine warmup schedule, label smoothing, validation-split model selection on macro-F1, early stopping.
* **Tested + linted under CI**: 16 unit tests over configs, datasets, and collation run on every push (`.github/workflows/ci.yml`).
* **One-click CD**: `.github/workflows/deploy-cloudrun.yml` pulls the released checkpoint and ships a scale-to-zero Cloud Run service (see `deploy/README.md`).
* **Experiment tracking**: integrated `Weights & Biases`.
* **Live playground**: static `docs/` site (GitHub Pages) with sample prompts, image upload **or webcam capture**, audio upload **or in-browser voice recording** (client-side 16 kHz WAV encoding), pointing at any inference endpoint.

## 📁 Repository Structure

* `app/`: FastAPI inference server.
* `docs/`: static playground site (GitHub Pages).
* `deploy/`: Cloud Run deployment guide + slim serving requirements.
* `src/`: core logic (configs, dataloaders, models, train/evaluate pipelines).
* `tests/`: unit tests (run by CI).
* `slurm/`: cluster job submission files.

## 🛠 Setup

```bash
pip install -r requirements.txt
```

## 🧠 Training & Evaluation

```bash
python src/data/ingestion.py                          # download datasets
python src/pipelines/train.py --data_dir /path/to/data
python src/pipelines/evaluate.py --data_dir /path/to/data   # test-split metrics
```

Backbones, learning rates, epochs, and the audio switch live in `config.yml`.

## 🌐 Serving

```bash
uvicorn app.main:app --port 8000
```

The server loads `models/best_multimodal.pt` (train one, or download the release
asset) and exposes `POST /predict` plus a simple UI at `/`. The `docs/`
playground can point at `http://localhost:8000` or a deployed Cloud Run URL.

## ☁️ Deployment

See `deploy/README.md` — one `gcloud run deploy` command, or the
**Deploy to Cloud Run** GitHub Action for release-based one-click deploys.

## Audio Processing Note 🎵

Audio is optional end-to-end: the playground records 16 kHz mono WAV in the
browser, and when `models/audio_sentiment.pt` is present the API runs the
speech-tone model and fuses it with the text+image prediction (the response's
`components` field shows each modality's read). Train your own with
`python src/pipelines/train_audio.py --audio_dir <RAVDESS> <CREMA-D>`; a
two-stage fine-tune (`--unfreeze_top N`) and a Whisper-small variant
(`slurm/train_audio_whisper_small.job`) are included.
