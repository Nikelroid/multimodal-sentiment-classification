# Multimodal Sentiment Classification

A highly robust, production-ready MLOps repository for predicting the sentiment (Negative, Neutral, Positive) of multimodal inputs using Modern Deep Learning stacks and APIs.

## 🚀 Key Features

* **Multi-Modal Fusion**: Dynamically processes & fuses Text (`RoBERTa`), Images (`ViT`), and Audio (`wav2vec2`).
* **Clean Architecture**: Refactored from dispersed Jupyter Notebooks into a strictly typed, modular pipeline (`src/models/`, `src/data/`, `src/pipelines/`).
* **Automated Data Ingestion**: One-command aggregation from sources like MSCTD and Kaggle InstaNY100K.
* **SLURM Ready**: Contains pre-configured batch scripts to queue on clusters effortlessly.
* **Experiment Tracking**: Integrated `Weights & Biases (wandb)` to log every batch, metric, and checkpoint automatically.
* **Beautiful FastAPI Server**: Production interface wrapped in a stunning glassmorphism UI.

## 📁 Repository Structure

* `app/`: FastAPI application server and UI templates.
* `data/`: Internal datastore handling downloaded and processed dataset files.
* `notebooks/`: Contains `test_development.ipynb` - a single unified playground for Jupyter experimentation.
* `slurm/`: Job submission files.
* `src/`: Core logic (Configuration, Dataloaders, Deep Learning Models, Preprocessors).

## 🛠 Setup & Installation

1. Create a `.env` file from the example:
```bash
cp .env.example .env
# Edit .env with your keys (Kaggle API, Github, WandB)
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🧠 Training & Pipelines

To execute the full lifecycle on a SLURM queue:
```bash
# Load your system envs properly in setup_env.sh
bash slurm/setup_env.sh
```

Or run steps manually:
```bash
python src/data/ingestion.py   # Download datasets
python src/pipelines/train.py  # Train Multimodal Network
```

## 🌐 Running the UI Web Server
Start the frontend interface and inference engine locally:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Navigate to `http://localhost:8000/`. You can submit text, upload images and standard wav files to generate predictions instantly.

## Audio Processing Note 🎵
Audio feature extraction is completely optional. Provide a `.wav` file to the UI, and it routes dynamically through `wav2vec2`. If audio is omitted, the framework gracefully applies zeros to the fusion space without crashing.
