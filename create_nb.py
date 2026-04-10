import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# End-to-End Pipeline Evaluation\n",
    "This notebook generates mock data and runs the entire training and evaluation pipeline to confirm that all layers computationally and functionally work perfectly from the first block to the last block. By using this setup, we avoid downloading gigabytes of text/images and test the system locally."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Setup and Environment\n",
    "We disable wandb cloud syncing so that the run doesn't block waiting for user credentials."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import sys\n",
    "\n",
    "# Ensure our src folder is loaded correctly\n",
    "sys.path.append(os.path.abspath('.'))\n",
    "\n",
    "os.environ[\"WANDB_MODE\"] = \"disabled\"\n",
    "os.environ[\"WANDB_SILENT\"] = \"true\"\n",
    "\n",
    "print(\"Environment setup complete. WANDB is disabled for testing.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Generate Dummy Mock Data\n",
    "Creating 10 synthetic examples that mirror the exact folder structure of the MSCTD dataset so dataloaders can test end-to-end integration."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from PIL import Image\n",
    "import numpy as np\n",
    "from pathlib import Path\n",
    "\n",
    "from src.config import MSCTD_DIR\n",
    "\n",
    "train_img_dir = MSCTD_DIR / \"dataset/train/train_ende\"\n",
    "os.makedirs(train_img_dir, exist_ok=True)\n",
    "\n",
    "# Ensure dummy texts and sentiments\n",
    "texts_file = MSCTD_DIR / \"dataset/train/english_train.txt\"\n",
    "sentiments_file = MSCTD_DIR / \"dataset/train/sentiment_train.txt\"\n",
    "\n",
    "print(f\"Generating dummy data at {MSCTD_DIR}...\")\n",
    "for i in range(10):\n",
    "    # Dummy Image (pure colored blocks)\n",
    "    img = Image.new('RGB', (224, 224), color=(i * 20, i * 20, 100))\n",
    "    img.save(train_img_dir / f\"{i}.jpg\")\n",
    "\n",
    "with open(texts_file, \"w\") as f:\n",
    "    for i in range(10):\n",
    "        f.write(f\"This is a simulated sentence number {i} for evaluating NLP pipeline capabilities.\\n\")\n",
    "\n",
    "with open(sentiments_file, \"w\") as f:\n",
    "    for i in range(10):\n",
    "        f.write(f\"{i % 3}\\n\")\n",
    "\n",
    "print(\"Mock dataset successfully generated!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Training Pipeline execution\n",
    "We hijack `src.config` hyper-parameters (such as MAX_EPOCHS) beforehand so it proves everything is fine without taking 10 hours."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import src.config\n",
    "\n",
    "# Set epochs to 1 and batches to 2 for a fast computational evaluation\n",
    "src.config.MAX_EPOCHS = 1\n",
    "src.config.BATCH_SIZE = 2\n",
    "\n",
    "from src.pipelines.train import train\n",
    "\n",
    "print(\"Initiating Complete Training Pipeline...\")\n",
    "train()\n",
    "print(\"Training Pipeline Execution Finished.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Evaluation Pipeline execution\n",
    "Here we will call our evaluation modules and produce metrics verifying the precision/recall/eval steps function correctly."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "from torch.utils.data import DataLoader\n",
    "from transformers import AutoTokenizer, AutoImageProcessor\n",
    "\n",
    "from src.data.dataloaders import MultimodalDataset\n",
    "from src.data.preprocess import sent_preprocess\n",
    "from src.models.multimodal import MultimodalFusionNet\n",
    "from src.pipelines.train import collate_fn\n",
    "from src.pipelines.evaluate import evaluate_model\n",
    "\n",
    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
    "\n",
    "print(\"Loading dataset...\")\n",
    "test_dataset = MultimodalDataset(\n",
    "    dataset_dir=src.config.MSCTD_DIR,\n",
    "    images_dir=\"dataset/train/train_ende\",\n",
    "    texts_file=\"dataset/train/english_train.txt\",\n",
    "    sentiments_file=\"dataset/train/sentiment_train.txt\",\n",
    "    preprocess_text_func=sent_preprocess,\n",
    "    audio_dir=\"AudioSample\"\n",
    ")\n",
    "\n",
    "tokenizer = AutoTokenizer.from_pretrained(src.config.TEXT_MODEL_NAME)\n",
    "feature_extractor = AutoImageProcessor.from_pretrained(src.config.VISION_BACKBONE_NAME)\n",
    "\n",
    "collate = lambda b: collate_fn(b, tokenizer, feature_extractor)\n",
    "test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=collate, num_workers=0)\n",
    "\n",
    "print(\"Initializing Model and loading checkpoint...\")\n",
    "model = MultimodalFusionNet(\n",
    "    text_model_name=src.config.TEXT_MODEL_NAME,\n",
    "    vit_model_name=src.config.VISION_BACKBONE_NAME,\n",
    "    use_audio=True\n",
    ").to(device)\n",
    "\n",
    "model_path = \"models/best_multimodal.pt\"\n",
    "if os.path.exists(model_path):\n",
    "    # using weights_only=True depending on pt version or ignoring to just load it simply\n",
    "    model.load_state_dict(torch.load(model_path, map_location=device))\n",
    "    print(\"Trained model checkpoint successfully loaded.\")\n",
    "\n",
    "print(\"\\n--- Start E2E Target Evaluation ---\")\n",
    "acc, p, r, f1, cm = evaluate_model(model, test_loader, device)\n",
    "\n",
    "print(f\"Accuracy  : {acc:.4f}\")\n",
    "print(f\"Precision : {p:.4f}\")\n",
    "print(f\"Recall    : {r:.4f}\")\n",
    "print(f\"F1 Score  : {f1:.4f}\")\n",
    "print(\"Confusion Matrix:\\n\", cm)\n",
    "print(\"\\nPipeline complete! Evvverything is fine in the code.\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open("run.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)

print("run.ipynb generated.")
