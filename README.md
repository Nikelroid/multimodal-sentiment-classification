# Multi-Modal Sentiment Classification

## Sharif University of Technology, EE Dept. <br/> Deep Learning Graduate Course, Dr. E. Fatemizadeh

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![Transformers](https://img.shields.io/badge/Transformers-4.0+-yellow.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)

![concept image](multimodal.jpeg)


## Description
This project implements a deep learning pipeline for **Multi-Modal Sentiment Analysis**, specifically designed to process and classify sentiment in image-text conversations. Built using **PyTorch**, the system leverages state-of-the-art models to analyze both visual and textual data. It utilizes **EfficientNet** for image feature extraction and **BERT** for textual embeddings, fusing these modalities to predict sentiments (Positive, Negative, Neutral).

The project is structured into distinct phases, evolving from basic data handling to complex multimodal fusion strategies, making it a comprehensive resource for understanding how to integrate Natural Language Processing (NLP) and Computer Vision (CV).

## Features
* **Robust Data Handling:** Custom PyTorch DataLoaders designed for the MSCTD (Multi-Modal Sentiment Classification and Time Dynamics) dataset.
* **Visual Sentiment Analysis:**
    * Face detection and extraction using **MTCNN** and **RetinaFace**.
    * Feature extraction using pre-trained **EfficientNet-B2**.
* **Textual Sentiment Analysis:**
    * Preprocessing pipelines involving tokenization, lemmatization, and stop-word removal.
    * Implementation of classical methods (**TF-IDF**, **SVM**) and deep learning approaches (**Word2Vec**, **BERT**).
* **Multimodal Fusion:** Strategies to concatenate and fuse embedded vectors from both image and text models for superior classification performance.
* **Model Evaluation:** Comprehensive metrics including Accuracy, Precision, Recall, and F1-Score with confusion matrix visualizations.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/nikelroid/multi-modal-sentiment-classification.git](https://github.com/nikelroid/multi-modal-sentiment-classification.git)
    cd multi-modal-sentiment-classification
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install torch torchvision torchaudio
    pip install transformers scikit-learn pandas numpy matplotlib seaborn
    pip install mtcnn pyenchant nltk
    ```

3.  **Download NLTK data (if required):**
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    ```

## Usage

The project is divided into four phases. You can run the Jupyter Notebooks for each phase sequentially:

### Phase 0: Data Preparation
* **Goal:** Initialize data loaders and prepare the MSCTD dataset.
* **Run:** `Phase-0/project_phase0.ipynb`

### Phase 1: Visual Analysis
* **Goal:** Extract facial features and analyze images for sentiment.
* **Run:** `Phase-1/Phase1-Part1.ipynb`, `Phase-1/Phase1-Part2.ipynb`, etc.

### Phase 2: Textual Analysis
* **Goal:** Train NLP models to classify sentiment based on text dialogue.
* **Run:** `Phase-2/Phase2.ipynb`

### Phase 3: Multimodal Fusion
* **Goal:** Combine pre-trained visual and textual models to train a final classifier.
* **Run:** `Phase-3/Phase3_Part1.ipynb`

## Contributing
Contributions are welcome! Please follow these steps:
1.  Fork the repository.
2.  Create a new feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

## License
This project is distributed under the MIT License. See `LICENSE` for more information.

## Contact/Support
For questions or support, please open an issue in this repository or contact the original participants:
* Nima Kelidari
* Ali Abbasi
* Amir Ahmad Shafiee

