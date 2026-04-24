# Deep Learning Practice

A structured practice repository for a Master's Degree program, implementing core deep learning architectures using TensorFlow/Keras. Each module covers a specific topic with practical examples using real-world datasets from Kaggle.

## Repository Structure

```
deep-learning-practice/
├── 01_Autoencoders/           # Autoencoders & unsupervised representation learning (WIP)
├── 02_SOM_Clustering/         # Self-Organizing Maps for unsupervised clustering (WIP)
├── 03_RBM_Recommender/        # Autoencoder-based recommendation system
│   └── recommender_system.py
├── 04_CNN_Classification/     # Convolutional Neural Networks for image classification
│   ├── cnn_fashion_classifier.py
│   └── cnn_horses_humans_lenet.py
├── data/                      # Local dataset storage (not tracked in git)
├── docs/                      # Additional documentation
├── notebooks/                 # Jupyter notebooks (experimental)
├── requirements.txt
└── README.md
```

## Modules

### 01 — Autoencoders *(in progress)*
Planned coverage: unsupervised autoencoders, dimensionality reduction, anomaly detection.

### 02 — SOM Clustering *(in progress)*
Planned coverage: Self-Organizing Maps, topology-preserving clustering.

### 03 — RBM Recommender
**Script**: [recommender_system.py](03_RBM_Recommender/recommender_system.py)

Book recommendation system built with a symmetric Autoencoder (encoder-decoder) trained on the [Good Books 10K](https://www.kaggle.com/datasets/zygmunt/goodbooks-10k) dataset from Kaggle.

- Architecture: Dense encoder `(N → 512 → 256 → 128)` + symmetric decoder `(128 → 256 → 512 → N)`
- Activation: ReLU + Dropout(0.2) in hidden layers, Sigmoid output
- Loss: Mean Squared Error (MSE)
- Generates personalized top-N book recommendations ranked by predicted rating

### 04 — CNN Classification
**Script 1**: [cnn_fashion_classifier.py](04_CNN_Classification/cnn_fashion_classifier.py)

Multi-class image classifier on Fashion MNIST (10 clothing categories).

- Architecture: VGG-style CNN — `Conv2D(32) → Conv2D(64) → Conv2D(128) → Dense(128) → Dense(10)`
- BatchNormalization + MaxPooling after each conv block
- EarlyStopping callback (patience=5), Dropout(0.4)
- Outputs: accuracy/loss curves + confusion matrix heatmap

**Script 2**: [cnn_horses_humans_lenet.py](04_CNN_Classification/cnn_horses_humans_lenet.py)

Binary image classifier (horse vs. human) using the classic LeNet-5 architecture on the [Horses or Humans](https://www.kaggle.com/datasets/sanikamal/horses-or-humans-dataset) Kaggle dataset.

- Architecture: LeNet-5 — `Conv(6) → Pool → Conv(16) → Pool → Dense(120) → Dense(84) → Dense(1)`
- Input: 64×64 RGB images loaded via `ImageDataGenerator`
- Loss: Binary Crossentropy

## Tech Stack

| Library | Purpose |
|---|---|
| TensorFlow / Keras | Neural network construction and training |
| NumPy | Numerical operations |
| Pandas | Dataset preprocessing, pivot tables |
| Matplotlib / Seaborn | Training curves, confusion matrix visualization |
| scikit-learn | Train/test split, classification metrics |
| kagglehub | Downloading datasets directly from Kaggle |

## Setup

```bash
pip install -r requirements.txt
```

To run any script, a Kaggle account and API token (`~/.kaggle/kaggle.json`) are required for dataset downloads via `kagglehub`.

```bash
python 03_RBM_Recommender/recommender_system.py
python 04_CNN_Classification/cnn_fashion_classifier.py
python 04_CNN_Classification/cnn_horses_humans_lenet.py
```

## Notes

- Code and comments are written in Spanish (master's degree coursework).
- `data/` directory is intended for local dataset caching; it is not committed to git.
- Modules 01 and 02 are directory placeholders pending future implementation.
