# CLAUDE.md — AI Context for deep-learning-practice

This file gives AI assistants the context needed to work effectively in this repository.

## Purpose

Educational practice repository for a Master's Degree program. Each subdirectory is a self-contained module implementing a specific deep learning architecture from scratch using TensorFlow/Keras. Scripts are standalone (no shared internal library); they can be read and modified independently.

## Language & Style

- All code comments and printed output are in **English**.
- No type annotations are used.
- No shared utilities or internal imports across modules — each script is self-contained.
- TensorFlow/Keras functional or sequential API (not subclassing) is preferred.

## Directory Map

| Path | Status | Topic |
|---|---|---|
| `01_Autoencoders/` | Placeholder | Autoencoders, representation learning |
| `02_SOM_Clustering/` | Placeholder | Self-Organizing Maps, unsupervised clustering |
| `03_RBM_Recommender/recommender_system.py` | Implemented | Autoencoder-based book recommender (Good Books 10K) |
| `04_CNN_Classification/cnn_fashion_classifier.py` | Implemented | VGG-style CNN on Fashion MNIST (10 classes) |
| `04_CNN_Classification/cnn_horses_humans_lenet.py` | Implemented | LeNet-5 binary classifier (horses vs. humans) |
| `data/` | Empty | Local dataset cache — never commit datasets |
| `notebooks/` | Empty | Intended for Jupyter notebooks |
| `docs/` | Empty | Additional documentation |

## Key Patterns in Implemented Scripts

### Data Loading
- Datasets are downloaded at runtime via `kagglehub.dataset_download(...)`.
- A Kaggle API token at `~/.kaggle/kaggle.json` is required.
- Fashion MNIST loads directly via `tf.keras.datasets.fashion_mnist.load_data()`.

### Model Construction
All models use `tf.keras.Sequential` or `tf.keras.Model` (functional API). Standard pattern:
```python
model = tf.keras.Sequential([...])
model.compile(optimizer='adam', loss=..., metrics=['accuracy'])
model.fit(X_train, y_train, epochs=..., validation_split=0.2, callbacks=[...])
```

### Evaluation & Visualization
Every script ends with matplotlib plots of training/validation loss and accuracy curves. Classification scripts also produce a seaborn confusion matrix heatmap.

## Conventions to Follow When Adding New Modules

1. Create a new numbered subdirectory: `05_<TopicName>/`.
2. Name the main script descriptively in snake_case: `topic_description.py`.
3. Keep the script self-contained — no imports from other modules in this repo.
4. Mirror the existing pattern: data loading → preprocessing → model definition → training → evaluation/visualization.
5. Write comments and any printed output in English.
6. Do not commit datasets or model weights (`.h5`, `.keras`, saved_models folders).

## Dependencies

```
tensorflow>=2.10
numpy
pandas
matplotlib
seaborn
scikit-learn
kagglehub
```

`requirements.txt` exists but is currently empty — populate it if adding new dependencies.

## What NOT to Do

- Do not add a shared `utils/` or `lib/` module — scripts are intentionally isolated.
- Do not switch frameworks (e.g., PyTorch) without explicit instruction; the entire repo uses TensorFlow/Keras.
- Do not commit the `data/` directory contents — datasets can be large and are re-downloaded via `kagglehub`.
- Do not translate comments to Spanish unless explicitly asked.

## Planned Future Modules (Placeholders)

- `01_Autoencoders/`: Vanilla autoencoder, denoising autoencoder, variational autoencoder (VAE).
- `02_SOM_Clustering/`: Self-Organizing Map trained on tabular or image data for cluster visualization.
