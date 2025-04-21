## 3-Phase Induction Motor Predictive Maintenance using DSP

This repository contains code for processing vibration/audio signals from a 3-phase induction motor, extracting DSP features (Mel spectrograms), and training a deep convolutional neural network for anomaly detection (predictive maintenance).

---

### Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset Structure](#dataset-structure)
- [Usage](#usage)
  - [Data Processing](#data-processing)
  - [Model Training](#model-training)
  - [Evaluation and Visualization](#evaluation-and-visualization)
- [Project Structure](#project-structure)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project demonstrates a pipeline for predictive maintenance of a 3-phase induction motor using digital signal processing (DSP) and deep learning. The key steps are:

1. **Feature Extraction**: Convert raw `.wav` recordings into Mel spectrograms.
2. **Dataset Preparation**: Load processed features into a PyTorch Dataset.
3. **Model Architecture**: Define a ResNet-inspired CNN with residual blocks and dropout.
4. **Training**: Train the network to distinguish between normal and anomalous motor sounds.
5. **Evaluation**: Assess performance on held-out test data and visualize metrics.

---

## Features

- Processing of raw audio files into `.npy` Mel spectrograms.
- PyTorch `Dataset` and `DataLoader` for efficient batching.
- Residual CNN architecture with dropout for regularization.
- Training loop with learning rate scheduler and CUDA support.
- Train/test split, accuracy tracking, and visualizations of loss & accuracy.

---

## Prerequisites

- Python 3.7+
- `librosa`
- `numpy`
- `matplotlib`
- `torch` (PyTorch)
- `scikit-learn`
- `tqdm`

Install dependencies via:
```bash
pip install -r requirements.txt
```

---

## Installation

1. **Clone the repository**:
   ```bash
git clone https://github.com/yourusername/induction-motor-pdm.git
cd induction-motor-pdm
```

2. **Install dependencies**:
   ```bash
pip install -r requirements.txt
```

---

## Dataset Structure

Organize your audio data as follows:
```
source_test/                # Original `.wav` files
├── section_01_source_test_anomaly_0047.wav
├── section_02_source_test_normal_0045.wav
└── ...
```

Processed features will be saved under `Additional_Dataset_Processed/` matching the original directory structure, but with `.npy` files of Mel spectrograms.

---

## Usage

### Data Processing

Run the data preprocessing script to extract Mel spectrograms:
```python
from processing import process_data, get_mel_spect

dataset_path = "source_test"
output_path = "Additional_Dataset_Processed"

process_data(dataset_path, output_path, get_mel_spect)
```

This walks through `source_test/`, computes Mel spectrograms for each `.wav`, and saves them as `.npy` files in `Additional_Dataset_Processed/`.

### Model Training

Define and train the CNN model:
```python
from model import DeepCNN
from dataset import AudioDataset
from torch.utils.data import DataLoader

# Prepare dataset and dataloader
data_dir = "Additional_Dataset_Processed"
dataset = AudioDataset(data_dir)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Initialize model and train
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepCNN(num_classes=2).to(device)
model.train_model(dataloader, num_epochs=20)
```

### Evaluation and Visualization

Split data, evaluate on test split, and plot metrics:
```python
from evaluation import train_and_evaluate, plot_metrics

# train_loader, test_loader = ...
train_losses, train_accs, test_accs = train_and_evaluate(
    model, train_loader, test_loader, num_epochs=10)
plot_metrics(train_losses, train_accs, test_accs)
```

---

## Project Structure

```
├── processing.py        # Data processing and feature extraction
├── dataset.py           # PyTorch Dataset definition
├── model.py             # Model architecture (Residual CNN)
├── evaluation.py        # Training and evaluation loops, plotting
├── requirements.txt     # Python package dependencies
└── README.md            # This file
```

---

## Results

After 20 epochs of training on GPU, the model achieves:
- **Training Accuracy**: ~94.6%
- **Test Accuracy**: varies (check plotted curves in `plot_metrics`)

Visualizations of loss and accuracy help identify overfitting and performance trends.

---

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests.

---

## License

Distributed under the MIT License. See `LICENSE` for details.


