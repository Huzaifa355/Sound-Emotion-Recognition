# 🎙️ Speech Emotion Recognition (SER)

Two deep learning approaches for classifying emotions from raw audio — trained on a custom labeled voice dataset using **TensorFlow** and **Librosa**.

---

## Overview

This repository contains two independent training pipelines for Speech Emotion Recognition, each using a different audio feature representation and model architecture. Both are implemented as Google Colab notebooks with full checkpoint management for resumable training.

| | Approach 1 | Approach 2 |
|---|---|---|
| **Notebook** | `mfcc_CNN_LSTM.ipynb` | `3_CHANNEL_MEL_SPECTROGRAM.ipynb` |
| **Feature** | 40-dim MFCC | 3-Channel Mel Spectrogram (Mel + Δ + ΔΔ) |
| **Architecture** | CNN + LSTM | CNN (Rectangular Kernels) + Bi-LSTM + Attention |
| **Input Shape** | `(188, 40)` | `(94, 128, 3)` |
| **Epochs** | 150 | 90 |
| **Precision** | Mixed Float16 | Mixed Float16 |

---

## Approaches

### Approach 1 — MFCC + CNN-LSTM (`mfcc_CNN_LSTM.ipynb`)

Extracts **40 Mel-Frequency Cepstral Coefficients** per frame and feeds the time-series into a stacked CNN-LSTM architecture.

**Feature Extraction**
- 40 MFCCs at 16 kHz, padded/truncated to 188 time steps
- Augmentation (training only): light Gaussian noise (~40 dB SNR), ±1 semitone pitch shift, 0.95–1.05× time stretch, 20% no-augmentation pass-through

**Model Architecture**
```
Input (188, 40)
  → Conv2D(32) + BN + Dropout(0.4)
  → Conv2D(64) + BN + Dropout(0.4)
  → Conv2D(128) + BN + Dropout(0.4)
  → LSTM(128)
  → Dense(128, relu) + Dropout(0.4)
  → Dense(N, softmax)
```

**Training Configuration**
- Optimizer: Adam with Cosine Decay LR schedule
- Loss: Label Smoothing Cross-Entropy
- L2 regularization: 0.0008
- Batch size: 16 | Max epochs: 150
- Early stopping: patience 25 (val loss) + patience 10 (LR reduce)
- Checkpointing: every 5 epochs to Google Drive

---

### Approach 2 — 3-Channel Mel Spectrogram + Bi-LSTM + Attention (`3_CHANNEL_MEL_SPECTROGRAM.ipynb`)

Stacks Mel spectrogram, first-order delta, and second-order delta (delta-delta) as a 3-channel image — capturing both spectral content and its temporal dynamics.

**Feature Extraction**
- 128 Mel bands, hop length 512, 3-second window at 16 kHz
- Output shape: `(94, 128, 3)` — time × frequency × channel
- Channel 0: log Mel spectrogram (dB)
- Channel 1: Δ (first derivative — rate of change)
- Channel 2: ΔΔ (second derivative — acceleration)
- Per-channel z-score normalization

**Model Architecture**
```
Input (94, 128, 3)
  → Conv2D(64, 5×3) + BN + ReLU + MaxPool(2×2) + Dropout   ← time focus
  → Conv2D(128, 3×5) + BN + ReLU + MaxPool(1×2) + Dropout  ← freq focus
  → Conv2D(256, 3×3) + BN + ReLU + MaxPool(1×2) + Dropout  ← texture
  → Reshape → Bidirectional LSTM(128)
  → Custom Attention Layer
  → Dense(256, relu) + Dropout(0.4)
  → Dense(N, softmax)
```

**Key Design Decisions**
- **Rectangular CNN kernels** (5×3 and 3×5) explicitly separate time and frequency learning
- **Bidirectional LSTM** with `return_sequences=True` captures context in both directions
- **Custom Attention Layer** learns to suppress silence and padding, focusing on emotionally salient frames
- **Warmup + Cosine Decay** LR schedule: linear warmup for first 10% of steps, then smooth cosine decay

**Training Configuration**
- Optimizer: Adam with WarmUp + Cosine Decay (`target_lr=1e-4`)
- Loss: Label Smoothing Cross-Entropy
- L2 regularization: 0.001
- Batch size: 16 | Max epochs: 90
- Transfer learning support: load previous checkpoint and fine-tune

---

## Dataset

Both notebooks expect the dataset organized as one folder per emotion class:

```
DataSet/
├── angry/
│   ├── sample_001.wav
│   └── ...
├── happy/
├── neutral/
├── sad/
└── ...        ← any number of emotion classes (auto-detected)
```

Emotion classes are **auto-detected** from folder names at runtime — no hardcoding required. The dataset is loaded from Google Drive as a zip archive.

> This project was trained on a custom voice dataset. Compatible with standard SER benchmarks like **RAVDESS**, **TESS**, **CREMA-D**, or **SAVEE** if organized in the same folder structure.

---

## Requirements

```bash
tensorflow >= 2.12
librosa
numpy
pandas
scikit-learn
matplotlib
seaborn
```

Both notebooks run on **Google Colab** (free tier or Pro). Mixed Float16 precision is enabled for faster training on GPU.

---

## Checkpoint Management

Both notebooks include a full checkpoint system that survives Colab disconnections:

- Periodic checkpoints saved every 5 epochs to Google Drive
- Best model always preserved (`best_model.keras`)
- Training log (`training_log.csv`) is appended across sessions
- Resume training across different Google accounts by sharing the Drive folder
- Utility functions: `inspect_checkpoints()`, `show_training_progress()`, `delete_old_checkpoints()`

---

## Results & Evaluation

After training, both notebooks generate:
- Accuracy and loss curves (train vs. validation)
- Confusion matrix (normalized and raw)
- Full `classification_report` per emotion class (precision, recall, F1)

---

## Repository Structure

```
├── mfcc_CNN_LSTM.ipynb                              # Approach 1
├── 3_CHANNEL_MEL_SPECTROGRAM_(Mel+Delta+DeltaDelta).ipynb  # Approach 2
└── README.md
```

---

## Why Two Approaches?

| Aspect | MFCC | 3-Channel Mel |
|---|---|---|
| Feature type | Compact cepstral coefficients | Full spectral image + dynamics |
| Temporal modeling | LSTM on flattened CNN features | Bi-LSTM with attention over time |
| Frequency resolution | 40 bands (perceptual) | 128 Mel bands |
| Model complexity | Lower | Higher |
| Good for | Fast inference, smaller models | Higher accuracy, richer features |

MFCCs are a classic and compact representation — fast to compute and proven in speech tasks. The 3-channel Mel approach trades compute for richer feature representation, with the delta channels explicitly encoding how the spectrum is changing over time — a strong signal for emotional prosody.

---

## Author

**Huzaifa Shafique**  
