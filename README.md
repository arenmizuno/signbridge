# SignBridge

## ADSP 31018 – Machine Learning II
### Final Project – University of Chicago  

**Instructor:** Batuhan Gundogdu, Gregory Green  
**Term:** Spring 2026  

---

## Team Members

- Aren Mizuno  
- Arthur Acker
- Jaysen Jensen 
- Lawrence Lin  

---

# Overview

SignBridge is a deep learning project focused on American Sign Language (ASL) translation using landmark-based sequence modeling. The project explores multiple neural network architectures for temporal sign classification using MediaPipe landmark sequences extracted from video.

The repository includes:
- Full preprocessing and dataset generation pipelines
- Exploratory data analysis workflows
- Partial and full dataset model training
- Transformer-based sequence modeling
- ONNX export and deployment preparation
- Real-time inference preparation using MediaPipe

The project was developed using PyTorch, Transformers, CNNs, GRUs, TCNs, and hybrid sequence architectures.

---

# Repository Structure

```text
SignBridge/
│
├── src/
│   ├── 00_Preprocessing.ipynb
│   ├── 01_EDA.ipynb
│   ├── 02_Partial_Model_Training.ipynb
│   ├── 03_Full_Model_Training.ipynb
│   └── 04_Transformer_Full_Training.ipynb
│
├── data/
│   ├── external/
│   └── processed/
│
├── models/
│   ├── checkpoints/
│   ├── logs/
│   └── onnx/
│
├── results/
│   ├── graphs/
│   ├── reports/
│   └── tables/
│
├── signbridge-ui/
│   ├── model/
│   └── signbridge-ui.py
│
└── README.md
```

---

# Dataset

This project uses the Kaggle ASL Signs competition dataset.

Dataset:
- 250 ASL sign classes
- Landmark-based temporal sequences
- Hand, pose, and facial keypoints
- Variable-length sign sequences

Kaggle Competition:
- ASL Signs Dataset
- https://www.kaggle.com/competitions/asl-signs

---

# Important Repository Notes

Large datasets and trained model checkpoints are **not included** in this GitHub repository due to GitHub file size limitations.

The following folders are excluded from version control:

```text
data/
models/checkpoints/
models/onnx/
```

These directories contain:
- Processed `.npz` and `.npy` arrays
- Full PyTorch checkpoints (`.pt`)
- ONNX deployment exports
- Quantized inference models

---

# Reproducing the Dataset

To regenerate the processed dataset:

1. Download the Kaggle ASL Signs dataset
2. Place the zip of the dataset into:

```text
data/external/
```

3. Run:

```text
src/00_Preprocessing.ipynb
```

The preprocessing notebook will:
- Clean landmark sequences
- Normalize features
- Generate train/validation/test splits
- Save processed arrays
- Create metadata and label mappings

Processed outputs will be saved into:

```text
data/processed/
```

---

# Model Development Pipeline Overview

## 1. Preprocessing

Notebook:
```text
src/00_Preprocessing.ipynb
```
The preprocessing pipeline converts the raw Kaggle ASL Signs landmark data into fixed-size numerical sequences that can be used for model training.

The raw dataset contains frame-by-frame MediaPipe landmark coordinates for each signing sequence. Each sample is first loaded from its corresponding parquet file, then filtered to keep only the most relevant landmarks for sign recognition. Instead of using every available landmark, the pipeline focuses on key facial landmarks, especially around the mouth and eyes, along with both left and right hand landmarks. This reduces noise while preserving the features most important for ASL classification.

Each sequence is transformed from a long landmark format into a wide frame-based format, where every row represents a video frame and every column represents a specific landmark coordinate. Missing landmark values are handled using interpolation, forward filling, and backward filling so that short tracking gaps do not remove otherwise useful samples. Sequences with too many missing hand detections are discarded because they do not contain enough reliable signing information.

The preprocessing step also normalizes hand positions to reduce variation between different signers, camera positions, and signing locations. In particular, hand landmarks are centered relative to the wrist and scaled by hand size, making the model focus more on hand shape and motion rather than absolute screen position.

To ensure all inputs have the same shape, each sequence is standardized to a fixed length of 96 frames. Longer sequences are center-cropped, while shorter sequences are padded. The pipeline also computes velocity features by taking frame-to-frame differences, allowing the model to learn both static landmark positions and motion patterns over time.

Finally, the processed features, labels, sequence lengths, class mappings, normalization statistics, and metadata are saved as reusable files. These outputs are used directly by the training notebooks, ensuring that the model receives clean, consistent, and deployment-ready input data.

---

## 2. Exploratory Data Analysis

Notebook:
```text
src/01_EDA.ipynb
```

EDA includes:
- Dataset dimensionality and split verification
- Class balance analysis across 250 ASL signs
- Sequence length distribution visualization
- Feature mean and standard deviation analysis
- Missing value and infinity checks
- Normalization validation
- Participant-aware split validation

Dataset summary:

| Split | Samples | Sequence Length | Features |
|-------|---------:|----------------:|----------:|
| Train | 55,642 | 96 | 708 |
| Validation | 9,291 | 96 | 708 |
| Test | 11,826 | 96 | 708 |

Additional statistics:
- Total classes: 250
- No NaN values detected across any split
- No infinite values detected across any split
- Landmark sequences were padded/truncated to 96 frames
- Each frame contains 708 landmark features from pose, hand, and facial keypoints

Key findings:
- Sequence lengths are highly right-skewed, with most signs occurring in shorter sequences while a large number of samples reach the 96-frame padding limit
- Train, validation, and test distributions remain visually consistent, indicating stable participant-aware splitting
- Feature standard deviation analysis revealed multiple feature groups with different variability ranges, reflecting differences between hand, pose, and facial landmark movement
- Normalization successfully standardized the majority of landmark features while preserving temporal variation between signs
- No missing or corrupted numerical values were detected after preprocessing, confirming dataset integrity before model training

EDA visualizations:
- Sequence length histograms
- Feature standard deviation distributions
- Split distribution comparisons
- Landmark variability analysis

Outputs are saved into:
```text
results/graphs/
results/tables/
```
---

## 3. Partial Dataset Training

Notebook:
```text
src/02_Partial_Model_Training.ipynb
```

This notebook was used for rapid benchmarking, architecture comparison, hyperparameter tuning, and stability testing on a smaller 25-sign subset before full-scale 250-sign training.

The notebook focuses on:
- Rapid architecture experimentation
- Hyperparameter tuning
- Stability and convergence testing
- Comparing temporal sequence modeling approaches
- Evaluating computational efficiency before full training

Models explored:
- MLP
- CNN
- LSTM
- GRU
- CNN + GRU
- Transformer
- CNN + Transformer
- TCN

Training setup:
- Landmark sequences padded/truncated to 96 frames
- Input dimensionality of 708 features per frame
- Data augmentation using temporal stretching, Gaussian noise, and frame dropout
- Weighted loss functions to reduce class imbalance effects
- Evaluation using Top-1, Top-3, and Top-5 accuracy

Model comparison summary:

| Model | Test Accuracy (Top 1) | Key Findings |
|---|---:|---|
| CNN | **83.1%** | Best overall performance with strong temporal-spatial feature extraction |
| CNN + GRU | 81.3% | Stable hybrid model combining convolutional and recurrent learning |
| TCN | 81.5% | Efficient sequence modeling with strong temporal performance |
| Transformer | 80.3% | Competitive global attention modeling but higher complexity |
| BiLSTM + Attention | 79.6% | Improved recurrent modeling with attention mechanisms |
| GRU | 78.9% | Strong temporal modeling with fewer parameters than LSTM |
| CNN + Transformer | 78.8% | Effective hybrid attention architecture but less stable |
| CNN + LSTM | 78.2% | Good sequential learning but slower training |
| LSTM | 75.1% | Captured temporal structure but required larger parameter counts |
| MLP | 54.1% | Poor performance due to lack of temporal sequence modeling |

Key findings:
- CNN-based architectures consistently outperformed pure recurrent models, indicating strong local temporal-spatial pattern learning from landmark sequences
- The standalone MLP performed significantly worse than sequence-aware architectures, confirming the importance of temporal modeling for ASL recognition
- GRU and LSTM models captured temporal dependencies effectively but required substantially larger parameter counts and longer training time
- Hybrid architectures such as CNN + GRU improved stability and accuracy by combining convolutional feature extraction with recurrent temporal modeling
- TCN models achieved strong performance while maintaining efficient parallel sequence processing and lower computational cost compared to recurrent architectures
- Transformer-based models demonstrated competitive performance and improved global temporal attention, though they required additional tuning and regularization for stable convergence

Based on the partial training experiments, the following architectures were prioritized for full-scale training:

| Priority | Model | Reason |
|---|---|---|
| 1 | CNN | Highest overall accuracy with efficient training and deployment |
| 2 | CNN + GRU | Strong hybrid temporal modeling and stable convergence |
| 3 | TCN | High performance with efficient sequence processing |
| 4 | Transformer | Strong attention-based modeling for long-range temporal relationships |

These architectures were further explored in:
```text
src/03_Full_Model_Training.ipynb
```

Outputs generated:
- Training history logs
- Validation accuracy comparisons
- Ablation study tables
- Confusion matrices
- Model checkpoints

Outputs are saved into:
```text
models/checkpoints/
models/logs/
results/graphs/
results/tables/
results/reports/
```
---

## 4. Full Dataset Training

Notebook:
```text
src/03_Full_Model_Training.ipynb
```

This notebook performs full-scale training on the complete 250-sign SignBridge dataset using the strongest architectures selected from the partial-model experiments.

The notebook focuses on:
- Large-scale training across all 250 ASL sign classes
- Comparing deeper versions of the selected architectures
- Evaluating accuracy, parameter count, training time, and model stability
- Identifying the best candidate architecture for deployment-oriented refinement

Primary architectures:
- CNN
- CNN + GRU
- TCN
- Transformer

Training setup:
- Full 250-sign dataset
- Landmark sequences padded/truncated to 96 frames
- 708 landmark features per frame
- Data augmentation using temporal stretching, Gaussian noise, and frame dropout
- Weighted loss functions to reduce class imbalance effects
- Evaluation using Top-1, Top-5, and Top-10 accuracy

Full training results:

| Model | Best Val Top-1 | Test Top-1 | Test Top-5 | Test Top-10 | Parameters | Train Time | Key Finding |
|---|---:|---:|---:|---:|---:|---:|---|
| Transformer | **59.10%** | **67.95%** | **86.21%** | 89.76% | 6,077,690 | 154.77 min | Best overall Top-1 accuracy, but highest training cost |
| TCN | 58.06% | 66.26% | 85.69% | 90.04% | 1,269,370 | 44.99 min | Strong temporal performance with efficient training |
| CNN | 58.33% | 65.19% | 86.07% | **90.51%** | 1,269,370 | 43.69 min | Best Top-10 accuracy and strong efficiency |
| CNN_GRU | 57.60% | 62.97% | 83.65% | 88.52% | 5,548,666 | 133.82 min | Captures temporal dependencies but underperformed relative to simpler models |

Key findings:
- Transformer achieved the highest Top-1 accuracy, suggesting that global attention helped capture long-range temporal dependencies across sign sequences
- TCN and CNN performed very competitively while requiring far less training time than Transformer and CNN_GRU models
- CNN achieved the highest Top-10 accuracy, indicating that it often ranked the correct sign among its top predictions even when Top-1 was incorrect
- CNN_GRU had the weakest performance among the four full-scale models despite having a large parameter count and long training time
- The results suggest that convolution-based architectures remained highly effective for landmark-based ASL classification, while Transformer models provided the strongest final accuracy at greater computational cost

Based on these results, Transformer was the best-performing full-scale model by Top-1 accuracy, while CNN and TCN remained attractive options for faster and more efficient deployment.

Outputs generated:
- Full-dataset model checkpoints
- Training history logs
- Ablation comparison tables
- Classification reports
- ONNX deployment exports

Outputs are saved into:
```text
models/checkpoints/
models/logs/
models/onnx/
results/graphs/
results/tables/
results/reports/
```
---

## 5. Transformer Full Training

Notebook:
```text
src/04_Transformer_Full_Training.ipynb
```

This notebook focuses on the final Transformer-based training and deployment workflow for the SignBridge ASL recognition system. Unlike the broader full-model comparison in `03_Full_Model_Training.ipynb`, this notebook specifically develops a deeper Transformer V3 architecture with stronger regularization and deployment preparation.

The notebook focuses on:
- Transformer encoder architectures for temporal landmark modeling
- Attention-based sequence learning across full ASL sign gestures
- Body-part specific feature projection for pose, left hand, right hand, and face landmarks
- Positional encoding to preserve temporal frame order
- ONNX export, quantization, and real-time inference preparation

Training setup:
- Full 250-sign SignBridge dataset
- Landmark sequences padded/truncated to 96 frames
- 708 landmark features per frame
- Transformer V3 architecture with 512-dimensional embeddings
- 8 attention heads and 4 Transformer encoder layers
- Class-balanced weighted cross-entropy loss
- Label smoothing of 0.15 for visually similar sign classes
- AdamW optimizer with weight decay
- OneCycle learning rate scheduling
- Early stopping based on validation Top-1 accuracy

Transformer V3 architecture highlights:
- Separate projection heads for pose, left hand, right hand, and face landmark groups
- Sinusoidal positional encoding for temporal ordering
- Multi-head self-attention for global sequence-level dependencies
- Stochastic depth for additional regularization
- Mean pooling over valid non-padded frames
- Final classifier head with LayerNorm, GELU activation, and dropout

Transformer V3 results:

| Metric | Result |
|---|---:|
| Validation Top-1 Accuracy | 59.98% |
| Test Top-1 Accuracy | 68.08% |
| Test Top-5 Accuracy | 88.91% |

Key findings:
- Transformer V3 achieved strong test performance, reaching 68.08% Top-1 accuracy and 88.91% Top-5 accuracy across 250 classes
- The large Top-5 improvement suggests that the model often ranks the correct sign among its highest-confidence predictions, even when the Top-1 prediction is incorrect
- Training curves show steady reduction in training loss, but validation accuracy plateaus while training accuracy continues increasing, indicating some overfitting
- The overfitting gap suggests that the Transformer benefits from strong regularization but may still require additional tuning, more augmentation, or earlier stopping
- The confusion matrix shows that many signs are classified well, but visually similar signs still create confusion, especially when gestures share handshape, motion, or location
- Body-part specific projection improved the model’s ability to separately represent pose, hands, and facial landmarks before fusing them into the Transformer encoder
- Compared with simpler CNN/TCN models, the Transformer provides stronger global temporal modeling but comes with higher computational cost and more sensitivity to regularization choices

Deployment preparation:
- Saved the best Transformer checkpoint
- Generated training curve visualizations
- Produced confusion matrix analysis
- Created a deployment bundle containing model configuration and label mappings
- Exported the model to ONNX format
- Explored quantized ONNX export for optimized inference

Outputs generated:
- Transformer checkpoint
- Deployment bundle
- Training curves
- Confusion matrix
- ONNX export
- Quantized ONNX export

Outputs are saved into:
```text
models/checkpoints/
models/onnx/
results/graphs/
results/reports/
```
---

# Evaluation

Models are evaluated using:
- Top-1 Accuracy
- Top-5 Accuracy
- Precision / Recall / F1
- Confusion matrices
- Per-class accuracy analysis

---

# Model Exporting

The repository supports multiple deployment formats.

## ONNX
Used for optimized inference and cross-platform deployment.

## TorchScript
Used for PyTorch deployment compatibility.

## Quantization
Dynamic quantization was explored for:
- Smaller model size
- Faster CPU inference
- Mobile deployment support

---

# Application Pipeline

```text
Webcam Feed
    ↓
MediaPipe Holistic  (landmark extraction, per frame)
    ↓
Extractor           (feature construction: position + velocity)
    ↓
SignCapture         (sign boundary detection)
    ↓
ONNX Runtime        (transformer model inference)
    ↓
Top-5 Display + TTS (OpenCV overlay + macOS say command)
```

---

## Stage 1: Landmark Extraction

Each video frame is passed to **MediaPipe Holistic** (`model_complexity=1`, detection confidence ≥ 0.5, tracking confidence ≥ 0.5), which returns 3D landmark coordinates for the face, left hand, and right hand.

Not all face landmarks are used. We select a subset of **76 face landmarks** that correspond to the lips, nose, and both eyes — the same subset used during training preprocessing. These are defined as:

- **Lips:** 40 landmarks covering the full lip contour
- **Nose:** 4 landmarks
- **Right eye:** 16 landmarks
- **Left eye:** 16 landmarks

Combined with **21 landmarks per hand**, the total is **118 landmarks per frame**, each with x, y, z coordinates in MediaPipe's normalized screen space ([0, 1] for x and y).

If a landmark group is not detected in a given frame (e.g. one hand is off-screen), those slots are filled with `NaN` at extraction time and later converted to `0.0`, matching how missing data was handled in training.

---

## Stage 2: Feature Engineering

From the 118 landmarks, a **354-dimensional position vector** is constructed per frame by concatenating all x coordinates, then all y, then all z, in the order: face → left hand → right hand.

**Right-hand normalization** is applied to make the model invariant to hand position and size. The wrist landmark (index 0) is subtracted from all right-hand landmarks to centre the hand, then all coordinates are divided by the distance from the wrist to the middle finger's MCP joint (landmark 12). This normalization is applied per frame independently.

A **354-dimensional velocity vector** is then computed as the frame-to-frame difference of the position vector (zero for the first frame). Position and velocity are concatenated to form a **708-feature vector per frame**, matching the model's expected input format exactly.

---

## Stage 3: Sign Boundary Detection

A `SignCapture` state machine determines when a complete sign has been made:

- **Capture starts** as soon as either hand is detected in frame
- **Capture ends** when the hand has been absent for 10 consecutive frames (~0.33s at 30fps), provided at least 10 frames were captured
- **Maximum length** is capped at 96 frames — if the user keeps their hand in frame for longer, inference is triggered immediately
- Sequences shorter than 10 frames are discarded as accidental detections

A thumbs-up gesture (detected by checking that the thumb tip is above the thumb base while all other fingers are curled) is used as a control gesture to clear the word stream rather than trigger a sign prediction.

---

## Stage 4: Model Inference

Once a sign is complete, the captured frame sequence is assembled into a `(96, 708)` window tensor, zero-padded if shorter than 96 frames. This is passed to the model along with the true sequence length.

The model (`signbridge_v3.onnx`) is a **transformer-based classifier** trained on 250 ASL words from the ASL Signs Kaggle dataset, using the landmark preprocessing approach from Hoyeol Sohn's 1st place solution. It was exported to **ONNX format** and runs on CPU via **ONNX Runtime**, removing the PyTorch dependency at inference time and making the system lightweight and portable.

The model outputs raw logits over 250 classes. A numerically stable softmax is applied:

```python
probs = np.exp(logits - logits.max())
probs /= probs.sum()
```

The **top-5 predictions** with their confidence scores are returned.

---

## Stage 5: Output

**On-screen display:** The top-5 predictions are shown as a semi-transparent overlay panel in the top-left corner of the video feed, with confidence bars and percentage scores. The top-1 prediction is highlighted in green. A word stream bar at the bottom of the screen accumulates accepted predictions across the session, color-coded by confidence (green ≥ 30%, orange ≥ 10%, red < 10%).

**Text-to-speech:** The top-1 word is spoken aloud using macOS's built-in `say` command, routed to MacBook Pro Speakers to avoid Bluetooth audio routing issues. TTS runs in a background daemon thread so the OpenCV video loop is never blocked.

A word is only added to the stream and spoken if its confidence exceeds 5% (`CONF_THRESHOLD = 0.05`) and it is not a repeat of the previous word.

---

## Known Limitations

**Domain gap.** The model was trained on a fixed set of signers. Users with different hand sizes, signing speeds, camera distances, or lighting conditions may see significantly lower accuracy than the test set numbers suggest. In practice, standing back so the upper body is visible in frame and using good lighting improves results.

**Gap between streaming and video.** Real-time streaming is more challenging than pre-recorded video because boundary-frame noise, camera latency, and inconsistent signing speed shift live inputs away from the clean training distribution.

**Single-word prediction.** The system predicts one sign at a time with no sentence-level context or grammar model applied on top.

---

# Technologies Used

## Deep Learning
- PyTorch
- Transformers
- Temporal CNNs
- GRUs / LSTMs

## Data Processing
- NumPy
- Pandas
- Scikit-learn

## Visualization
- Matplotlib

## Computer Vision
- OpenCV
- MediaPipe

## Deployment
- ONNX Runtime
- TorchScript

---
