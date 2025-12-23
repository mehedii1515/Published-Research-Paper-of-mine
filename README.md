# Enhanced DNA Sequence Classification Through Hyperparameter Optimized Convolutional Neural Networks

**Conference:** 2025 International Conference on Electrical, Computer and Communication Engineering (ECCE)  
**DOI:** 10.1109/ECCE64574.2025.11013923  
**Date:** 13-15 February 2025  

## 📌 Abstract
DNA sequence classification is a fundamental task in bioinformatics, with applications ranging from gene prediction to disease diagnosis. This repository contains the methodology and results for a parameter-tuned Convolutional Neural Network (CNN) designed to classify DNA sequences.

By transforming raw DNA sequences into numerical representations through $k$-mer feature extraction and employing extensive hyperparameter tuning, the proposed model achieves a classification accuracy of **97%** on both the **Promoter** and **Splice** datasets. This performance outperforms baseline machine learning models and previous state-of-the-art methods.

## 👥 Authors
* **Md Mehedi Hasan** (Port City International University)
* **Rayhanuzzaman** (International Islamic University Chittagong)
* **Hura Jannat Poly** (Port City International University)
* **Azmain Yakin Srizon** (Rajshahi University of Engineering & Technology)

## 🛠 Methodology

The classification pipeline consists of five key steps: dataset selection, preprocessing, CNN architecture design, hyperparameter tuning, and training strategies.

### 1. Data Preprocessing & Feature Extraction
* **Cleaning:** Removal of spaces, tabs, and non-DNA characters.
* **Feature Extraction:** Utilizes the $k$-mer technique to split sequences into overlapping substrings.
* **Encoding:** $k$-mers are encoded as numerical vectors.
* **Configuration:** The study utilized $k=3$ and $k=4$ for extraction.

### 2. CNN Model Architecture
The architecture is designed to capture local dependencies in DNA sequences.
* **Convolutional Layers:** Multiple layers with kernel sizes of 3, 5, and 7, and filter counts of 64, 128, and 256.
* **Activation:** ReLU function $f(x)=max(0,x)$.
* **Pooling:** MaxPooling layers for downsampling and dimensionality reduction.
* **Regularization:** Dropout layers with a rate of 0.5 to prevent overfitting.
* **Classification:** A Fully Connected layer (512 units) followed by a Softmax output layer.

### 3. Optimization & Training
* **Hyperparameter Tuning:** Grid search was used to optimize learning rate (0.0001 - 0.01), kernel size, filter count, and number of layers (2-4).
* **Callbacks:**
    * `EarlyStopping`: Monitors validation loss to stop redundant epochs.
    * `ReduceLROnPlateau`: Dynamically adjusts the learning rate when validation loss plateaus.
    * `ModelCheckpoint`: Saves the model with the best validation loss.

## 📊 Datasets

The model was evaluated on two biological datasets:

| Dataset | Classes | Samples | Sequence Length | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Promoter** | 2 (Positive/Negative) | 106 | 57 bp | Gene activation sequences. |
| **Splice** | 3 (EI, IE, N) | 3,190 | 60 bp | Splice junctions (Exon/Intron). |

## 📈 Results

The parameter-tuned CNN achieved superior performance compared to traditional ML models (SVM, KNN, Random Forest) and previous deep learning approaches.

### Promoter Dataset Metrics
| Metric | Class 0 | Class 1 | Overall |
| :--- | :--- | :--- | :--- |
| **Precision** | 1.00 | 0.94 | **0.97** |
| **Recall** | 0.94 | 1.00 | **0.97** |
| **F1-score** | 0.97 | 0.97 | **0.97** |
| **Accuracy** | - | - | **97%** |

### Splice Dataset Metrics
| Metric | Class 0 | Class 1 | Class 2 | Overall |
| :--- | :--- | :--- | :--- | :--- |
| **Precision** | 0.97 | 0.94 | 0.98 | **0.97** |
| **Recall** | 0.95 | 0.95 | 0.98 | **0.96** |
| **F1-score** | 0.96 | 0.95 | 0.98 | **0.97** |
| **Accuracy** | - | - | - | **97%** |

### Comparison with State-of-the-Art
| Model | Promoter Accuracy | Splice Accuracy |
| :--- | :--- | :--- |
| **U. M. Akkaya et al. (CNN dna2vec)** | 95.45% | 96.85% |
| **Our Model (Parameter-tuned CNN)** | **97%** | **97%** |

## 💻 Research Instruments & Tech Stack

* **Language:** Python 3.10
* **Framework:** TensorFlow 2.0 and Keras
* **Preprocessing:** scikit-learn (GridSearchCV, scaling, splitting)
* **Hardware:** NVIDIA T4 GPU
* **Visualization:** Matplotlib and Seaborn

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{hasan2025enhanced,
  title={Enhanced DNA Sequence Classification Through Hyperparameter Optimized Convolutional Neural Networks},
  author={Hasan, Md Mehedi and Rayhanuzzaman and Poly, Hura Jannat and Srizon, Azmain Yakin},
  booktitle={2025 International Conference on Electrical, Computer and Communication Engineering (ECCE)},
  year={2025},
  organization={IEEE},
  doi={10.1109/ECCE64574.2025.11013923}
}
