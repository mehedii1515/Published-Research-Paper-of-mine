# Enhanced DNA Sequence Classification Through Hyperparameter Optimized Convolutional Neural Networks

[cite_start]**Conference:** 2025 International Conference on Electrical, Computer and Communication Engineering (ECCE) [cite: 1]  
[cite_start]**DOI:** 10.1109/ECCE64574.2025.11013923 [cite: 1]  
[cite_start]**Date:** 13-15 February 2025 [cite: 2]  

## 📌 Abstract
[cite_start]DNA sequence classification is a fundamental task in bioinformatics, with applications ranging from gene prediction to disease diagnosis[cite: 18]. [cite_start]This repository contains the methodology and results for a parameter-tuned Convolutional Neural Network (CNN) designed to classify DNA sequences[cite: 62].

[cite_start]By transforming raw DNA sequences into numerical representations through $k$-mer feature extraction and employing extensive hyperparameter tuning, the proposed model achieves a classification accuracy of **97%** on both the **Promoter** and **Splice** datasets[cite: 21, 24]. [cite_start]This performance outperforms baseline machine learning models and previous state-of-the-art methods[cite: 25, 201].

## 👥 Authors
* [cite_start]**Md Mehedi Hasan** (Port City International University) [cite: 6, 8]
* [cite_start]**Rayhanuzzaman** (International Islamic University Chittagong) [cite: 9, 10]
* [cite_start]**Hura Jannat Poly** (Port City International University) [cite: 11, 13]
* [cite_start]**Azmain Yakin Srizon** (Rajshahi University of Engineering & Technology) [cite: 15, 16]

## 🛠 Methodology

[cite_start]The classification pipeline consists of five key steps: dataset selection, preprocessing, CNN architecture design, hyperparameter tuning, and training strategies[cite: 65].

### 1. Data Preprocessing & Feature Extraction
* [cite_start]**Cleaning:** Removal of spaces, tabs, and non-DNA characters[cite: 68].
* [cite_start]**Feature Extraction:** Utilizes the $k$-mer technique to split sequences into overlapping substrings[cite: 99, 100].
* [cite_start]**Encoding:** $k$-mers are encoded as numerical vectors[cite: 102].
* [cite_start]**Configuration:** The study utilized $k=3$ and $k=4$ for extraction[cite: 104].

### 2. CNN Model Architecture
[cite_start]The architecture is designed to capture local dependencies in DNA sequences[cite: 135].
* [cite_start]**Convolutional Layers:** Multiple layers with kernel sizes of 3, 5, and 7, and filter counts of 64, 128, and 256[cite: 118, 144, 146].
* [cite_start]**Activation:** ReLU function $f(x)=max(0,x)$[cite: 116, 117].
* [cite_start]**Pooling:** MaxPooling layers for downsampling and dimensionality reduction[cite: 119, 122].
* [cite_start]**Regularization:** Dropout layers with a rate of 0.5 to prevent overfitting[cite: 123, 126].
* [cite_start]**Classification:** A Fully Connected layer (512 units) followed by a Softmax output layer[cite: 130, 131].

### 3. Optimization & Training
* [cite_start]**Hyperparameter Tuning:** Grid search was used to optimize learning rate (0.0001 - 0.01), kernel size, filter count, and number of layers (2-4)[cite: 138, 142, 148].
* **Callbacks:**
    * [cite_start]`EarlyStopping`: Monitors validation loss to stop redundant epochs[cite: 153].
    * [cite_start]`ReduceLROnPlateau`: Dynamically adjusts the learning rate when validation loss plateaus[cite: 160].
    * [cite_start]`ModelCheckpoint`: Saves the model with the best validation loss[cite: 168].

## 📊 Datasets

[cite_start]The model was evaluated on two biological datasets[cite: 20]:

| Dataset | Classes | Samples | Sequence Length | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Promoter** | 2 (Positive/Negative) | 106 | 57 bp | [cite_start]Gene activation sequences[cite: 84, 85]. |
| **Splice** | 3 (EI, IE, N) | 3,190 | 60 bp | [cite_start]Splice junctions (Exon/Intron)[cite: 88, 89]. |

## 📈 Results

[cite_start]The parameter-tuned CNN achieved superior performance compared to traditional ML models (SVM, KNN, Random Forest) and previous deep learning approaches[cite: 206].

### Promoter Dataset Metrics
| Metric | Class 0 | Class 1 | Overall |
| :--- | :--- | :--- | :--- |
| **Precision** | 1.00 | 0.94 | [cite_start]**0.97** [cite: 191] |
| **Recall** | 0.94 | 1.00 | [cite_start]**0.97** [cite: 191] |
| **F1-score** | 0.97 | 0.97 | [cite_start]**0.97** [cite: 191] |
| **Accuracy** | - | - | [cite_start]**97%** [cite: 191] |

### Splice Dataset Metrics
| Metric | Class 0 | Class 1 | Class 2 | Overall |
| :--- | :--- | :--- | :--- | :--- |
| **Precision** | 0.97 | 0.94 | 0.98 | [cite_start]**0.97** [cite: 194] |
| **Recall** | 0.95 | 0.95 | 0.98 | [cite_start]**0.96** [cite: 194] |
| **F1-score** | 0.96 | 0.95 | 0.98 | [cite_start]**0.97** [cite: 194] |
| **Accuracy** | - | - | - | [cite_start]**97%** [cite: 194] |

### Comparison with State-of-the-Art
| Model | Promoter Accuracy | Splice Accuracy |
| :--- | :--- | :--- |
| **U. M. Akkaya et al. (CNN dna2vec)** | 95.45% | 96.85% |
| **Our Model (Parameter-tuned CNN)** | **97%** | **97%** |
[cite_start]*Source: [cite: 200]*

## 💻 Research Instruments & Tech Stack

* [cite_start]**Language:** Python 3.10 [cite: 175]
* [cite_start]**Framework:** TensorFlow 2.0 and Keras [cite: 176]
* [cite_start]**Preprocessing:** scikit-learn (GridSearchCV, scaling, splitting) [cite: 177, 178]
* [cite_start]**Hardware:** NVIDIA T4 GPU [cite: 180]
* [cite_start]**Visualization:** Matplotlib and Seaborn [cite: 181]

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
