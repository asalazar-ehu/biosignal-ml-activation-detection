# biosignal-ml-activation-detection
This repository contains the code for reproducing the experimentation of the paper entitled "Detection and Classification of Physiological Activation States Through Biosignal Analysis Using Machine Learning" published in MDPI Sensor journal. This repository covers all 4 experiments described in the paper, using Matlab and Python code.

# Detection and Classification of Physiological Activation States Through Biosignal Analysis Using Machine Learning

This repository contains the official code and implementation for the paper:  
**"Detection and Classification of Physiological Activation States Through Biosignal Analysis Using Machine Learning"** published in *Sensors (MDPI)*.

## 📝 Abstract
Monitoring autonomic nervous system responses offers significant potential for personalized health applications and early intervention. This study presents a machine learning framework for binary classification of physiological activation states using electrocardiogram (ECG) and respiratory (RESP) signals from the WESAD dataset. Stress and amusement were grouped as Activation, whereas baseline and meditation were grouped as Non-Activation. The processing pipeline incorporates R-peak detection, temporal segmentation, handcrafted feature extraction, and subject-independent evaluation. Conventional machine learning classifiers and deep learning architectures were evaluated, together with a proposed Hybrid Multi-Scale Temporal Convolutional--Conformer (MST-Conformer) combining temporal representations learned from ECG and RESP waveforms with handcrafted physiological features. Performance was assessed using 15-fold Leave-One-Subject-Out (LOSO) cross-validation. Adding respiration-derived features to ECG-derived features improved conventional machine learning performance, increasing the highest balanced accuracy from 0.752 to 0.807 and AUC from 0.802 to 0.886. The complete Hybrid MST-Conformer incorporating 27 physiological features achieved a mean accuracy of 0.825, balanced accuracy of 0.820, F1-score of 0.763, and AUC of 0.904. Subject-level analysis also revealed substantial physiological heterogeneity. These results support the potential of combining learned temporal representations with handcrafted physiological information for subject-independent physiological activation classification.

## 📂 Repository Structure
The repository is organized as follows:
* `codigo/` - Contains all the source code, scripts, and Jupyter notebooks used for data preprocessing, model training, and evaluation.
* `LICENSE` - MIT License terms.

## 🚀 Getting Started

### Prerequisites
To run the code, you will need **Python 3.x** [o R, cambia el lenguaje si es necesario] and the following libraries:
* `numpy`
* `pandas`
* `scikit-learn`
* [Añade aquí otras librerías clave que uses, como scipy, tensorflow, pytorch, etc.]

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com[TuUsuario]/biosignal-ml-activation-detection.git
   ```
2. [Opcional] If you have a requirements file, run:
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Data Availability
The dataset used in this study is publicly available for reproducibility. You can download the biosignal data from the following permanent repository:
* **Repository:** [Zenodo / Figshare / etc.]
* **DOI:** [Pega aquí el enlace al DOI de tus datos cuando lo tengas, ej: https://doi.org]

## 🔬 Usage
To replicate the results or train the models, follow these steps:
1. **Preprocessing:** Run `codigo/preprocessing.py` to clean the raw biosignals.
2. **Model Training:** Run `codigo/train.py` to train the Machine Learning classifiers.
[Modifica estos pasos según cómo se llamen tus scripts reales y cómo se ejecuten].

## ✉️ Contact
For any questions, issues, or collaborations, please contact:
* **Asier Salazar Ramirez** - asier.salazar@ehu.eus
* **Affiliation:** University of the Basque Country (UPV/EHU)
