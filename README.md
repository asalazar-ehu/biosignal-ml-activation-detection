# biosignal-ml-activation-detection 
This repository contains the code for reproducing the experimentation of the paper entitled "Detection and Classification of Physiological Activation States Through Biosignal Analysis Using Machine Learning" published in MDPI Sensor journal. This repository covers all 4 experiments described in the paper, using Matlab and Python code.

# Detection and Classification of Physiological Activation States Through Biosignal Analysis Using Machine Learning

This repository contains the official code and implementation for the paper:  
**"Detection and Classification of Physiological Activation States Through Biosignal Analysis Using Machine Learning"** published in *Sensors (MDPI)*.

## 📝 Abstract
Monitoring autonomic nervous system responses offers significant potential for personalized health applications and early intervention. This study presents a machine learning framework for binary classification of physiological activation states using electrocardiogram (ECG) and respiratory (RESP) signals from the WESAD dataset. Stress and amusement were grouped as Activation, whereas baseline and meditation were grouped as Non-Activation. The processing pipeline incorporates R-peak detection, temporal segmentation, handcrafted feature extraction, and subject-independent evaluation. Conventional machine learning classifiers and deep learning architectures were evaluated, together with a proposed Hybrid Multi-Scale Temporal Convolutional--Conformer (MST-Conformer) combining temporal representations learned from ECG and RESP waveforms with handcrafted physiological features. Performance was assessed using 15-fold Leave-One-Subject-Out (LOSO) cross-validation. Adding respiration-derived features to ECG-derived features improved conventional machine learning performance, increasing the highest balanced accuracy from 0.752 to 0.807 and AUC from 0.802 to 0.886. The complete Hybrid MST-Conformer incorporating 27 physiological features achieved a mean accuracy of 0.825, balanced accuracy of 0.820, F1-score of 0.763, and AUC of 0.904. Subject-level analysis also revealed substantial physiological heterogeneity. These results support the potential of combining learned temporal representations with handcrafted physiological information for subject-independent physiological activation classification.

## 📂 Repository Structure
The repository is organized as follows:
* `experiment_1_2/` - Scripts and notebooks for Experiments 1 and 2 (Conventional ML and feature analysis).
* `experiment_3/` - Deep learning baseline models and  the proposed Hybrid MST-Conformer model (using only 9 features).
* `experiment_4/` - Implementation and training of the proposed Hybrid MST-Conformer model and conventional ML models, both using 27 features.
* `requirements.txt` - File containing all required Python libraries, in case you decide to run the code locally instead of in Google Colab.
* `LICENSE` - MIT License terms.

## 🚀 Getting Started

### Prerequisites
To run the project code, you will need **Python 3.x** and **MATLAB**. The required Python dependencies can be automatically installed using the provided requirements file. However, the uploaded python files are python notebook that can be run directly on Google Colab, for the sake of simplicity. Thus, the packages listed in the `requirements.txt` file are only needed in case you want to adapt them for being run locally in your own computer.

In any case, independently from where you decide to run the python code, you must update the path of the *PRECOMPUTED_DIR* varibale in the python codes, as well as the paths where you are going to load both the features and raw signals (variables *df_feat* and *signals_mat* in block 2A).

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
* **Repository:** Zenodo: `Dataset for: Detection and Classification of Physiological Activation States Through Biosignal Analysis Using Machine Learning`
* **DOI:** https://doi.org/10.5281/zenodo.23337725

## 🔬 Usage
Each folder contains specific scripts for its corresponding stage:
1. **Experiments 1 & 2:** Navigate to `experiment_1_and_2/` to run feature extraction and traditional ML classifiers.
2. **Experiment 3:** Check `experiment_3/` to evaluate deep learning baselines and to train the proposed multi-scale temporal hybrid model using only 9 handcrafted features.
3. **Experiment 4 (MST-Conformer):** Check `experiment_4/` to train and evaluate the proposed multi-scale temporal hybrid model under 15-fold LOSO cross-validation and the traditional ML algorithms using all the 27 handcrafted features.

## ✉️ Contact
For any questions, issues, or collaborations, please contact:
* **Asier Salazar Ramirez** - asier.salazar@ehu.eus
* **Affiliation:** University of the Basque Country (UPV/EHU)
