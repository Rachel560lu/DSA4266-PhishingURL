
# Phishing Website Detection

## Project Overview

This project aims to detect phishing websites by applying various machine learning and deep learning models. The main goal is to classify websites as phishing or legitimate by analyzing their features. 

## Features

- Analysis of a large dataset from the UC Irvine Machine Learning Repository
- Exploration of conventional ML models (e.g., K-Nearest Neighbors, Decision Trees, and Logistic Regression)
- Development and optimization of hybrid deep learning models (CNN-BiLSTM)
- Comprehensive preprocessing and feature engineering steps
- Performance evaluation using metrics like Accuracy, Precision, and Recall

## Directory Structure

```
/root
  |-- DSA4266_Code.ipynb               # The main code implementation
  |-- Autoencoder_tuning.py            # Autoencoder tuning
```

## Dataset

We used the **PhiUSIIL Phishing URL Dataset** from the UC Irvine Machine Learning Repository. The dataset contains 235,795 samples, with features including URL structure, webpage content characteristics, and more. It is labeled with 0 for phishing and 1 for legitimate websites.

## Approach

### Data Preprocessing
- Feature correlation analysis and removal of highly correlated features.
- Dimensionality reduction using feature selection.
- Creation of four feature subsets:
  - All features (40 features)
  - URL-related features (10 features)
  - Website-related features (21 features)
  - URL-only feature (1 feature)

### Models Used
- **K-Nearest Neighbors (KNN)**
- **Decision Trees (DT) and Random Forest (RF)**
- **Logistic Regression (LR) with Autoencoder**
- **Convolutional Neural Network (CNN)**
- **Bidirectional Long Short-Term Memory (BiLSTM)**
- **Hybrid CNN-BiLSTM model**


## Results

- Our models, particularly the CNN-BiLSTM hybrid, achieved high accuracy, demonstrating the potential of deep learning in phishing detection.
- Comprehensive evaluation highlighted the trade-offs and benefits of using different model combinations and feature subsets.

## Installation and Requirements

To replicate this project:

1. Clone this repository:
   ```bash
   git clone https://github.com/Rachel560lu/DSA4266-PhishingURL.git
   ```
2. Navigate to the project directory:
   ```bash
   cd DSA4266-PhishingURL
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
