# Voice Recognition Project

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=flat&logo=jupyter&logoColor=white)](https://jupyter.org/)

A comprehensive machine learning project implementing voice recognition using audio feature extraction and classification algorithms.

## Overview

This project demonstrates a complete voice recognition pipeline that analyzes audio recordings from 10 American speakers (5 men and 5 women) to build an accurate speaker identification system. The implementation uses Mel Frequency Cepstral Coefficients (MFCC) for feature extraction and explores various machine learning approaches including statistical tests, dimensionality reduction, and classification models.

## Features

- **Audio Feature Extraction**: MFCC computation for timbre analysis
- **Statistical Analysis**: Hotelling's T-squared tests for group comparisons
- **Dimensionality Reduction**: PCA and t-SNE visualization
- **Classification Models**: Naive Bayes, LDA, QDA, and Gaussian Mixture Models
- **Model Evaluation**: Comprehensive performance analysis and comparison
- **Data Visualization**: Interactive plots for exploratory data analysis

## Results

The project achieves high accuracy in voice recognition tasks:

- **Gender Classification**: 100% accuracy using Mahalanobis distance-based classification
- **Speaker Identification**: Up to 95%+ accuracy with optimized Gaussian Mixture Models (GMM)
- **Statistical Significance**: Significant differences confirmed between male and female speaker groups (p < 0.05)
- **Model Comparison**: Comprehensive evaluation of Naive Bayes, LDA, QDA, and GMM approaches

## Technical Implementation

### Audio Processing
- Load and preprocess WAV audio files
- Extract 10 MFCC coefficients per audio sample
- Temporal averaging for fixed-length feature vectors

### Machine Learning Pipeline
- Data standardization and preprocessing
- Statistical hypothesis testing
- Dimensionality reduction techniques
- Multiple classification algorithms with hyperparameter tuning
- Cross-validation and performance metrics

### Key Algorithms
- **MFCC Extraction**: Time-frequency domain feature extraction
- **PCA**: Principal Component Analysis for dimensionality reduction
- **Hotelling's T² Test**: Multivariate statistical testing
- **Gaussian Mixture Models**: Probabilistic modeling for speaker distributions
- **Discriminant Analysis**: LDA and QDA for classification

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/voice-recognition-project.git
   cd voice-recognition-project
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Demo

Run the demo script to see the core audio processing pipeline:
```bash
python demo.py
```

This will generate synthetic audio, extract MFCC features, and create visualizations.

## Data

The audio recordings used in this project are not included in the repository due to size and privacy considerations. The dataset consists of WAV files from 10 American speakers (5 male, 5 female) with 50 recordings each.

To run the notebook, you'll need to:
1. Create a `data/raw/` directory
2. Add WAV audio files following the naming convention: `{gender}{speaker_id:04d}_us_{gender}{speaker_id:04d}_{recording_id:05d}.wav`
   - gender: 'm' or 'f'
   - speaker_id: 0001-0005 for males, 0001-0005 for females (mapped to speaker IDs 0-9)
   - recording_id: 00001-00050

## Usage

1. Ensure the `data/raw` folder contains the audio recordings
2. Open the Jupyter notebook:
   ```bash
   jupyter notebook voice_recognition_project.ipynb
   ```
3. Run all cells to execute the complete analysis pipeline

## Project Structure

```
voice-recognition-project/
├── voice_recognition_project.ipynb  # Complete analysis notebook with:
│   ├── Professional introduction and overview
│   ├── Audio feature extraction (MFCC)
│   ├── Statistical analysis and hypothesis testing
│   ├── Dimensionality reduction (PCA, t-SNE)
│   ├── Multiple ML model implementations
│   ├── Comprehensive model evaluation
│   ├── Conclusions and technical insights
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation
├── LICENSE                          # MIT License
└── data/
    └── raw/                         # Audio recordings (not included)
```
## Results

The project achieves high accuracy in speaker identification through:
- Gender classification with 100% accuracy using Mahalanobis distance
- Speaker identification using optimized GMM models
- Comprehensive model comparison and selection

## License

MIT License - see LICENSE file for details.
