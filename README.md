# OCR Detection and Recognition using Deep Learning

This project implements a simple Optical Character Recognition (OCR) system using deep learning. The model is designed to detect and recognize text from images, utilizing a combination of Convolutional Neural Networks (CNNs) for feature extraction and Recurrent Neural Networks (RNNs) for sequence modeling.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Training](#model-training)
  - [Model Evaluation](#model-evaluation)
  - [Inference](#inference)
- [Model Architecture](#model-architecture)
- [Dependencies](#dependencies)
- [License](#license)

## Project Overview

The objective of this project is to develop an OCR system capable of recognizing text in images. The model uses a CNN to extract features from the input images and an RNN with CTC (Connectionist Temporal Classification) loss to predict the sequence of characters. This allows the model to handle varying lengths of text without requiring pre-segmented characters.

## Dataset

For this project, you can use an existing OCR dataset such as:
- **SynthText**: A synthetic text dataset for scene text detection.
- **MJSynth (Synth90k)**: A large-scale synthetic text dataset for scene text recognition.

Alternatively, you can create your own synthetic dataset of images containing text with labels in a CSV file.

## Installation

To set up the project locally, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/ocr_project.git
    cd ocr_project
    ```

2. Create a virtual environment and activate it:

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Project Structure

```plaintext
ocr_project/
│
├── data/
│   ├── raw/
│   │   └── images/                  # Directory containing raw images for OCR
│   ├── processed/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── annotations.csv              # CSV file with image paths and corresponding labels
│
├── notebooks/
│   └── ocr_exploration.ipynb        # Jupyter notebook for initial data exploration
│
├── src/
│   ├── data_preprocessing.py        # Script for data preprocessing
│   ├── model.py                     # Neural network model definition
│   ├── train_model.py               # Script to train the OCR model
│   ├── evaluate_model.py            # Script to evaluate the OCR model
│   └── inference.py                 # Script for making predictions on new images
│
├── models/
│   └── ocr_model.h5                 # Trained OCR model
│
├── utils/
│   ├── image_utils.py               # Utility functions for image processing
│   └── data_utils.py                # Utility functions for data handling
│
├── requirements.txt                 # List of dependencies
└── README.md                        # Project documentation

