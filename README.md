# OCR Detection and Recognition using Deep Learning

This project implements a simple OCR system using a deep learning approach. The model is designed to detect and recognize text from images.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Evaluation](#model-evaluation)
- [Inference](#inference)
- [License](#license)

## Project Overview
The objective of this project is to develop an OCR system capable of recognizing text in images. The model uses a CNN for feature extraction and an RNN for sequence modeling, trained with CTC loss to handle varying text lengths.

## Dataset
The dataset contains images of text along with their corresponding labels. You can create your own synthetic dataset or use existing OCR datasets like SynthText or MJSynth.

## Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
