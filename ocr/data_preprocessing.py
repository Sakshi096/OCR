import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_data(csv_file):
    # Load annotations from CSV
    df = pd.read_csv(csv_file)
    images = []
    labels = []
    
    for index, row in df.iterrows():
        image = cv2.imread(row['image_path'], cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (128, 32))  # Resize images to fixed size
        images.append(image)
        labels.append(row['label'])
    
    return np.array(images), labels

def encode_labels(labels, char_list):
    char_to_idx = {char: idx for idx, char in enumerate(char_list)}
    encoded_labels = [[char_to_idx[char] for char in label] for label in labels]
    return pad_sequences(encoded_labels, padding='post')

def preprocess_data(csv_file, char_list):
    images, labels = load_data(csv_file)
    images = images / 255.0  # Normalize images
    encoded_labels = encode_labels(labels, char_list)
    return images, encoded_labels
