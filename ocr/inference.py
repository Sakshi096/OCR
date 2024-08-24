import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('models/ocr_model.h5', compile=False)

# Character set (e.g., A-Z, a-z, 0-9, special characters)
char_list = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

def decode_label(encoded_label):
    return ''.join([char_list[int(char)] for char in encoded_label])

# Load a new image
image = cv2.imread('data/raw/new_image.png', cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (128, 32))  # Resize to match training size
image = image / 255.0  # Normalize

# Predict the text in the image
predicted_label = model.predict(np.expand_dims(image, axis=0))
decoded_label = decode_label(predicted_label[0])

print(f'The detected text is: {decoded_label}')
