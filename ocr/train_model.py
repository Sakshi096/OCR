from model import build_ocr_model
from data_preprocessing import preprocess_data

# Character set (e.g., A-Z, a-z, 0-9, special characters)
char_list = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# Load and preprocess data
X_train, y_train = preprocess_data('data/annotations.csv', char_list)

# Build the OCR model
model = build_ocr_model(input_shape=(32, 128, 1), num_classes=len(char_list))

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=0.2)

# Save the trained model
model.save('models/ocr_model.h5')
