from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report

# Load the trained model
model = load_model('models/ocr_model.h5', compile=False)

# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
