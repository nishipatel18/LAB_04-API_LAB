import numpy as np
import joblib
import os
from train import run_training

# Load the trained model
model = joblib.load("model/model.pkl")

def predict_wine(features):
    """
    Predict wine class given 13 features.
    features: list of 13 float values
    """
    input_data = np.array([features])
    prediction = model.predict(input_data)
    return int(prediction[0])

if __name__ == "__main__":
    if os.path.exists("model/model.pkl"):
        print("Model loaded successfully")
    else:
        os.makedirs("model", exist_ok=True)
        run_training()
