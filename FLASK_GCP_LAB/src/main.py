from flask import Flask, request, jsonify
from predict import predict_wine
import os

app = Flask(__name__)

# Map numeric model output to human-readable class
label_map = {
    0: "Class 0 (Barolo)",
    1: "Class 1 (Grignolino)",
    2: "Class 2 (Barbera)"
}

# Wine feature names for reference
FEATURE_NAMES = [
    "alcohol", "malic_acid", "ash", "alcalinity_of_ash",
    "magnesium", "total_phenols", "flavanoids", "nonflavanoid_phenols",
    "proanthocyanins", "color_intensity", "hue",
    "od280_od315", "proline"
]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Extract 13 features from JSON
    features = [float(data[f]) for f in FEATURE_NAMES]
    print(f"Received features: {features}")

    # Call model
    prediction = predict_wine(features)

    # Convert numeric class to label
    pred_label = label_map.get(prediction, str(prediction))

    return jsonify({'prediction': pred_label})

if __name__ == '__main__':
    app.run(
        debug=True,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080))
    )
