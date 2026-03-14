from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from predict import predict_data
import numpy as np

app = FastAPI()

class CancerData(BaseModel):
    """
    Pydantic BaseModel representing breast cancer tumor measurements.
    Uses the 30 features from the Breast Cancer dataset.
    For simplicity, we use the 10 most important features (mean values).
    """
    mean_radius: float
    mean_texture: float
    mean_perimeter: float
    mean_area: float
    mean_smoothness: float
    mean_compactness: float
    mean_concavity: float
    mean_concave_points: float
    mean_symmetry: float
    mean_fractal_dimension: float

class CancerResponse(BaseModel):
    response: int
    label: str

@app.get("/")
def root():
    return {"message": "Breast Cancer Prediction API - FastAPI Lab"}

@app.post("/predict", response_model=CancerResponse, status_code=status.HTTP_200_OK)
def predict(data: CancerData):
    try:
        # Convert input to numpy array (pad with zeros for remaining 20 features)
        features = np.array([[
            data.mean_radius, data.mean_texture, data.mean_perimeter,
            data.mean_area, data.mean_smoothness, data.mean_compactness,
            data.mean_concavity, data.mean_concave_points,
            data.mean_symmetry, data.mean_fractal_dimension,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ]])

        prediction = predict_data(features)
        pred_int = int(prediction[0])
        label = "Benign" if pred_int == 1 else "Malignant"

        return CancerResponse(response=pred_int, label=label)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
