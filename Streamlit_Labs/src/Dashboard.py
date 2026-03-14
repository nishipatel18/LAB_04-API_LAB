import json
import requests
import streamlit as st
from pathlib import Path
from streamlit.logger import get_logger

FASTAPI_BACKEND_ENDPOINT = "http://localhost:8000"
FASTAPI_CANCER_MODEL_LOCATION = Path(__file__).resolve().parents[2] / 'FastAPI_Labs' / 'model' / 'cancer_model.pkl'

DIAGNOSIS_TYPES = {0: "Malignant", 1: "Benign"}

LOGGER = get_logger(__name__)


def predict_tumor(client_input: dict) -> None:
    """Send prediction request and display result."""
    result_container = st.empty()
    try:
        with st.spinner('Predicting...'):
            response = requests.post(
                f'{FASTAPI_BACKEND_ENDPOINT}/predict',
                json=client_input,
                timeout=10
            )

        if response.status_code == 200:
            cancer_content = response.json()
            tumor_class = cancer_content.get("response")
            label = cancer_content.get("label", "")

            if tumor_class in DIAGNOSIS_TYPES:
                if tumor_class == 1:
                    result_container.success(f"Prediction: {label} (Non-cancerous) ✅")
                else:
                    result_container.error(f"Prediction: {label} (Cancerous) ⚠️")
            else:
                result_container.error("Invalid prediction response")
                LOGGER.error(f"Unexpected response: {tumor_class}")
        else:
            st.toast(f':red[Status: {response.status_code}. Check backend]', icon="🔴")
    except Exception as e:
        st.toast(':red[Backend error. Refresh and retry]', icon="🔴")
        LOGGER.error(f"Prediction error: {e}")


def run():
    st.set_page_config(page_title="Breast Cancer Prediction Demo", page_icon="🏥")

    with st.sidebar:
        # Backend status check
        try:
            backend_request = requests.get(FASTAPI_BACKEND_ENDPOINT, timeout=5)
            st.success("Backend online ✅" if backend_request.status_code == 200 else st.warning("Problem connecting 😭"))
        except requests.ConnectionError as ce:
            LOGGER.error(f"Backend error: {ce}")
            st.error("Backend offline 😱")

        st.info("Configure tumor measurements")

        mean_radius = st.slider("Mean Radius", 6.0, 30.0, 14.0, 0.1)
        mean_texture = st.slider("Mean Texture", 9.0, 40.0, 19.0, 0.1)
        mean_perimeter = st.slider("Mean Perimeter", 40.0, 190.0, 92.0, 0.1)
        mean_area = st.slider("Mean Area", 140.0, 2500.0, 655.0, 1.0)
        mean_smoothness = st.slider("Mean Smoothness", 0.05, 0.17, 0.10, 0.001)
        mean_compactness = st.slider("Mean Compactness", 0.02, 0.35, 0.10, 0.001)
        mean_concavity = st.slider("Mean Concavity", 0.0, 0.45, 0.09, 0.001)
        mean_concave_points = st.slider("Mean Concave Points", 0.0, 0.20, 0.05, 0.001)
        mean_symmetry = st.slider("Mean Symmetry", 0.10, 0.30, 0.18, 0.001)
        mean_fractal_dimension = st.slider("Mean Fractal Dimension", 0.05, 0.10, 0.06, 0.001)

        test_input_file = st.file_uploader('Upload test prediction file', type=['json'])

        if test_input_file:
            st.write('Preview file')
            test_input_data = json.load(test_input_file)
            st.json(test_input_data)
            st.session_state["IS_JSON_FILE_AVAILABLE"] = True
            st.session_state["test_input_data"] = test_input_data
        else:
            st.session_state["IS_JSON_FILE_AVAILABLE"] = False

        predict_button = st.button('Predict')

    st.write("# Breast Cancer Prediction! 🏥")

    if predict_button:
        if FASTAPI_CANCER_MODEL_LOCATION.is_file():
            if st.session_state.get("IS_JSON_FILE_AVAILABLE"):
                client_input = st.session_state["test_input_data"]['input_test']
            else:
                client_input = {
                    "mean_radius": mean_radius,
                    "mean_texture": mean_texture,
                    "mean_perimeter": mean_perimeter,
                    "mean_area": mean_area,
                    "mean_smoothness": mean_smoothness,
                    "mean_compactness": mean_compactness,
                    "mean_concavity": mean_concavity,
                    "mean_concave_points": mean_concave_points,
                    "mean_symmetry": mean_symmetry,
                    "mean_fractal_dimension": mean_fractal_dimension
                }
            predict_tumor(client_input)
        else:
            LOGGER.warning('cancer_model.pkl not found')
            st.toast(':red[Model not found. Run train.py in FastAPI_Labs]', icon="🔥")


if __name__ == "__main__":
    run()
