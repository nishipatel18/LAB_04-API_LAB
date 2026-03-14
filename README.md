# API_Labs

This repository contains three hands-on labs focused on deploying machine learning models as APIs using different frameworks and deployment strategies.

---

## Lab 1: FLASK_GCP_LAB 

A Flask-based REST API that classifies wine into three categories using chemical properties. The app includes a Streamlit frontend and is containerized with Docker for deployment on Google Cloud Run.

**Dataset:** Wine (178 samples, 13 features, 3 classes — Barolo, Grignolino, Barbera)
**Model:** XGBoost Classifier
**Stack:** Flask, XGBoost, Streamlit, Docker, Google Cloud Run




---

## Lab 2: FastAPI_Labs 

A FastAPI-based REST API that predicts whether a breast tumor is malignant or benign. It uses Pydantic models for request/response validation and provides auto-generated Swagger docs.

**Dataset:** Breast Cancer Wisconsin (569 samples, 30 features, 2 classes — Malignant, Benign)
**Model:** Logistic Regression with StandardScaler pipeline
**Stack:** FastAPI, Pydantic, Scikit-learn

---

## Lab 3: Streamlit_Labs 

A Streamlit dashboard that serves as the frontend for the FastAPI_Labs backend. Users can input tumor measurements via sliders or upload a JSON file to get predictions.

**Dataset:** Breast Cancer Wisconsin (served via FastAPI_Labs)
**Model:** Logistic Regression (hosted by FastAPI_Labs)
**Stack:** Streamlit, Requests
