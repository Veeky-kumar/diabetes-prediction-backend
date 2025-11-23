from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(
    title="Diabetes Prediction API",
    description="Predict diabetes using ML model",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://diabetes-prediction-frontend-delta.vercel.app",
        "http://localhost:5173"  
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = None
scaler = None

@app.on_event("startup")
def load_model():
    global model, scaler
    try:
        model = joblib.load("models/model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        print("Model and scaler loaded successfully.")
    except FileNotFoundError:
        print("Model or scaler not found. Please run train.py first.")
        
class DiabetesInput(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

@app.get("/")
def home():
    return {"message": "Diabetes Prediction API is running!"}

@app.post("/predict")
def predict(data: DiabetesInput):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please run training first.")

    try:
        values = np.array([[
            data.Pregnancies,
            data.Glucose,
            data.BloodPressure,
            data.SkinThickness,
            data.Insulin,
            data.BMI,
            data.DiabetesPedigreeFunction,
            data.Age
        ]])

        scaled_values = scaler.transform(values)

        pred = model.predict(scaled_values)[0]
        probability = model.predict_proba(scaled_values)[0][1]

        result = "Diabetic" if pred == 1 else "Not Diabetic"

        return {
            "prediction": result,
            "probability": float(probability)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
