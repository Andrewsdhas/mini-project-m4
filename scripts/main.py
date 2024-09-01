from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

# Load the model
model = joblib.load('models/best_model.pkl')

# Define the request body
class PatientData(BaseModel):
    age: float
    gender: str
    bmi: float
    hypertension: int
    heart_disease: int
    smoking_history: str
    HbA1c_level: float
    blood_glucose_level: float

# Define the prediction endpoint
@app.post('/predict')
def predict(data: PatientData):
    """Predicts if patient has diabetes or not from patient data
        Gender - Male, Female, Other
        Smoking_history - No Info, current, never, former"""
    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df)
    return {'diabetes_status': int(prediction[0])}
