from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Define the request body
class PredictionRequest(BaseModel):
    features: list

# Initialize FastAPI app
app = FastAPI()

# Load the trained model
with open('models/trained_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define a prediction endpoint
@app.post("/predict")
def predict(request: PredictionRequest):
    # Convert request features to numpy array
    features = np.array(request.features).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(features)
    prediction_label = "malignant" if prediction[0] == 0 else "benign"
    
    # Return prediction result
    return {"prediction": prediction_label}
