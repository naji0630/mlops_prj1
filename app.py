from fastapi import FastAPI
import pickle
from pydantic import BaseModel
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Load the saved model
with open("iris_model.pkl", "rb") as file:
    model = pickle.load(file)

# Define a request body structure
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Define a prediction endpoint
@app.post("/predict")
async def predict(features: IrisFeatures):
    # Convert input data to the format expected by the model
    input_data = np.array([[features.sepal_length, features.sepal_width, features.petal_length, features.petal_width]])
    
    # Perform prediction
    prediction = model.predict(input_data)
    return {"prediction": int(prediction[0])}

