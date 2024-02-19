import pickle
import numpy as np
from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel
import uvicorn

# Create a fastAPI instance
app = FastAPI(title="Real Estate Price Prediction", \
    description="API to predict the permit approval time of a house", version="1.0")

# Load the model
pathname = "/Users/bhargobdeka/Desktop/Projects/Real-estate-ML/app/trained_model.pkl"

def load_model():
    with open(pathname, "rb") as f:
        global model
        model = pickle.load(f)
    
# Create a class for the permit approval time
class RealEstate(BaseModel):
    number_of_units: float
    longitude: float
    latitude: float
    construction: float
    demolition: float
    renovation: float
    residential: float
    commercial: float
    other: float
    industrial: float
    governmental: float
    habitation: float

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Real Estate Price Prediction API"}
    
@app.post("/predict")

def predict(realestate:RealEstate):
    # Convert the data to a numpy array
    data = np.array([realestate.number_of_units, realestate.longitude, \
        realestate.latitude, realestate.construction, realestate.demolition, \
            realestate.renovation, realestate.residential, realestate.commercial, \
                realestate.other, realestate.industrial, realestate.governmental, \
                    realestate.habitation]).reshape(1, -1)
    # Load the model
    load_model()
    # Make a prediction
    prediction = model.predict(data).tolist()
    return {"prediction": np.exp(prediction[0])}   

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8080)