import pickle
import numpy as np
from fastapi import FastAPI
# from contextlib import asynccontextmanager
from pydantic import BaseModel
# import uvicorn

# Create a fastAPI instance
app = FastAPI(title="Permit Approval Time Prediction", \
    description="API to predict the permit approval time in days", version="1.0")

# Load the trained model
def load_model():
    with open("/app/trained_model.pkl", "rb") as f:
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
    return {"message": "Welcome to the Permit Approval Time Prediction API"}
    
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
    return {"prediction": np.round(np.exp(prediction[0]))}   

# if __name__ == "__main__":
#     uvicorn.run(app, host="localhost", port=8080)