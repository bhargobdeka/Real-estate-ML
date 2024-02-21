import pickle
import numpy as np
from fastapi import FastAPI
from typing import List
from pydantic import BaseModel
# import uvicorn

# Create a fastAPI instance
app = FastAPI(title="Permit Approval Time Prediction", \
    description="API to predict the permit approval time in batches", version="1.1")

class RealEstate(BaseModel):
    batches: List[List[float]]

# Load the trained model
# pathname = "/Users/bhargobdeka/Desktop/Projects/Real-estate-ML/with_batch/app/trained_model.pkl"
pathname = "/app/trained_model.pkl"
def load_model():
    with open(pathname, "rb") as f:
        global model
        model = pickle.load(f)

@app.post("/predict")
def predict(realestate:RealEstate):
    # Convert the data to a numpy array
    data = np.array(realestate.batches)
    # Load the model
    load_model()
    # Make a prediction
    prediction = model.predict(data).tolist()
    prediction = np.round(np.exp(prediction)).tolist()
    return {"prediction": prediction}

# if __name__ == "__main__":
#     uvicorn.run(app, host="localhost", port=8080)