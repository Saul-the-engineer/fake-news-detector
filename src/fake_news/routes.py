import logging
from typing import List

import numpy as np
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from fake_news.model.model_config import random_forest
from fake_news.model.tree_based import RandomForestModel
from fake_news.settings import Settings
from fake_news.utils.features import construct_datapoint

# Set up logging
logging.basicConfig(
    format="%(levelname)s - %(asctime)s - %(filename)s - %(message)s",
    level=logging.DEBUG,
)
LOGGER = logging.getLogger(__name__)

# Set up tags for the API documentation
ROUTER = APIRouter(tags=["Inference"])

# Define your data models
class Statement(BaseModel):
    text: str

class Prediction(BaseModel):
    label: float
    probs: List[float]

# Load model once at startup
model: RandomForestModel = None

def load_model(settings: Settings):
    """Load the RandomForest model."""
    global model
    if model is None:
        config = settings.model_config  # Adjust this based on where the config is stored
        model = RandomForestModel(config)

@ROUTER.on_event("startup")
async def startup_event():
    """Load the model when the app starts."""
    settings = Request.state.settings  # Assuming settings are already loaded in the app state
    load_model(settings)

# Inference route for predicting fakeness
@ROUTER.post("/api/predict-fakeness", response_model=Prediction)
def predict_fakeness(statement: Statement):
    """Predict whether the statement is fake or real."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    datapoint = construct_datapoint(statement.text)  # Ensure this function is defined
    probs = model.predict([datapoint])  # Make sure `predict` method is available in your model
    label = np.argmax(probs, axis=1)  # Get the index of the max probability
    prediction = Prediction(label=label[0], probs=list(probs[0]))
    
    # Log the prediction for monitoring
    LOGGER.info(prediction)
    
    return prediction
