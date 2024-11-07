import logging
from typing import List
import numpy as np
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from fake_news.models.tree_based import RandomForestModel
from fake_news.utils.datapoint_constructor import Datapoint
from api.settings import Settings

LOGGER = logging.getLogger(__name__)

ROUTER = APIRouter(tags=["Predictions"])

# Dependency to access settings
def get_settings() -> Settings:
    return app.state.settings  # Ensure your FastAPI app state is correctly set up

# Load model configuration
def get_model(settings: Settings) -> RandomForestModel:
    model_dir = settings.trained_model_dir
    featurizer_output_path = settings.featurizer_output_path

    config = {
        "model_output_path": model_dir,
        "featurizer_output_path": featurizer_output_path,
        "credit_bins_path": settings.credit_bins_path,
    }
    return RandomForestModel(config)

def construct_datapoint(input: str) -> Datapoint:
    return Datapoint(
        statement=input,
        barely_true_count=0.0,
        false_count=0.0,
        half_true_count=0.0,
        mostly_true_count=0.0,
        pants_fire_count=0.0,
    )

class Statement(BaseModel):
    text: str

class Prediction(BaseModel):
    label: float
    probs: List[float]

@ROUTER.post("/api/predict-fakeness", response_model=Prediction)
async def predict_fakeness(statement: Statement, settings: Settings = Depends(get_settings)):
    """Make a prediction on the fakeness of a statement."""
    model = get_model(settings)
    datapoint = construct_datapoint(statement.text)
    
    # Make a prediction
    probs = model.predict([datapoint])  # Ensure this method matches your model's expectations
    label = np.argmax(probs, axis=1)

    prediction = Prediction(label=label[0], probs=list(probs[0]))
    LOGGER.info(f"Prediction made: {prediction}")

    return prediction
