import logging
from typing import List

import numpy as np
from fastapi import (
    APIRouter,
    Depends,
    Request,
)
from pydantic import BaseModel

from fake_news.models.tree_based import RandomForestModel
from fake_news.utils.construct_datapoint import construct_datapoint

LOGGER = logging.getLogger(__name__)

ROUTER = APIRouter(tags=["Predictions"])


# Load model configuration
def get_model(settings: Request) -> RandomForestModel:
    """Load the model based on app settings."""
    config = {
        "model_output_path": settings.app.state.settings.model_dir,
        "featurizer_output_path": settings.app.state.settings.model_dir,
    }
    return RandomForestModel(config)


class Statement(BaseModel):
    text: str


class Prediction(BaseModel):
    label: float
    probs: List[float]


@ROUTER.post("/api/predict-fakeness", response_model=Prediction)
async def predict_fakeness(statement: Statement, model: RandomForestModel = Depends(get_model)):
    """Predict the fakeness of a statement."""
    datapoint = construct_datapoint(statement.text)
    probs = model.predict([datapoint])
    label = np.argmax(probs, axis=1)

    prediction = Prediction(label=label[0], probs=list(probs[0]))
    LOGGER.info(prediction)

    return prediction
