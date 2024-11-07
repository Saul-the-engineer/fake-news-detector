import logging
import os
from typing import List
import numpy as np
from fastapi import FastAPI, APIRouter, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from fake_news.models.tree_based import RandomForestModel
from api.settings import Settings

# Set up logging
log_level = logging.DEBUG if os.getenv("ENV") == "development" else logging.INFO
logging.basicConfig(format="%(levelname)s - %(asctime)s - %(filename)s - %(message)s", level=log_level)
LOGGER = logging.getLogger(__name__)

# Initialize settings
settings = Settings()

# Create FastAPI app instance
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Consider specifying your frontend's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set the settings in the app's state for access in routes
app.state.settings = settings

# Create a router
ROUTER = APIRouter(tags=["Predictions"])

# Load model configuration only once
model = RandomForestModel({
    "model_output_path": settings.trained_model_dir,
    "featurizer_output_path": settings.featurizer_output_path,
    "credit_bins_path": settings.credit_bins_path,
})

class Datapoint(BaseModel):
    id: str
    statement_json: str  # Assuming it's a JSON string representation
    label: str  # Adjust as per your actual requirements
    subject: str
    speaker: str
    speaker_title: str
    state_info: str = None  # Make optional
    party_affiliation: str
    context: str
    justification: str
    barely_true_count: float
    false_count: float
    half_true_count: float
    mostly_true_count: float
    pants_fire_count: float

def construct_datapoint(input_text: str) -> Datapoint:
    return Datapoint(
        id=0,  # Default ID, you might want to change this to a unique ID generator
        statement_json={"statement": input_text},  # Assuming statement_json is a dictionary with the statement
        label=None,  # Default label; can be set to None until the model predicts it
        subject="General",  # Default subject
        speaker="Unknown",  # Default speaker
        speaker_title="Citizen",  # Default title for the speaker
        state_info=None,  # No state information provided by default
        party_affiliation="None",  # Default party affiliation
        context="General context not provided.",  # Default context message
        justification="No justification available.",  # Default justification message
        barely_true_count=0.0,  # Default count for barely true
        false_count=0.0,  # Default count for false
        half_true_count=0.0,  # Default count for half true
        mostly_true_count=0.0,  # Default count for mostly true
        pants_fire_count=0.0,  # Default count for pants on fire
    )

class Statement(BaseModel):
    text: str

class Prediction(BaseModel):
    label: float
    probs: List[float]

@ROUTER.post("/api/predict-fakeness", response_model=Prediction)
async def predict_fakeness(statement: Statement, settings: Settings = Depends(lambda: app.state.settings)):
    """Make a prediction on the fakeness of a statement."""
    datapoint = construct_datapoint(statement.text)
    
    # Make a prediction
    probs = model.predict([datapoint])  # Ensure this method matches your model's expectations
    label = np.argmax(probs, axis=1)

    prediction = Prediction(label=float(label[0]), probs=list(probs[0]))
    LOGGER.info(f"Prediction made: {prediction}")

    return prediction

# Status check endpoint
@app.get("/")
def status_check():
    """Status check endpoint."""
    return {
        "status": "running",
        "version": "1.0.0",
        "uptime": "API is up and running smoothly."
    }

# Include the router
app.include_router(ROUTER)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
