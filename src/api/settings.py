from pydantic import BaseModel
class Settings(BaseModel):
    trained_model_dir: str = "models/trained/2024-11-07/random_forest"
    featurizer_output_path: str = "models/trained/2024-11-07/random_forest"
    credit_bins_path: str = "data/processed/credit_bins.json"