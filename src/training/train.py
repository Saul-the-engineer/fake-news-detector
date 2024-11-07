import argparse
import json
import logging
import os
import pickle
import random
from datetime import datetime
from shutil import copy
from typing import (
    Any,
    Dict,
)

import mlflow
import numpy as np

from fake_news.models.transformer_based import RobertaModel
from fake_news.models.tree_based import RandomForestModel
from fake_news.utils.reader import read_json_data

import torch


logging.basicConfig(
    format="%(levelname)s - %(asctime)s - %(filename)s - %(message)s",
    level=logging.DEBUG,
)
LOGGER = logging.getLogger(__name__)


def read_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        type=str,
        default="src/fake_news/model/model_config/random_forest.json",
        help="Path to the config file",
    )
    return parser.parse_args()


def set_random_seed(val: int = 1) -> None:
    """Set the random seed for reproducibility."""
    random.seed(val)
    np.random.seed(val)
    # Torch-specific random-seeds
    torch.manual_seed(val)
    torch.cuda.manual_seed_all(val)


def get_model(config: Dict[str, Any]) -> RandomForestModel:
    """Instantiate the model based on the configuration."""
    if config["model"] == "random_forest":
        return RandomForestModel(config)
    elif config["model"] == "roberta":
        return RobertaModel(config)
    else:
        raise ValueError(f"Invalid model type {config['model']} provided")


def setup_mlflow(config: Dict[str, Any], config_file: str) -> str:
    """Set up MLflow experiment and return the model output path."""
    
    # Set up MLflow experiment
    mlflow.set_experiment(config["model"])

    # Create a base directory with the current date and model type
    current_date = datetime.now().strftime("%Y-%m-%d")
    model_type = config["model"]
    model_output_path = os.path.join("models", "trained", current_date, model_type)

    # Make the directory if it doesn't exist
    os.makedirs(model_output_path, exist_ok=True)

    # Copy the configuration file to the run directory
    copy(config_file, os.path.join(model_output_path, "config.json"))

    # Return the path where model files will be saved
    return model_output_path


def load_data(config: Dict[str, Any]) -> Dict[str, Any]:
    try:
        train_data = read_json_data(os.path.abspath(config["train_data_path"]))
        val_data = read_json_data(os.path.abspath(config["val_data_path"]))
        test_data = read_json_data(os.path.abspath(config["test_data_path"]))
        return {"train": train_data, "val": val_data, "test": test_data}
    except FileNotFoundError as e:
        LOGGER.error(f"Data file not found: {e}")
        raise


def train_model(
    model: RandomForestModel,
    train_data: Any,
    val_data: Any,
    model_output_path: str,
) -> None:
    """Train the model and save it to the specified path."""
    LOGGER.info("Training model...")
    model.train(train_data, val_data, cache_featurizer=False)
    model.save(os.path.join(model_output_path, "model.pkl"))


def evaluate_model(
    model: Any, data: Dict[str, Any], model_output_path: str
) -> Dict[str, Dict[str, float]]:
    logs_dir = os.path.join(model_output_path, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    LOGGER.addHandler(logging.FileHandler(os.path.join(logs_dir, "evaluation.log")))

    LOGGER.info("Evaluating model...")
    val_metrics = model.compute_metrics(data["val"], split="val")
    test_metrics = model.compute_metrics(data["test"], split="test")
    LOGGER.info(f"Validation metrics: {val_metrics}")
    LOGGER.info(f"Test metrics: {test_metrics}")

    # Convert metrics to JSON-serializable format
    val_metrics = {k: (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v)
                   for k, v in val_metrics.items()}
    test_metrics = {k: (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v)
                    for k, v in test_metrics.items()}

    # Save metrics as JSON
    metrics_dir = os.path.join(model_output_path, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    with open(os.path.join(metrics_dir, "val_metrics.json"), "w") as f:
        json.dump(val_metrics, f)
    with open(os.path.join(metrics_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f)

    return {"val_metrics": val_metrics, "test_metrics": test_metrics}


def train_and_evaluate(config: Dict[str, Any], model_output_path: str) -> None:
    data = load_data(config)
    model = get_model(config)

    with mlflow.start_run() as run:
        with open(os.path.join(model_output_path, "meta.json"), "w") as f:
            json.dump({"mlflow_run_id": run.info.run_id}, f)
        mlflow.set_tags({"evaluate": config["evaluate"]})

        if not config["evaluate"]:
            train_model(model, data["train"], data["val"], model_output_path)

        mlflow.log_params(model.get_params())

        metrics = evaluate_model(model, data, model_output_path)

        # Log metrics to MLflow
        mlflow.log_metrics(metrics["val_metrics"])
        mlflow.log_metrics(metrics["test_metrics"])


def main() -> None:
    """Main entry point for training the model."""
    # Read command-line arguments
    args = read_args()

    # Read configuration file
    with open(args.config_file) as f:
        config = json.load(f)

    # Set random seed for reproducibility
    set_random_seed(config.get("random_seed", 1))

    # Set up MLflow and model output path
    model_output_path = setup_mlflow(config, args.config_file)

    # Train and evaluate the model
    train_and_evaluate(config, model_output_path)


if __name__ == "__main__":
    """Example usage: python src/training/train.py --config-file src/fake_news/models/model_configs/random_forest.json"""
    main()
