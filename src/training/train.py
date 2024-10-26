import argparse
import json
import logging
import os
import random
from shutil import copy
from typing import Dict

import mlflow
import numpy as np

# from fake_news.model.transformer_based import RobertaModel
from fake_news.models.tree_based import RandomForestModel
from fake_news.utils.reader import read_json_data

# import torch


logging.basicConfig(
    format="%(levelname)s - %(asctime)s - %(filename)s - %(message)s",
    level=logging.DEBUG,
)
LOGGER = logging.getLogger(__name__)


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        type=str,
        default="src/fake_news/model/model_config/random_forest.json",
        help="Path to the config file",
    )
    return parser.parse_args()


def set_random_seed(val: int = 1) -> None:
    random.seed(val)
    np.random.seed(val)
    # # Torch-specific random-seeds
    # torch.manual_seed(val)
    # torch.cuda.manual_seed_all(val)


def get_model(config: Dict) -> RandomForestModel:
    if config["model"] == "random_forest":
        return RandomForestModel(config)
    # Add more models here as needed
    else:
        raise ValueError(f"Invalid model type {config['model']} provided")


def setup_mlflow(config: Dict, config_file: str) -> str:
    """Set up MLflow experiment and return the model output path."""
    mlflow.set_experiment(config["model"])
    model_output_path = os.path.abspath(config["model_output_path"])
    os.makedirs(model_output_path, exist_ok=True)
    copy(config_file, model_output_path)  # Copy the config file to the model output path
    return model_output_path


def main() -> None:
    """Main entry point for training the model."""
    # Read command-line arguments
    args = read_args()

    # Read configuration file
    with open(args.config_file) as f:
        config = json.load(f)

    # Set random seed for reproducibility
    set_random_seed()

    # Set up MLflow and model output path
    model_output_path = setup_mlflow(config, args.config_file)

    # Read training, validation, and test data
    train_data = read_json_data(os.path.abspath(config["train_data_path"]))
    val_data = read_json_data(os.path.abspath(config["val_data_path"]))
    test_data = read_json_data(os.path.abspath(config["test_data_path"]))

    # Initialize model
    model = get_model(config)

    # Start MLflow run
    with mlflow.start_run() as run:
        # Save MLflow run ID
        with open(os.path.join(model_output_path, "meta.json"), "w") as f:
            json.dump({"mlflow_run_id": run.info.run_id}, f)

        # Log model training configuration
        mlflow.set_tags({"evaluate": config["evaluate"]})

        # Train the model if not in evaluate mode
        if not config["evaluate"]:
            LOGGER.info("Training model...")
            model.train(train_data, val_data, cache_featurizer=True)
            model.save(os.path.join(model_output_path, "model.pkl"))

        # Log model parameters
        mlflow.log_params(model.get_params())

        # Evaluate model performance
        LOGGER.info("Evaluating model...")
        val_metrics = model.compute_metrics(val_data, split="val")
        test_metrics = model.compute_metrics(test_data, split="test")
        LOGGER.info(f"Validation metrics: {val_metrics}")
        LOGGER.info(f"Test metrics: {test_metrics}")

        # Log metrics to MLflow
        mlflow.log_metrics(val_metrics)
        mlflow.log_metrics(test_metrics)


if __name__ == "__main__":
    """Example usage: python src/training/train.py --config-file src/fake_news/models/model_configs/random_forest.json"""
    main()
