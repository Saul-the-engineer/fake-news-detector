import argparse
import json
import logging
import os
import random
from shutil import copy

import mlflow
import numpy as np

#from fake_news.model.transformer_based import RobertaModel
from fake_news.model.tree_based import RandomForestModel
from fake_news.utils.reader import read_json_data

#import torch


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


if __name__ == "__main__":
    # Read arguments
    args = read_args()

    # Read config
    with open(args.config_file) as f:
        config = json.load(f)

    # Set random seed
    set_random_seed(42)

    # Start MLflow run
    mlflow.set_experiment(config["model"])

    # Get base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("Base Directory:", base_dir)

    # Get model output path
    model_output_path = os.path.join(base_dir, config["model_output_path"])
    print("model_output_path", model_output_path)

    # Update full model output path
    config["model_output_path"] = model_output_path

    # Create model output directory
    os.makedirs(model_output_path, exist_ok=True)

    # Copy config to model directory
    copy(args.config_file, model_output_path)

    # Train and evaluate model
    with mlflow.start_run() as run:
        # Save meta information
        with open(os.path.join(model_output_path, "meta.json"), "w") as f:
            json.dump({"mlflow_run_id": run.info.run_id}, f)
        # Log tags
        mlflow.set_tags({"evaluate": config["evaluate"]})

        # Construct full paths for the data files
        train_data_path = os.path.abspath(config["train_data_path"])
        val_data_path = os.path.abspath(config["val_data_path"])
        test_data_path = os.path.abspath(config["test_data_path"])
        print("train_data_path", train_data_path)

        # Read data
        LOGGER.info("Reading data...")
        train_datapoints = read_json_data(train_data_path)
        val_datapoints = read_json_data(val_data_path)
        test_datapoints = read_json_data(test_data_path)

        # Initialize model based on config
        if config["model"] == "random_forest":
            config["featurizer_output_path"] = os.path.join(
                base_dir, config["featurizer_output_path"]
            )
            model = RandomForestModel(config)
        # elif config["model"] == "roberta":
        #     model = RobertaModel(config)
        else:
            raise ValueError(f"Invalid model type {config['model']} provided")

        # Train the model if not in evaluate mode
        if not config["evaluate"]:
            LOGGER.info("Training model...")
            model.train(train_datapoints, val_datapoints, cache_featurizer=True)
            if config["model"] == "random_forest":
                # Cache model weights on disk
                model.save(os.path.join(model_output_path, "model.pkl"))

        # Log model parameters
        mlflow.log_params(model.get_params())
        LOGGER.info("Evaluating model...")
        val_metrics = model.compute_metrics(val_datapoints, split="val")
        LOGGER.info(f"Val metrics: {val_metrics}")
        test_metrics = model.compute_metrics(test_datapoints, split="test")
        LOGGER.info(f"Test metrics: {test_metrics}")
        mlflow.log_metrics(val_metrics)
        mlflow.log_metrics(test_metrics)
        mlflow.log_metrics(val_metrics)
        mlflow.log_metrics(test_metrics)
