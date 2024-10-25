import argparse
import csv
import json
import logging
import os
from typing import (
    Dict,
    List,
)

from src.fake_news.features.preprocessing import (
    Datapoint,
    normalize_and_clean,
)

logging.basicConfig(
    format="%(levelname)s - %(asctime)s - %(filename)s - %(message)s",
    level=logging.DEBUG,
)
LOGGER = logging.getLogger(__name__)

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data-path", type=str, required=True,
                      help="Path to raw training data")
    parser.add_argument("--val-data-path", type=str, required=True,
                      help="Path to raw validation data")
    parser.add_argument("--test-data-path", type=str, required=True,
                      help="Path to raw test data")
    parser.add_argument("--output-dir", type=str, required=True,
                      help="Directory to save cleaned data")
    return parser.parse_args()

def read_datapoints(datapath: str) -> List[Dict]:
    LOGGER.info(f"Reading data from {datapath}")
    with open(datapath) as f:
        reader = csv.DictReader(
            f,
            delimiter="\t",
            fieldnames=[
                "id", "statement_json", "label", "statement", "subject",
                "speaker", "speaker_title", "state_info", "party_affiliation",
                "barely_true_count", "false_count", "half_true_count",
                "mostly_true_count", "pants_fire_count", "context",
                "justification",
            ],
        )
        return [row for row in reader]

def save_datapoints(datapoints: List[Dict], filepath: str):
    LOGGER.info(f"Saving cleaned data to {filepath}")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(datapoints, f, indent=2)

def main():
    args = read_args()
    
    # Read data
    train_datapoints = read_datapoints(args.train_data_path)
    val_datapoints = read_datapoints(args.val_data_path)
    test_datapoints = read_datapoints(args.test_data_path)

    # Clean data
    LOGGER.info("Normalizing and cleaning data...")
    train_datapoints = normalize_and_clean(train_datapoints)
    val_datapoints = normalize_and_clean(val_datapoints)
    test_datapoints = normalize_and_clean(test_datapoints)

    # Save cleaned data
    save_datapoints(
        train_datapoints, 
        os.path.join(args.output_dir, "cleaned_train_data.json")
    )
    save_datapoints(
        val_datapoints, 
        os.path.join(args.output_dir, "cleaned_val_data.json")
    )
    save_datapoints(
        test_datapoints, 
        os.path.join(args.output_dir, "cleaned_test_data.json")
    )

if __name__ == "__main__":
    main()