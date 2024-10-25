"""Preprocess the raw data and save the cleaned data as JSON files."""

import argparse
import csv
import json
import os
from typing import (
    Dict,
    List,
)

from fake_news.utils.features import normalize_and_clean


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data-path", type=str, default="train_data.tsv")
    parser.add_argument("--val-data-path", type=str, default="val_data.tsv")
    parser.add_argument("--test-data-path", type=str, default="test_data.tsv")
    parser.add_argument("--output-dir", type=str, default="data/processed")
    return parser.parse_args()


def construct_full_path(relative_path: str) -> str:
    """Constructs a full path based on the project's root directory."""
    # Get the base directory (project root)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # Construct the full path
    return os.path.join(base_dir, "data", "raw", relative_path)


def read_datapoints(datapath: str) -> List[Dict]:
    with open(datapath) as f:
        reader = csv.DictReader(
            f,
            delimiter="\t",
            fieldnames=[
                "id",
                "statement_json",
                "label",
                "statement",
                "subject",
                "speaker",
                "speaker_title",
                "state_info",
                "party_affiliation",
                "barely_true_count",
                "false_count",
                "half_true_count",
                "mostly_true_count",
                "pants_fire_count",
                "context",
                "justification",
            ],
        )
        return [row for row in reader]


if __name__ == "__main__":
    args = read_args()

    # Construct full paths for the data files
    train_data_path = construct_full_path(args.train_data_path)
    val_data_path = construct_full_path(args.val_data_path)
    test_data_path = construct_full_path(args.test_data_path)

    # Read data points from the files
    train_datapoints = read_datapoints(train_data_path)
    val_datapoints = read_datapoints(val_data_path)
    test_datapoints = read_datapoints(test_data_path)

    # Normalize and clean the data
    train_datapoints = normalize_and_clean(train_datapoints)
    val_datapoints = normalize_and_clean(val_datapoints)
    test_datapoints = normalize_and_clean(test_datapoints)

    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Save the cleaned data as JSON files
    with open(os.path.join(args.output_dir, "cleaned_train_data.json"), "w") as f:
        json.dump(train_datapoints, f)

    with open(os.path.join(args.output_dir, "cleaned_val_data.json"), "w") as f:
        json.dump(val_datapoints, f)

    with open(os.path.join(args.output_dir, "cleaned_test_data.json"), "w") as f:
        json.dump(test_datapoints, f)
