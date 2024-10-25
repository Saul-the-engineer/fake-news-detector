"""Compute optimal bins for credit columns."""

import argparse
import json
import os

import numpy as np
import pandas as pd


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data-path", type=str, default="data/processed/train.json")
    parser.add_argument("--output-path", type=str, default="data/processed/optimal_credit_bins.json")
    return parser.parse_args()


def construct_full_path(relative_path: str) -> str:
    """Construct full path based on the relative path."""
    return os.path.abspath(relative_path)

if __name__ == "__main__":
    args = read_args()
    
    # Construct full path for the train data file
    train_data_path = construct_full_path(args.train_data_path)
    
    # Read the training data
    train_df = pd.read_json(train_data_path, orient="records")
    
    optimal_credit_bins = {}
    for credit_name in ["barely_true_count",
                        "false_count",
                        "half_true_count",
                        "mostly_true_count",
                        "pants_fire_count",]:
        optimal_credit_bins[credit_name] = list(np.histogram_bin_edges(train_df[credit_name], bins=10))
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(args.output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Save the optimal bins to a JSON file
    with open(args.output_path, "w") as f:
        json.dump(optimal_credit_bins, f)

    print(optimal_credit_bins)  # Print the bins for confirmation
