import argparse
import json
import logging
import os
from pathlib import Path

import pandas as pd

from src.fake_news.features.credit_binning import CreditBinComputer

logging.basicConfig(
    format="%(levelname)s - %(asctime)s - %(filename)s - %(message)s",
    level=logging.DEBUG
)
LOGGER = logging.getLogger(__name__)

def read_args():
    parser = argparse.ArgumentParser(
        description="Compute optimal credit bins from training data"
    )
    parser.add_argument(
        "--train-data-path",
        type=str,
        required=True,
        help="Path to training data JSON file"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save computed bins JSON file"
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=10,
        help="Number of bins to use (default: 10)"
    )
    return parser.parse_args()

def save_bins(bins: Dict, output_path: str):
    """Save computed bins to JSON file."""
    LOGGER.info(f"Saving computed bins to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(bins, f, indent=2)

def main():
    args = read_args()
    
    # Load training data
    LOGGER.info(f"Loading training data from {args.train_data_path}")
    train_df = pd.read_json(args.train_data_path, orient="records")
    
    # Compute optimal bins
    bin_computer = CreditBinComputer(n_bins=args.n_bins)
    optimal_credit_bins = bin_computer.compute_optimal_bins(train_df)
    
    # Save results
    save_bins(optimal_credit_bins, args.output_path)
    LOGGER.info("Done!")

if __name__ == "__main__":
    main()