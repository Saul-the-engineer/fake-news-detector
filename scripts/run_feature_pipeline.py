# pylint: disable=W0105
"""Run complete feature engineering pipeline."""
import argparse
import logging
import os
import subprocess
from pathlib import Path
from typing import List

logging.basicConfig(
    format="%(levelname)s - %(asctime)s - %(filename)s - %(message)s",
    level=logging.DEBUG
)
LOGGER = logging.getLogger(__name__)

def read_args():
    parser = argparse.ArgumentParser(
        description="Run complete feature engineering pipeline"
    )
    parser.add_argument(
        "--raw-train-data",
        type=str,
        required=True,
        help="Path to raw training data"
    )
    parser.add_argument(
        "--raw-val-data",
        type=str,
        required=True,
        help="Path to raw validation data"
    )
    parser.add_argument(
        "--raw-test-data",
        type=str,
        required=True,
        help="Path to raw test data"
    )
    parser.add_argument(
        "--processed-data-dir",
        type=str,
        required=True,
        help="Directory to save processed data"
    )
    parser.add_argument(
        "--credit-bins-path",
        type=str,
        required=True,
        help="Path to save credit bins"
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=10,
        help="Number of bins for credit features"
    )
    return parser.parse_args()

def run_script(script_path: str, args: List[str]) -> None:
    """Run a Python script with given arguments."""
    cmd = ["python", script_path] + args
    LOGGER.info(f"Running command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        LOGGER.error(f"Script failed with exit code {e.returncode}")
        raise

def main():
    args = read_args()
    
    # Create output directories
    os.makedirs(args.processed_data_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.credit_bins_path), exist_ok=True)
    
    # Step 1: Normalize and clean data
    LOGGER.info("Step 1: Normalizing and cleaning data...")
    normalize_script = "scripts/normalize_and_clean.py"
    normalize_args = [
        "--train-data-path", args.raw_train_data,
        "--val-data-path", args.raw_val_data,
        "--test-data-path", args.raw_test_data,
        "--output-dir", args.processed_data_dir
    ]
    run_script(normalize_script, normalize_args)
    
    # Step 2: Compute credit bins
    LOGGER.info("Step 2: Computing credit bins...")
    cleaned_train_data = os.path.join(args.processed_data_dir, "cleaned_train_data.json")
    credit_bins_script = "scripts/compute_credit_bins.py"
    credit_bins_args = [
        "--train-data-path", cleaned_train_data,
        "--output-path", args.credit_bins_path,
        "--n-bins", str(args.n_bins)
    ]
    run_script(credit_bins_script, credit_bins_args)
    
    LOGGER.info("Feature engineering pipeline completed successfully!")

if __name__ == "__main__":
    """
    Script ran in the following way:
    python scripts/run_feature_pipeline.py \
        --raw-train-data data/raw/train.tsv \
        --raw-val-data data/raw/val.tsv \
        --raw-test-data data/raw/test.tsv \
        --processed-data-dir data/processed \
        --credit-bins-path data/processed/credit_bins.json
    """

    main()
