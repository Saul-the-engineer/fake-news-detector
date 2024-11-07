import logging
from typing import (
    Dict,
    List,
)

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

class CreditBinComputer:
    """Computes optimal credit bins for different credit-related features."""
    
    CREDIT_FEATURES = [
        "barely_true_count",
        "false_count",
        "half_true_count",
        "mostly_true_count",
        "pants_fire_count"
    ]
    
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        
    def compute_optimal_bins(self, data: pd.DataFrame) -> Dict[str, List[float]]:
        """
        Compute optimal histogram bins for credit-related features.
        
        Args:
            data: DataFrame containing credit-related columns
            
        Returns:
            Dictionary mapping feature names to their optimal bin edges
        """
        LOGGER.info(f"Computing optimal bins using {self.n_bins} bins")
        optimal_credit_bins = {}
        
        for credit_feature in self.CREDIT_FEATURES:
            if credit_feature not in data.columns:
                LOGGER.warning(f"Feature {credit_feature} not found in data")
                continue
                
            LOGGER.debug(f"Computing bins for {credit_feature}")
            bin_edges = list(np.histogram_bin_edges(
                data[credit_feature],
                bins=self.n_bins
            ))
            optimal_credit_bins[credit_feature] = bin_edges
            
        return optimal_credit_bins