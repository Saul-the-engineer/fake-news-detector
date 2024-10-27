from typing import (
    Any,
    Dict,
    List,
)

import numpy as np


def convert_to_standard_types(data: Dict[str, Any]):
    """Recursively convert numpy types to standard Python types."""
    if isinstance(data, dict):
        return {k: convert_to_standard_types(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_standard_types(item) for item in data]
    elif isinstance(data, np.int64):
        return int(data)  # Convert to standard Python int
    elif isinstance(data, np.float64):
        return float(data)  # Convert to standard Python float
    else:
        return data  # Return other types unchanged
