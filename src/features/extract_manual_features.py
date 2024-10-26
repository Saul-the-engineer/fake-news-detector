def extract_manual_features(datapoints: List[Datapoint], optimal_credit_bins: Dict) -> List[Dict]:
    all_features = []
    for datapoint in datapoints:
        features = {}
        features["speaker"] = datapoint.speaker
        features["speaker_title"] = datapoint.speaker_title
        features["state_info"] = datapoint.state_info
        features["party_affiliation"] = datapoint.party_affiliation
        # Compute credit score features
        datapoint = dict(datapoint)
        for feat in [
            "barely_true_count",
            "false_count",
            "half_true_count",
            "mostly_true_count",
            "pants_fire_count",
        ]:
            features[feat] = str(compute_bin_idx(datapoint[feat], optimal_credit_bins[feat]))
        all_features.append(features)
    return all_features


def extract_statements(datapoints: List[Datapoint]) -> List[str]:
    return [datapoint.statement for datapoint in datapoints]


def construct_datapoint(input: str) -> Datapoint:
    return Datapoint(
        **{
            "statement": input,
            "barely_true_count": float("nan"),
            "false_count": float("nan"),
            "half_true_count": float("nan"),
            "mostly_true_count": float("nan"),
            "pants_fire_count": float("nan"),
        }
    )





def compute_bin_idx(val: float, bins: List[float]) -> int:
    for idx, bin_val in enumerate(bins):
        if val <= bin_val:
            return idx
