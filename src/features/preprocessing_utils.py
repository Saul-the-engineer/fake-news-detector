import json
import logging
import os
import pickle
from copy import deepcopy
from functools import partial
from typing import (
    Dict,
    List,
    Optional,
)

import numpy as np
from pydantic import BaseModel
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import (
    FeatureUnion,
    Pipeline,
)
from sklearn.preprocessing import FunctionTransformer

from fake_news.utils.constants import (
    CANONICAL_SPEAKER_TITLES,
    CANONICAL_STATE,
    PARTY_AFFILIATIONS,
    SIX_WAY_LABEL_TO_BINARY,
)
from fake_news.utils.construct_datapoint import Datapoint

logging.basicConfig(
    format="%(levelname)s - %(asctime)s - %(filename)s - %(message)s",
    level=logging.DEBUG,
)
LOGGER = logging.getLogger(__name__)


# NOTE: Making sure that all normalization operations preserve immutability of inputs
def normalize_labels(datapoints: List[Dict]) -> List[Dict]:
    normalized_datapoints = []
    for datapoint in datapoints:
        # First do simple cleaning
        normalized_datapoint = deepcopy(datapoint)
        normalized_datapoint["label"] = SIX_WAY_LABEL_TO_BINARY[datapoint["label".lower().strip()]]
        normalized_datapoints.append(normalized_datapoint)
    return normalized_datapoints


def normalize_and_clean_speaker_title(datapoints: List[Dict]) -> List[Dict]:
    normalized_datapoints = []
    for datapoint in datapoints:
        # First do simple cleaning
        normalized_datapoint = deepcopy(datapoint)
        old_speaker_title = normalized_datapoint["speaker_title"]
        old_speaker_title = old_speaker_title.lower().strip().replace("-", " ")
        # Then canonicalize
        if old_speaker_title in CANONICAL_SPEAKER_TITLES:
            old_speaker_title = CANONICAL_SPEAKER_TITLES[old_speaker_title]
        normalized_datapoint["speaker_title"] = old_speaker_title
        normalized_datapoints.append(normalized_datapoint)
    return normalized_datapoints


def normalize_and_clean_party_affiliations(datapoints: List[Dict]) -> List[Dict]:
    normalized_datapoints = []
    for datapoint in datapoints:
        normalized_datapoint = deepcopy(datapoint)
        if normalized_datapoint["party_affiliation"] not in PARTY_AFFILIATIONS:
            normalized_datapoint["party_affiliation"] = "none"
        normalized_datapoints.append(normalized_datapoint)
    return normalized_datapoints


def normalize_and_clean_state_info(datapoints: List[Dict]) -> List[Dict]:
    normalized_datapoints = []
    for datapoint in datapoints:
        normalized_datapoint = deepcopy(datapoint)
        old_state_info = normalized_datapoint["state_info"]
        old_state_info = old_state_info.lower().strip().replace("-", " ")
        if old_state_info in CANONICAL_STATE:
            old_state_info = CANONICAL_STATE[old_state_info]
        normalized_datapoint["state_info"] = old_state_info
        normalized_datapoints.append(normalized_datapoint)
    return normalized_datapoints


def normalize_and_clean_counts(datapoints: List[Dict]) -> List[Dict]:
    normalized_datapoints = []
    for idx, datapoint in enumerate(datapoints):
        normalized_datapoint = deepcopy(datapoint)
        for count_col in [
            "barely_true_count",
            "false_count",
            "half_true_count",
            "mostly_true_count",
            "pants_fire_count",
        ]:
            if count_col in normalized_datapoint:
                normalized_datapoint[count_col] = float(normalized_datapoint[count_col])
        normalized_datapoints.append(normalized_datapoint)
    return normalized_datapoints


def normalize_and_clean(datapoints: List[Dict]) -> List[Dict]:
    return normalize_and_clean_speaker_title(
        normalize_and_clean_party_affiliations(
            normalize_and_clean_state_info(normalize_and_clean_counts(normalize_labels(datapoints)))
        )
    )
