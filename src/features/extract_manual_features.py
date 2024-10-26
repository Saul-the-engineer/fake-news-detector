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


class TreeFeaturizer(object):
    def __init__(self, featurizer_cache_path: str, config: Optional[Dict] = None):
        # NOTE: Here you can add feature caching which helps if it's too expensive
        # to compute features from scratch for each run
        if os.path.exists(featurizer_cache_path):
            LOGGER.info("Loading featurizer from cache...")
            with open(featurizer_cache_path, "rb") as f:
                self.combined_featurizer = pickle.load(f)
        else:
            LOGGER.info("Creating featurizer from scratch...")
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            # Load optimal credit bins
            with open(os.path.join(base_dir, config["credit_bins_path"])) as f:
                optimal_credit_bins = json.load(f)
            dict_featurizer = DictVectorizer()
            tfidf_featurizer = TfidfVectorizer()

            statement_transformer = FunctionTransformer(extract_statements)
            manual_feature_transformer = FunctionTransformer(
                partial(extract_manual_features, optimal_credit_bins=optimal_credit_bins)
            )

            manual_feature_pipeline = Pipeline(
                [
                    ("manual_features", manual_feature_transformer),
                    ("manual_featurizer", dict_featurizer),
                ]
            )

            ngram_feature_pipeline = Pipeline(
                [
                    ("statements", statement_transformer),
                    ("ngram_featurizer", tfidf_featurizer),
                ]
            )

            self.combined_featurizer = FeatureUnion(
                [
                    ("manual_feature_pipe", manual_feature_pipeline),
                    ("ngram_feature_pipe", ngram_feature_pipeline),
                ]
            )

    def get_all_feature_names(self) -> List[str]:
        all_feature_names = []
        for name, pipeline in self.combined_featurizer.transformer_list:
            final_pipe_name, final_pipe_transformer = pipeline.steps[-1]
            all_feature_names.extend(final_pipe_transformer.get_feature_names())
        return all_feature_names

    def fit(self, datapoints: List[Datapoint]) -> None:
        self.combined_featurizer.fit(datapoints)

    def featurize(self, datapoints: List[Datapoint]) -> np.array:
        return self.combined_featurizer.transform(datapoints)

    def save(self, featurizer_cache_path: str):
        LOGGER.info("Saving featurizer to disk...")
        with open(featurizer_cache_path, "wb") as f:
            pickle.dump(self.combined_featurizer, f)


def compute_bin_idx(val: float, bins: List[float]) -> int:
    for idx, bin_val in enumerate(bins):
        if val <= bin_val:
            return idx
