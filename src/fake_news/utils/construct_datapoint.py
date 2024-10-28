from fake_news.utils.datapoint_constructor import Datapoint


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
