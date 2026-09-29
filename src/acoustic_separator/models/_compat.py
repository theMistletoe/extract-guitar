"""Small helpers needed by vendored MSST model code."""
from typing import List


def prefer_target_instrument(config) -> List[str]:
    if getattr(config.training, "target_instrument", None):
        return [config.training.target_instrument]
    return list(config.training.instruments)
