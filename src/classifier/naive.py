from typing import Sequence, List, Tuple
from ..type import StateThresholds


def naive_classification(data: Sequence[int],
                         thresholds: StateThresholds) -> str:
    return ''.join(['E' if x < thresholds[0]
                    else 'H' if x < thresholds[1]
                    else 'D' if x < thresholds[2]
                    else 'R'
                    for x in data])
