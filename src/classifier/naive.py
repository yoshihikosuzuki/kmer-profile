from typing import Sequence, List, Tuple


def naive_classification(data: Sequence[int],
                         thresholds: Tuple[int, int, int]) -> str:
    return ''.join(['E' if x < thresholds[0]
                    else 'H' if x < thresholds[1]
                    else 'D' if x < thresholds[2]
                    else 'R'
                    for x in data])
