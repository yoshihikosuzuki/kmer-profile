from typing import Optional, Sequence, Tuple, Dict
from logzero import logger
from bits.util import RelCounter


def naive_classification(data: Sequence[int],
                         thresholds: Tuple[int, int, int]) -> str:
    """
    positional arguments:
      @ data        : Positional counts
      @ threadholds : Of (E/H, H/D, D/R), respectively.
    """
    return ''.join(['E' if x < thresholds[0]
                    else 'H' if x < thresholds[1]
                    else 'D' if x < thresholds[2]
                    else 'R'
                    for x in data])


def find_depths_and_thres(hist: RelCounter) -> Optional[Tuple[Dict[str, int], Dict[str, int]]]:
    exts = []
    for i in range(2, max(hist) - 1):
        if hist[i - 1] > hist[i] and hist[i] < hist[i + 1]:
            exts.append((i, "min"))
        elif hist[i - 1] < hist[i] and hist[i] > hist[i + 1]:
            exts.append((i, "max"))
    if [y for _, y in exts[:4]] != ["min", "max", "min", "max"]:
        logger.error(f"Failed to detect peaks and thresholds: {exts}")
        return
    eh, h, hd, d = [x for x, _ in exts[:4]]
    depths = {'E': 1, 'H': h, 'D': d, 'R': d * 2}
    thres = {'EH': eh, 'HD': hd, 'DR': d * 2}
    logger.info(f"depths = {depths}")
    logger.info(f"thres = {thres}")
    return (depths, thres)
