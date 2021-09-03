from typing import Optional, Tuple, Dict
from logzero import logger
from bits.util import RelCounter
from ._util import plus_sigma


def find_depths_and_thres(hist: RelCounter,
                          r_n_sigma: int = 6,
                          dr_n_sigma: int = 4) -> Optional[Tuple[Dict[str, int], Dict[str, int]]]:
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
    r = int(plus_sigma(d, r_n_sigma))
    dr = int(plus_sigma(d, dr_n_sigma))
    depths = {'E': 1, 'H': h, 'D': d, 'R': r}
    thres = {'EH': eh, 'HD': hd, 'DR': dr}
    logger.info(f"depths = {depths}")
    logger.info(f"thres = {thres}")
    return (depths, thres)
