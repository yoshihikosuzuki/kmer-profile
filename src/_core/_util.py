from typing import Tuple
from math import sqrt
from scipy.stats import binom_test, skellam
from .. import Etype


def plus_sigma(cnt: int,
               sigma: int) -> float:
    """Compute `cnt` + n-sigma (= `sigma`) of Poisson(`cnt`).
    """
    return cnt + sigma * sqrt(cnt)


def minus_sigma(cnt: int,
                sigma: int) -> float:
    """Compute `cnt` - n-sigma (= `sigma`) of Poisson(`cnt`).
    """
    return plus_sigma(cnt, -sigma)


def linear_interpolation(x: int,
                         p1: Tuple[int, int],
                         p2: Tuple[int, int]) -> int:
    # TODO: use PosCnt
    x1, y1 = p1
    x2, y2 = p2
    assert x1 < x and x < x2
    return y1 + (y2 - y1) * ((x - x1) / (x2 - x1))


def calc_p_error(cout: int,
                 cin: int,
                 erate: float,
                 etype: Etype) -> float:
    """Pr{ Count cout -> cin is due to error in `etype` | erate }
    """
    return binom_test(cin if etype == Etype.SELF else cout - cin,
                      cout,
                      erate,
                      alternative="greater")


def _p_trans(_type: str,
             b: int,
             e: int,
             cb: int,
             ce: int,
             cov: int,
             lread: int,
             verbose: bool) -> float:
    assert all([isinstance(c, int) for c in (cb, ce)]), "Counts must be integers"
    sf_lambda = cov * abs(e - b) / lread
    p_or_logp = (skellam.pmf if _type == "p"
                 else skellam.logpmf)(abs(ce - cb), sf_lambda, sf_lambda)
    if verbose:
        print(f"{cb} @ {b} -> {ce} @ {e} (cov={cov}) = {p_or_logp}")
    return p_or_logp


def calc_p_trans(b: int,
                 e: int,
                 cb: int,
                 ce: int,
                 cov: int,
                 lread: int,
                 verbose: bool = False) -> float:
    """Pr{ cb @ b -> ce @ e by sampling fluctuation | mean depth = cov }
    """
    return _p_trans("p", b, e, cb, ce, cov, lread, verbose)


def calc_logp_trans(b: int,
                    e: int,
                    cb: int,
                    ce: int,
                    cov: int,
                    lread: int,
                    verbose: bool = False) -> float:
    """Log Pr{ cb @ b -> ce @ e by sampling fluctuation | mean depth = cov }
    """
    return _p_trans("logp", b, e, cb, ce, cov, lread, verbose)
