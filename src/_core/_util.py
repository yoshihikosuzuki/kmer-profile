from math import sqrt
from scipy.stats import binom_test, skellam
from .._type import Etype


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


def calc_p_trans(b: int,
                 e: int,
                 cb: int,
                 ce: int,
                 cov: int,
                 lread: int = 20000,
                 verbose: bool = False) -> float:
    """Pr{ cb @ b -> ce @ e by sampling fluctuation | mean depth = cov }
    """
    sf_lambda = cov * (e - b) / lread
    p = skellam.pmf(int(ce-cb), sf_lambda, sf_lambda)
    if verbose:
        print(f"{cb} @ {b} -> {ce} @ {e} (cov={cov}) = {p}")
    return p
