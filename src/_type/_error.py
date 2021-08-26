from dataclasses import dataclass, field
from typing import Callable, List, Tuple
from enum import Enum


class Ctype(Enum):
    HP = 0
    DS = 1
    TS = 2


class Etype(Enum):
    SELF = 0
    OTHERS = 1


class Wtype(Enum):
    DROP = 0
    GAIN = 1


@dataclass
class ErrorModel:
    """Class for a sequence context-dependent error rate model.

    instance variables:
      @ max_ulen  : Maximum # of units (i.e. # of HPs, # of DS units, etc.) considered.
      @ fit_model : Function (unit length) |-> (error probability).

    instance methods:
      @ pe(n) : Error rate given the number of unit `n`.
    """
    max_ulen:  int
    fit_model: Callable[[int], float]
    _pe:       List[float] = field(init=False)

    def __post_init__(self) -> None:
        self._pe = [0] + [round(self.fit_model(n), 3)
                          for n in range(1, self.max_ulen + 1)]

    def pe(self, n: int) -> float:
        return self._pe[min(n, self.maxlen)]


@dataclass
class SeqCtx:
    """Class for feature length and error rate for count drops/gains, given a sequence context type.

    instance variables:
      @ emodel : Error rate model.
      @ lens   : `lens[DROP][i]` = #units of HP/DS/TS up to i causing count drop at i -> (i+1).
                 `lens[GAIN][i]` = #units of HP/DS/TS from (i-K+1) causing count gain at (i-1) -> i.
      @ erates : Error rates of drop/gain at each position.
    """
    emodel: ErrorModel
    lens:   Tuple[len(Wtype) * (List[int],)]
    erates: Tuple[len(Wtype) * (List[float],)] = field(init=False)

    def __post_init__(self) -> None:
        self.erates = tuple([self.emodel.pe(n) for n in self.lens[w]]
                            for w in Wtype)
