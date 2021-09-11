from dataclasses import dataclass, field
from typing import Callable, Optional, List, Tuple, NamedTuple, Dict
from enum import Enum
from bits.seq import ExplicitRepr
from bits.util import RelCounter


### ------------------------------ ###
#    Foundamental enums and types
### ------------------------------ ###

class Ctype(Enum):
    """Sequence context type."""
    HP = 0
    DS = 1
    TS = 2


class Etype(Enum):
    """Error type."""
    SELF = 0
    OTHERS = 1


class Wtype(Enum):
    """Wall type."""
    DROP = 0
    GAIN = 1


class State(Enum):
    ERROR = 0
    HAPLO = 1
    DIPLO = 2
    REPEAT = 3


STATES = 'EHDR'

StateT = str
CtxLenT = int
CinT, CoutT = int, int


### ------------------------------ ###
#    Error model parameters
### ------------------------------ ###

@dataclass
class ErrorModel:
    """Class for a sequence context-dependent error rate model.

    instance variables:
      @ max_ulen  : Maximum # of units (i.e. # of HPs, # of DS units, etc.) considered.
      @ fit_model : Function (unit length) |-> (error probability).

    instance methods:
      @ pe(n) : Error rate given the number of unit `n`.
    """
    max_ulen:  CtxLenT
    fit_model: Callable[[CtxLenT], float]
    _pe:       List[float] = field(init=False)

    def __post_init__(self) -> None:
        self._pe = [0] + [round(self.fit_model(n), 3)
                          for n in range(1, self.max_ulen + 1)]

    def pe(self, n: CtxLenT) -> float:
        return self._pe[min(n, self.max_ulen)]


### ------------------------------ ###
#    Perror and Count thresholds
### ------------------------------ ###

class ThresT(Enum):
    """Threshold type for Pr{error} and c_in.
    It holds: "cthres[ThresT][Etype] = min c_in s.t. Pr{error|c_} < pthres[ThresT][Etype]"
    `INIT`  = Loose pthres, meaning "if c_in >= cthres, then it is *never* due to error in `Etype`"
    `FINAL` = Tight pthres, meaning "if c_in >= cthres, then it is *likely* due to error in `Etype`"
    """
    INIT = 0
    FINAL = 1


PerrorThres = Dict[ThresT, Dict[Etype, float]]
CountThres = Dict[ThresT, Dict[Tuple[Ctype, CtxLenT, CoutT, Etype], CinT]]


### ------------------------------ ###
#    Sequence context
### ------------------------------ ###

""" `ctx[i][ctype][DROP]` = #units of HP/DS/TS up to i-1 causing count *drop* at (i-1) -> i.
    `ctx[i][ctype][GAIN]` = #units of HP/DS/TS from (i-K+1) causing count *gain* at (i-1) -> i.
"""
SeqCtx = List[Tuple[len(Ctype) * (Tuple[List[int], List[int]],)]]


### ------------------------------ ###
#    Interval
### ------------------------------ ###

class PerrorInO(NamedTuple):
    b: float
    e: float

class ErrorIntvl(NamedTuple):
    """Simple, immutable interval object for wall pairing."""
    b:  int
    e:  int
    pe: float


@dataclass
class Intvl:
    """Full, mutable interval object for classification targets."""
    b:      int
    e:      int
    pe:     float = 0.   # TODO: logpe? E in O?
    pe_o:   PerrorInO = PerrorInO(b=0., e=0.)
    asgn:   str = '-'
    cb:     Optional[int] = None
    ce:     Optional[int] = None
    ccb:    Optional[int] = None
    cce:    Optional[int] = None
    is_rel: bool = False


### ------------------------------ ###
#    Profiled read
### ------------------------------ ###

@dataclass
class ProfiledRead(ExplicitRepr):
    """Read with a count profile.

    positional arguments:
      @ _seq   : Original, complete nucleotide sequence of the read.
      @ seq    : == `_seq[K - 1:]`. The length is same as the count profile.
      @ id     : Read ID (1-indexed).
      @ name   : Read Name.
      @ K      : Of k-mers.
      @ counts : Count profile. `length(seq) == length(counts) == length(states) == self.length`

    optional arguments:
      @ states : Label (E/H/D/R) for each k-mer.

    instance variables:
      @ length : Length of the count profile (i.e. shorter than original read length).
    """
    _seq:   str
    id:     int
    K:      int
    counts: List[int]
    states: Optional[List[str]] = None
    name:   Optional[str] = None

    @property
    def seq(self) -> str:
        return self._seq[self.K - 1:]

    @property
    def length(self) -> int:
        return len(self.seq)

    @property
    def _length(self) -> int:
        return len(self._seq)

    def __post_init__(self):
        assert self.length == len(self.counts)
        assert self.states is None or self.length == len(self.states)

    def __repr__(self) -> str:
        return self._order_repr(["id", "name", "K", "seq", "counts", "states"])

    def count_freqs(self,
                    max_count: Optional[int] = None) -> RelCounter:
        """Convert count profile to aggregated counts.

        optional arguments:
          @ max_count : Maximum k-mer count.
                        Larger counts are capped to this value.
        """
        return RelCounter(self.counts if max_count is None
                          else [min(count, max_count) for count in self.counts])
