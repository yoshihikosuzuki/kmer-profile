from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class ErrorModel:
    """Class for sequence context-dependent error rates.

    instance variables:
      @ maxlen : Maximum length of the sequence feature to be considered.
      @ _pe    : Error rate model given a feature length.
                 Practically `self.pe()` is used.
      @ name   : Like "HP", "DS", "TS".
      @ cols   : For drops and gains in visualization.
    """
    maxlen: int
    _pe: List[float]
    name: str
    cols: Tuple[str, str]

    def pe(self,
           l: int) -> float:
        """Sequencing error rate given a length of some feature.

        positional arguments:
          @ l : Length of the feature (e.g. homopolymer length)
        """
        return self._pe[min(l, self.maxlen)]


@dataclass
class SeqCtx:
    """Sequence context information for count drops/gains.

    instance variables:
      @ lens  : = (llens, rlens)
      @ llens : `llens[i]` = length of HP/DS/TS up to position i
                This influences the sequencing error probability of
                a count drop at position i -> (i+1).
      @ rlens : `rlens[i]` = length of HP/DS/TS from position (i-K+1)
                This influences the sequencing error probability of
                a count gain at position (i-1) -> i.
      @ emodel : For this sequence context feature.
    """
    lens: Tuple[List[int], List[int]]
    emodel: ErrorModel

    def __post_init__(self):
        self.erates = tuple([self.emodel.pe(x) for x in data]
                            for data in self.lens)
