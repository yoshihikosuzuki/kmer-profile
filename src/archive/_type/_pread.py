from dataclasses import dataclass
from typing import Optional, List
from bits.seq import ExplicitRepr
from bits.util import RelCounter


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
