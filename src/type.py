from dataclasses import dataclass
from typing import Optional, List, NamedTuple
from collections import Counter
from BITS.seq.io import SeqRecord

# Labels for 4-class annotation
KMER_STATES = "EHDR"


class StateThresholds(NamedTuple):
    """For naive classification."""
    error_haplo: int
    haplo_diplo: int
    diplo_repeat: int


class RelCounter(Counter):
    """Subclass of Counter with a method returning the relative frequencies."""

    def relative(self):
        tot = sum(self.values())
        return {k: v / tot * 100 for k, v in self.items()}


@dataclass
class ProfiledRead(SeqRecord):
    """Read with a count profile.

    positional arguments:
      @ seq    : Bases of the read.
      @ counts : Count profile.
      @ states : Label (E/H/D/R) for each k-mer.
    """
    counts: List[int]
    states: Optional[str] = None

    def __repr__(self) -> str:
        return self._order_repr(["states", "counts", "seq"])

    def count_freqs(self,
                    max_count: Optional[int] = None) -> RelCounter:
        """Convert count profile to frequency losing positional information.

        optional arguments:
          @ max_count : Maximum k-mer count.
                        Larger counts are capped to this value.
        """
        c = RelCounter(self.counts if max_count is None
                       else [min(count, max_count) for count in self.counts])
        # Remove data from the firt k-1 bases
        c.pop(0, None)
        return c
