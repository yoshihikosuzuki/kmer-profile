from os.path import splitext
from dataclasses import dataclass
from typing import Optional, List, NamedTuple
from bits.seq import SeqRecord, load_db, load_fasta, load_fastq
from bits.util import RelCounter
import fastk


@dataclass
class ProfiledRead(SeqRecord):
    """Read with a count profile.

    positional arguments:
      @ seq    : Bases of the read.
      @ id     : Read ID.
      @ K      : Of k-mers.
      @ counts : Count profile.

    optional arguments:
      @ states : Label (E/H/D/R) for each k-mer.
    """
    id: int
    K: int
    counts: List[int]
    states: Optional[List[str]] = None

    def __post_init__(self):
        assert len(self.seq) == len(self.counts), \
            f"Inconsistent length between seq({len(self.seq)}) and counts({len(self.counts)})"
        if self.states is not None:
            assert len(self.seq) == len(self.states), \
                f"Inconsistent length between seq({len(self.seq)}) and states({len(self.states)})"

    def __repr__(self) -> str:
        return self._order_repr(["K", "id", "counts", "states", "seq"])

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


def load_pread(read_id: int,
               fastk_prefix: str,
               seq_fname: Optional[str] = None,
               K: Optional[int] = None) -> ProfiledRead:
    """Utility for making a single profile read.

    positional arguments:
      @ read_id      : Read ID (1, 2, 3, ...).
      @ fastk_prefix : Prefix of .prof file.
      @ seq_fname    : Sequence file name.
    """
    counts = fastk.profex(fastk_prefix, read_id, K=K)
    if seq_fname is not None:
        ext = splitext(seq_fname)[1]
        assert ext in (".db", ".dam", ".fasta", ".fastq"), \
            "Not supported input file"
        seq = (load_fasta if ext == ".fasta"
               else load_fastq if ext == ".fastq"
               else load_db)(seq_fname, read_id)[0].seq
        if K is None:
            seq = seq[K - 1:]
    else:
        seq = 'N' * len(counts)
    return ProfiledRead(id=read_id,
                        K=K,
                        seq=seq,
                        counts=counts)
