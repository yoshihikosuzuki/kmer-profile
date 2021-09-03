from dataclasses import dataclass, field
from typing import Tuple
from bits.util import RelCounter
import fastk
import kmer_profiler as kp


@dataclass
class ClassParams:
    fastk_prefix: str
    seq_fname:    str
    max_count:    int = 100
    read_len:     int = 20000
    hist:         RelCounter = field(init=False)
    depths:       Tuple[len(kp.STATES) * (int,)] = field(init=False)
    dthres:       Tuple[(len(kp.STATES) - 1) * (int,)] = field(init=False)
    emodels:      Tuple[len(kp.Ctype) * (kp.ErrorModel,)] = field(init=False)
    cthres:       kp.CountThres = field(init=False)

    def __post_init__(self) -> None:
        # Calculating global H/D-depths
        self.hist = fastk.histex(self.fastk_prefix, max_count=self.max_count)
        self.depths, self.dthres = kp.find_depths_and_thres(self.hist)

        # Sequence context-dependent error rates
        self.emodels = [kp.ErrorModel(*kp.ERR_PARAMS[c.value]) for c in kp.Ctype]
        self.cthres = kp.calc_cthres(self, verbose=False)


def classify_read(read_id: int,
                  cp: ClassParams,
                  verbose: bool = False):
    # Load the read with count profile
    pread = kp.load_pread(read_id, cp.fastk_prefix, cp.seq_fname)

    # Compute sequence contexts
    pread.ctx = kp.calc_seq_ctx(pread._seq, pread.K)

    # Compute error probabilities and find walls
    kp.find_walls(pread, cp, verbose=verbose)

    # Find reliable intervals and correct wall counts
    kp.find_rel_intvls(pread, cp, verbose=verbose)

    # Classify reliable intervals

    # Classify the rest of the k-mers

    return pread
