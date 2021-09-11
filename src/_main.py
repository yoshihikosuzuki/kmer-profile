from dataclasses import dataclass, field
from typing import Union, Optional, Tuple, List
from multiprocessing.pool import Pool
from math import sqrt
from logzero import logger
from bits.seq import db_to_n_reads, FastqRecord, save_fastq
from bits.util import RelCounter, Scheduler, run_distribute
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

    DR_RATIO:     float = field(init=False)

    def __post_init__(self) -> None:
        # Calculating global H/D-depths
        self.hist = fastk.histex(self.fastk_prefix, max_count=self.max_count)
        self.depths, self.dthres = kp.find_depths_and_thres(self.hist)

        # Sequence context-dependent error rates
        self.emodels = [kp.ErrorModel(*kp.ERR_PARAMS[c.value]) for c in kp.Ctype]
        self.cthres = kp.calc_cthres(self, verbose=False)

        # Constants for classification
        self.DR_RATIO = 1 + kp.N_SIGMA_R * (1 / sqrt(self.depths['D']))


def classify_read(pread:   kp.ProfiledRead,
                  cp:      ClassParams,
                  verbose: bool = False) -> kp.ProfiledRead:
    """Main routine for classification of a single read."""
    # Compute sequence contexts
    pread.ctx = kp.calc_seq_ctx(pread._seq, pread.K)

    # Compute error probabilities and find walls
    kp.find_walls(pread, cp, verbose=verbose)

    # Find reliable intervals and correct wall counts
    kp.find_rel_intvls(pread, cp, verbose=verbose)

    # Classify reliable intervals
    kp.classify_rel(pread, cp, verbose=verbose)

    # Classify the rest of the k-mers
    kp.classify_unrel(pread, cp, verbose=verbose)

    return pread


def classify_read_by_id(read_id: int,
                        cp:      ClassParams,
                        verbose: bool = False) -> kp.ProfiledRead:
    """Utility to start classification with a read ID."""
    pread = kp.load_pread(read_id, cp.fastk_prefix, cp.seq_fname)
    return classify_read(pread, cp, verbose)


RangeT = Tuple[int, int]


def _classify_reads(read_id_range: RangeT,
                    fastk_prefix:  str,
                    seq_fname:     str,
                    max_count:     int,
                    read_len:      int) -> List[FastqRecord]:
    cp = kp.ClassParams(fastk_prefix, seq_fname, max_count, read_len)
    preads = kp.load_preads(read_id_range, fastk_prefix, seq_fname)
    classes = [None] * len(preads)
    for i, pread in enumerate(preads):
        classify_read(pread, cp)
        classes[i] = FastqRecord(seq=pread._seq,
                                 name=pread.name,
                                 qual='N' * (pread.K - 1) + pread.states)
    return classes


def classify_reads(read_id_ranges: Union[RangeT, List[RangeT]],
                   fastk_prefix:   str,
                   seq_fname:      str,
                   max_count:      int = 100,
                   read_len:       int = 20000,
                   n_core:         int = 1) -> List[FastqRecord]:
    if n_core == 1:
        return _classify_reads(read_id_ranges,
                               fastk_prefix,
                               seq_fname,
                               max_count,
                               read_len)
    classes = []
    with Pool(n_core) as pool:
        for ret in pool.starmap(_classify_reads,
                                [(read_id_range,
                                  fastk_prefix,
                                  seq_fname,
                                  max_count,
                                  read_len)
                                 for read_id_range in read_id_ranges]):
            classes += ret
    return classes


def main(fastk_prefix: str,
         db_fname:     str,
         max_count:    int = 100,
         read_len:     int = 20000,
         scheduler:    Scheduler = None,
         n_distribute: int = 1,
         n_core:       int = 1) -> None:
    """Entry point for classification of all reads."""
    n_reads = db_to_n_reads(db_fname)
    n_split = n_distribute * n_core
    n_unit = -(-n_reads // n_split)
    read_id_ranges = [(1 + i * n_unit,
                       min([1 + (i + 1) * n_unit - 1, n_reads]))
                      for i in range(-(-n_reads // n_unit))]
    logger.debug(f"(start_dbid, end_dbid)={read_id_ranges}")
    classes = run_distribute(
        func=classify_reads,
        args=read_id_ranges,
        shared_args=dict(fastk_prefix=fastk_prefix,
                         seq_fname=db_fname,
                         max_count=max_count,
                         read_len=read_len),
        scheduler=scheduler,
        n_distribute=n_distribute,
        n_core=n_core,
        max_cpu_hour=72,
        max_mem_gb=100,
        tmp_dname="tmp",
        job_name="ClassPro",
        out_fname="preads.pkl",
        log_level="debug")
    save_fastq(classes, f"{fastk_prefix}.pyclass")
