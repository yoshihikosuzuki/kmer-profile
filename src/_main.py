from dataclasses import dataclass, field
from typing import Union, Optional, Tuple, List
import argparse
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
    show_hist:    bool = False
    width:        int = 600
    height:       int = None
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

        if self.show_hist:
            kp.CountHistVisualizer(width=self.width,
                                   height=self.height,
                                   relative=True,
                                   show_legend=False) \
                .add_trace(self.hist, opacity=1) \
                .show()

    def set_custom_depths_and_thres(self, depths, dthres):
        self.depths = depths
        self.dthres = dthres
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="ClassProPy: Python version of ClassPro (only development purpose)")
    parser.add_argument(
        "fastk_prefix",
        type=str,
        help="Prefix of the FASTK output files.")
    parser.add_argument(
        "db_fname",
        type=str,
        help="Input DAZZ_DB file name.")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose mode.")
    parser.add_argument(
        "-r",
        "--read_id",
        type=int,
        default=0,
        help="Run for single read of this ID. [0]")
    parser.add_argument(
        "-c",
        "--max_count",
        type=int,
        default=70,
        help="Value of k-mer counts to be capped. [70]")
    parser.add_argument(
        "-l",
        "--read_len",
        type=int,
        default=20000,
        help="Average read length. [20000]")
    parser.add_argument(
        "-m",
        "--job_manager",
        type=str,
        default="slurm",
        help="Name of job manager. Must be one of {'slurm', 'sge'}. ['slurm']")
    parser.add_argument(
        "-p",
        "--partition",
        type=str,
        default="compute",
        help="Name of partition to which jobs are submitted using the job manager. ['compute']")
    parser.add_argument(
        "-x",
        "--n_distribute",
        type=int,
        default=25,
        help="Number of jobs to be submitted. []")
    parser.add_argument(
        "-y",
        "--n_core",
        type=int,
        default=64,
        help="Number of cores per job. `n_distribute * n_core` is the total number of cores used []")
    args = parser.parse_args()
    return args


def main() -> None:
    """Entry point for classification of all reads."""
    args = parse_args()
    fastk_prefix = args.fastk_prefix
    db_fname = args.db_fname
    max_count = args.max_count
    read_len = args.read_len

    if args.read_id > 0:
        cp = kp.ClassParams(fastk_prefix, db_fname, max_count, read_len)
        pread = classify_read_by_id(args.read_id, cp, verbose=args.verbose)
        return

    assert args.job_manager in ("slurm", "sge")
    scheduler = Scheduler(args.job_manager,
                          "sbatch" if args.job_manager == "slurm" else "qsub",
                          args.partition,
                          prefix_command="shopt -s expand_aliases; source ~/.bashrc; ml FASTK")
    n_distribute = args.n_distribute
    n_core = args.n_core

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
        max_mem_gb=500,
        tmp_dname="tmp",
        job_name="ClassPro",
        out_fname="preads.pkl",
        log_level="debug")
    save_fastq(classes, f"{fastk_prefix}.pyclass")
