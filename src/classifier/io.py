from os.path import splitext
from typing import Optional
from bits.seq import load_db, load_fasta, load_fastq
import fastk
from ..type import ProfiledRead, ErrorModel
from .context import calc_seq_ctx


def get_pread(read_id: int,
              fastk_prefix: str,
              seq_fname: str,
              hp_emodel: ErrorModel,
              ds_emodel: ErrorModel,
              ts_emodel: ErrorModel) -> ProfiledRead:
    """Load a read, calculate sequence contexts, and trim the first (K-1) bases."""
    pread = load_pread(read_id, fastk_prefix, seq_fname)
    pread.ctx = calc_seq_ctx(pread.seq, pread.K, hp_emodel, ds_emodel, ts_emodel)
    pread.counts = pread.counts[pread.K - 1:]
    pread.seq = pread.seq[pread.K - 1:]
    return pread


def load_pread(read_id: int,
               fastk_prefix: str,
               seq_fname: Optional[str] = None) -> ProfiledRead:
    """Load a single count profile and optionally its sequence.
    Note that the profile is always zero-padded.

    positional arguments:
      @ read_id      : Read ID (1, 2, 3, ...).
      @ fastk_prefix : Prefix of .prof file.
      @ seq_fname    : Sequence file name. If not specified, 'N's are set.
    """
    counts, K = fastk.profex(fastk_prefix, read_id,
                             zero_padding=True, return_k=True)
    if seq_fname is not None:
        ext = splitext(seq_fname)[1]
        assert ext in (".db", ".dam", ".fasta", ".fastq"), \
            f"{seq_fname}: Not supported file type (db/dam/fast[a|q] are supported)"
        seq = (load_fasta if ext == ".fasta"
               else load_fastq if ext == ".fastq"
               else load_db)(seq_fname, read_id)[0].seq
    else:
        seq = 'N' * len(counts)
    assert len(seq) == len(counts)
    return ProfiledRead(id=read_id,
                            K=K,
                            seq=seq,
                            counts=counts)
