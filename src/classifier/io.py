from os.path import splitext
from typing import Union, Optional
from bits.seq import FastaRecord, FastqRecord, DazzRecord, load_db, load_fasta, load_fastq
import fastk
from ..type import ProfiledRead, ErrorModel
from .context import calc_seq_ctx


def get_pread(read_id: int,
              fastk_prefix: str,
              seq_fname: str,
              hp_emodel: ErrorModel,
              ds_emodel: ErrorModel,
              ts_emodel: ErrorModel) -> ProfiledRead:
    """Load a read and calculate feature lengths of sequence contexts."""
    pread = load_pread(read_id, fastk_prefix, seq_fname)
    pread.ctx = calc_seq_ctx(pread._seq, pread.K, hp_emodel, ds_emodel, ts_emodel)
    return pread


Read = Union[FastaRecord, FastqRecord, DazzRecord]


def load_read(seq_fname: str,
              read_id: int,
              case: str = "original") -> Read:
    ext = splitext(seq_fname)[1]
    assert ext in (".db", ".dam", ".fasta", ".fastq"), "Unsupported file type"
    load_func = (load_fasta if ext == ".fasta"
                 else load_fastq if ext == ".fastq"
                 else load_db)
    return load_func(seq_fname, read_id, case)[0]


def load_pread(read_id: int,
               fastk_prefix: str,
               seq_fname: Optional[str] = None,
               case: str = "original") -> ProfiledRead:
    """Load a single count profile and optionally its sequence.
    Note that the profile is always zero-padded.

    positional arguments:
      @ read_id      : Read ID (1, 2, 3, ...).
      @ fastk_prefix : Prefix of .prof file.
      @ seq_fname    : Sequence file name. If not specified, 'N's are set.
    """
    counts, K = fastk.profex(fastk_prefix,
                             read_id,
                             zero_padding=False,
                             return_k=True)
    L = len(counts) + K - 1
    if seq_fname is not None:
        read = load_read(seq_fname, read_id, case)
        seq, name = read.seq, read.name
        assert read.length < K or read.length == L, \
            "Profile length + K - 1 != Read length"
    else:
        seq, name = 'N' * L, None
    return ProfiledRead(_seq=seq,
                        id=read_id,
                        name=name,
                        K=K,
                        counts=counts)
