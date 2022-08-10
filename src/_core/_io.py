from os.path import splitext
from typing import Union, Optional, List, Tuple
from bits.seq import FastaRecord, FastqRecord, DazzRecord, load_db, load_fasta, load_fastq
import fastk
from .. import ProfiledRead


Read = Union[FastaRecord, FastqRecord, DazzRecord]


def load_read(seq_fname: str,
              read_id: Optional[Union[int, Tuple[int, int]]],
              verbose: bool = False) -> Read:
    ext = splitext(seq_fname)[1]
    assert ext in (".db", ".dam", ".fasta", ".fastq", ".class"), "Unsupported file type"
    load_func = (load_fasta if ext == ".fasta"
                 else load_fastq if ext in (".fastq", ".class")
                 else load_db)
    return load_func(seq_fname, read_id, case="upper", verbose=verbose)


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
    counts, K = fastk.profex(fastk_prefix,
                             read_id,
                             zero_padding=False,
                             return_k=True)
    L = len(counts) + K - 1
    if seq_fname is not None:
        read = load_read(seq_fname, read_id)
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


def load_preads(read_id_range: Optional[Tuple[int, int]],
                fastk_prefix: str,
                seq_fname: Optional[str] = None) -> List[ProfiledRead]:
    """Load multiple profiles."""
    _, K = fastk.profex(fastk_prefix, 1, return_k=True)
    reads = load_read(seq_fname, read_id_range, verbose=True)
    b, e = read_id_range
    return [ProfiledRead(_seq=read.seq,
                         id=read_id,
                         name=read.name,
                         K=K,
                         counts=fastk.profex(fastk_prefix,
                                             read_id,
                                             zero_padding=False,
                                             return_k=False))
            for read_id, read in zip(range(b, e + 1), reads)]
