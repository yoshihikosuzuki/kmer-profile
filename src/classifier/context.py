from dataclasses import dataclass
from typing import Sequence, List, Tuple, Dict
from copy import deepcopy
import numpy as np
from scipy.stats import binom, binom_test, poisson, skellam, norm
from scipy.special import digamma
from logzero import logger
from ..type import ProfiledRead, SeqCtx, ErrorModel


def make_hp_emodel(max_hplen: int,
                   a: float,
                   b: float) -> ErrorModel:
    """Utility for making a homopolymer error model.

    positional arguments:
      @ max_hplen : Maximum homopolymer length to be considered
      @ a, b      : Error rate is computed by ax^2 + b, where x is the
                    homopolymer length.
    """
    return ErrorModel(maxlen=max_hplen,
                      _pe=[0] + [round(a * x * x + b, 3)
                                 for x in range(1, max_hplen + 1)],
                      name="HP",
                      cols=("#0041ff", "#ff7700"))


def make_ds_emodel(max_ncopy: int,
                   a: float,
                   b: float) -> ErrorModel:
    """Utility for making a dinucleotide satellite error model.

    positional arguments:
      @ max_ncopy : Maximum copy number of the satellite to be considered
      @ a, b      : Error rate is computed by ax^2 + b, where x is the
                    copy number.
    """
    return ErrorModel(maxlen=max_ncopy,
                      _pe=[0] + [round(a * x * x + b, 3)
                                 for x in range(1, max_ncopy + 1)],
                      name="DS",
                      cols=("#35a16b", "#910079"))


def calc_seq_ctx(seq: str,
                 K: int,
                 hp_emodel: ErrorModel,
                 ds_emodel: ErrorModel,
                 ts_emodel: ErrorModel = None) -> List[SeqCtx]:
    """Calculate the length of HP, DS, TS (Homopolymer, dinucleotide
    satellite, trinucleotide satellite, respectively.) for each position.

    positional arguments:
      @ seq  : *Original* (= the first (K - 1) bases are untrimmed)
               sequence of a read.
      @ type : Must be one of {"HP", "DS", "TS"}.
               (Homopolymer, dinucleotide satellite, trinucleotide satellite,
                respectively.)
      @ K    : Of K-mers.
    """
    def _calc_hp_lens(seq: str) -> List[int]:
        lens = []
        x = seq[0]
        c = 1
        lens.append(c)
        for i in range(1, len(seq)):
            if seq[i] == x:
                c += 1
            else:
                c = 1
            lens.append(c)
            x = seq[i]
        assert len(seq) == len(lens)
        return lens

    def _calc_disat_lens(seq: str) -> List[int]:
        # TODO: do simultaneously with HP
        lens = []
        x = seq[0]
        c = 0
        lens.append(c)
        y = seq[1]
        c = 0 if x == y else 1
        lens.append(c)
        z = seq[2]
        c = 0 if y == z else 1
        lens.append(c)
        for i in range(3, len(seq)):
            if seq[i] == z:
                c = 0
            elif x == z and y == seq[i]:
                c += 1
            elif seq[i] != y:
                c = 1
            lens.append(c if c <= 2 else 2 + (c - 2) // 2)
            x = y
            y = z
            z = seq[i]
        assert len(seq) == len(lens)
        return lens

    def _calc_trisat_lens(seq: str) -> List[int]:
        # TODO: implement
        pass

    rseq = seq[::-1]
    return (SeqCtx(lens=(_calc_hp_lens(seq)[K - 1:],
                         list(reversed(_calc_hp_lens(rseq)))[:-(K - 1)]),
                   emodel=hp_emodel),
            SeqCtx(lens=(_calc_disat_lens(seq)[K - 1:],
                         list(reversed(_calc_disat_lens(rseq)))[:-(K - 1)]),
                   emodel=ds_emodel))


def calc_p_errors(pread: ProfiledRead) -> None:
    pread.pe = {error_type: {change_type: [max(p_list)
                                           for p_list in zip(*[[calc_p_error(i, pread, ctx, error_type, change_type)
                                                                for i in range(pread.length)]
                                                               for ctx in pread.ctx])]
                             for change_type in ("drop", "gain")}
                for error_type in ("self", "others")}


def calc_p_error(i: int,
                 pread: ProfiledRead,
                 ctx: SeqCtx,
                 error_type: str,
                 change_type: str) -> float:
    """Compute the probability that the count [drop|gain] at (i-1) -> i
    is because of error(s) in [self|others].

    positional argument:
      @ i     : Position index.
      @ pread : Profiled read.
      @ ctx   : Sequence context feature vectors. Element in `pread.ctx`.
      @ error_type  : Must be one of {"self", "others"}.
      @ change_type : Must be one of {"drop", "gain"}.
    """
    # TODO: Is it OK to take max among context types within this function?
    assert change_type in ("drop", "gain")
    assert error_type in ("self", "others")
    # NOTE: If the count data is invalid, set Pr=0 for error in self
    #       and Pr=1 for errors in others
    p_invalid = 0. if error_type == "self" else 1.
    if not (0 < i and i < pread.length):
        return p_invalid
    cp, ci = pread.counts[i - 1:i + 1]
    if change_type == "drop":
        if cp <= ci:   # not a drop
            return p_invalid
        return binom_test(ci if error_type == "self" else cp - ci,
                          cp,
                          ctx.erates[0][i - 1],
                          alternative="greater")
    else:
        if cp >= ci:   # not a gain
            return p_invalid
        return binom_test(cp if error_type == "self" else ci - cp,
                          ci,
                          ctx.erates[1][i],
                          alternative="greater")


def recalc_p_errors(pread):
    pread.corrected_pe = {error_type: {change_type: [max(p_list)
                                           for p_list in zip(*[[recalc_p_error(i, pread, ctx, error_type, change_type)
                                                                for i in range(pread.length)]
                                                               for ctx in pread.ctx])]
                             for change_type in ("drop", "gain")}
                for error_type in ("self", "others")}


def recalc_p_error(i, pread, ctx, error_type, change_type) -> float:
    """Compute the probability that the count [drop|gain] at (i-1) -> i
    is because of error(s) in [self|others].

    positional argument:
      @ i     : Position index.
      @ pread : Profiled read.
      @ ctx   : Sequence context feature vectors. Element in `pread.ctx`.
      @ error_type  : Must be one of {"self", "others"}.
      @ change_type : Must be one of {"drop", "gain"}.
    """
    # TODO: Is it OK to take max among context types within this function?
    assert change_type in ("drop", "gain")
    assert error_type in ("self", "others")
    # NOTE: If the count data is invalid, set Pr=0 for error in self
    #       and Pr=1 for errors in others
    p_invalid = 0. if error_type == "self" else 1.
    if not (0 < i and i < pread.length):
        return p_invalid
    cp, ci = pread.corrected_counts[i - 1:i + 1]
    if change_type == "drop":
        if cp <= ci:   # not a drop
            return p_invalid
        return binom_test(ci if error_type == "self" else cp - ci,
                          cp,
                          ctx.erates[0][i - 1],
                          alternative="greater")
    else:
        if cp >= ci:   # not a gain
            return p_invalid
        return binom_test(cp if error_type == "self" else ci - cp,
                          ci,
                          ctx.erates[1][i],
                          alternative="greater")
