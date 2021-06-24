from typing import List, Tuple
from scipy.stats import binom_test
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
                      cols=("dodgerblue", "coral"))


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
                      cols=("teal", "firebrick"))

def make_ts_emodel(max_ncopy: int,
                   a: float,
                   b: float) -> ErrorModel:
    """Utility for making a trinucleotide satellite error model.

    positional arguments:
      @ max_ncopy : Maximum copy number of the satellite to be considered
      @ a, b      : Error rate is computed by ax^2 + b, where x is the
                    copy number.
    """
    return ErrorModel(maxlen=max_ncopy,
                      _pe=[0] + [round(a * x * x + b, 3)
                                 for x in range(1, max_ncopy + 1)],
                      name="TS",
                      cols=("olive", "indigo"))


def calc_seq_ctx(seq: str,
                 K: int,
                 hp_emodel: ErrorModel,
                 ds_emodel: ErrorModel,
                 ts_emodel: ErrorModel) -> List[SeqCtx]:
    """Calculate the length of HP, DS, TS (Homopolymer, dinucleotide
    satellite, trinucleotide satellite, respectively.) for each position.

    positional arguments:
      @ seq  : *Original* (= the first (K - 1) bases are untrimmed)
               sequence of a read.
      @ K    : Of K-mers.
      @ [hp|ds|ts]_emodel : Error models.
    """

    def _calc_ctx(seq: str) -> Tuple[List[int], List[int], List[int]]:
        hp_lens = [1] * len(seq)
        ds_lens = [0] * len(seq)
        ts_lens = [0] * len(seq)
        for i in range(len(seq)):
            if i >= 1:
                if seq[i - 1] == seq[i]:
                    hp_lens[i] = hp_lens[i - 1] + 1
                else:
                    ds_lens[i] = 1
                    if i >= 3:
                        if seq[i - 3:i - 1] == seq[i - 1:i + 1]:
                            ds_lens[i] = ds_lens[i - 2] + 1
            if i >= 2:
                if seq[i - 2] == seq[i - 1] == seq[i]:
                    continue
                ts_lens[i] = 1
                if i >= 5:
                    if seq[i - 5:i - 2] == seq[i - 2:i + 1]:
                        ts_lens[i] = ts_lens[i - 3] + 1
        return (hp_lens, ds_lens, ts_lens)

    hp_l, ds_l, ts_l = _calc_ctx(seq)
    hp_r, ds_r, ts_r = _calc_ctx(seq[::-1])
    return (SeqCtx(lens=(hp_l[K - 1:],
                         list(reversed(hp_r))[:-(K - 1)]),
                   emodel=hp_emodel),
            SeqCtx(lens=(ds_l[K - 1:],
                         list(reversed(ds_r))[:-(K - 1)]),
                   emodel=ds_emodel),
            SeqCtx(lens=(ts_l[K - 1:],
                         list(reversed(ts_r))[:-(K - 1)]),
                   emodel=ts_emodel))


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
