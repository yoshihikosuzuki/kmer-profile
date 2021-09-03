from typing import List, Tuple
from .. import Ctype, SeqCtx


def calc_seq_ctx(seq: str,
                 K: int) -> SeqCtx:
    """Utility for returning a list of `SeqCtx` objects given a sequence.

    positional arguments:
      @ seq     : *Original* (= the first (K - 1) bases are NOT trimmed) sequence of a read.
      @ K       : Of K-mers.

    return value:
      @ ctx : Sequence context per position for each i) sequence type and ii) wall type.
                `ctx[i][ctype][DROP]` = #units of HP/DS/TS up to i-1 causing count drop at (i-1) -> i.
                `ctx[i][ctype][GAIN]` = #units of HP/DS/TS from (i-K+1) causing count gain at (i-1) -> i.
    """
    def _calc_ctx(seq: str) -> Tuple[len(Ctype) * (List[int],)]:
        """Core function that calcualtes sequence feature lengths for HP, DS, and TS given a sequence.
        """
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

    lhp, lds, lts = [[0] + x[K-1:-1] for x in _calc_ctx(seq)]
    rhp, rds, rts = [list(reversed(x))[:-(K - 1):] for x in _calc_ctx(seq[::-1])]
    return list(zip(zip(lhp, rhp), zip(lds, rds), zip(lts, rts)))
