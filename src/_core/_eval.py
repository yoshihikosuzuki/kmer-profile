from typing import List, Optional
from logzero import logger
from bits.seq import FastqRecord
import fastk


def calc_acc(est_class: List[FastqRecord],
             truth_class: List[FastqRecord],
             start_id: int = 0,
             num_reads: int = 100,
             min_pdiff_toshow: Optional[float] = None,
             fastk_prefix: Optional[str] = None,
             min_rep_cnt: int = 100) -> None:
    """Show read-wise accuracy of the classification.

    positional arguments:
      @ est_class   : Classifications by an estimator.
      @ truth_class : Ground-truth classifications.

    optional arguments:
      @ start_id         : Start read ID to be considered.
      @ num_reads        : Reads in [`start_id`..`start_id + num_reads`] are inspected.
      @ min_pdiff_toshow : Show classification info for reads with (100 - accuracy) <= this.
      @ fastk_prefix     : If specified, repeat k-mer rate is also shown.
      @ min_rep_cnt      : K-mers with counts >= this will be treated as repeat.
    """
    tot_len = tot_diff = 0
    for i in range(start_id, min(len(truth_class), start_id + num_reads)):
        t, e = truth_class[i], est_class[i]
        assert t.name == e.name and t.seq == e.seq
        L = t.length

        if fastk_prefix is not None:
            counts = fastk.profex(fastk_prefix, i + 1, zero_padding=False, return_k=False)
            n_rep = sum([c >= min_rep_cnt for c in counts])
            p_rep = n_rep / L * 100

        n_diff = sum([tc != ec for tc, ec in zip(t.qual, e.qual)])
        p_diff = n_diff / L * 100
        tot_len += L
        tot_diff += n_diff

        if min_pdiff_toshow is not None and p_diff >= min_pdiff_toshow:
            rep = f"[{p_rep:5.1f} %repeat]" if fastk_prefix is not None else ""
            print(f"Read {i + 1:6}: {p_diff:5.1f} % ({n_diff:5} / {L:5}) {rep}")

    acc = (1 - tot_diff / tot_len) * 100
    logger.info(f"Accuracy = {acc:5.1f} % ({num_reads} reads from {start_id}..{start_id + num_reads})")
