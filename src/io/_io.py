import os
from typing import Optional, Union, Tuple, List
from bits.seq import load_db
from bits.util import RelCounter, run_command
from ..type import ProfiledRead


def pullback_hoco(hoco_profile: List[int],
                  normal_seq: str) -> List[int]:
    """Project back hoco profile onto normal space.
    """
    assert hoco_profile[0] == 0, "Must have (K-1) 0-counts"
    pb_profile = [None] * len(normal_seq)
    i_normal = i_hoco = 0
    pb_profile[i_normal] = hoco_profile[i_hoco]
    for i_normal in range(1, len(normal_seq)):
        if normal_seq[i_normal] != normal_seq[i_normal - 1]:
            i_hoco += 1
        pb_profile[i_normal] = hoco_profile[i_hoco]
    assert i_hoco == len(hoco_profile) - 1, "Inconsistent lengths"
    return pb_profile


def load_pread(db_fname: str,
               fastk_prefix: str,
               read_id: int,
               K: int,
               fastk_prefix_hoco: Optional[str] = None) \
        -> Optional[Union[ProfiledRead, Tuple[ProfiledRead, ProfiledRead]]]:
    """
    positional arguments:
      @ db_fname     : A DAZZ_DB file name.
      @ fastk_prefix : Prefix of the output files of FastK.
      @ read_id      : 1, 2, 3, ...
      @ K            : For k-mers.

    optional arguments:
      @ fastk_prefix_hoco : Hoco profiles made from the same `db_fname`.
                            If specified, returns both the normal profile and
                            the (pullback'ed) hoco profile.
    """
    if not (isinstance(db_fname, str) and os.path.exists(db_fname)):
        return None
    seq = load_db(db_fname, read_id)[0].seq
    counts = load_profex(fastk_prefix, read_id)
    assert len(counts) == len(seq) - (K - 1), "Inconsistent lengths"
    pread = ProfiledRead(K=K,
                         id=read_id,
                         seq=seq,
                         counts=[0] * (K - 1) + counts)
    if fastk_prefix_hoco is None:
        return pread
    # Load and pullback the hoco profile onto the original sequence
    counts_hoco = [0] * (K - 1) + load_profex(fastk_prefix_hoco, read_id)
    normal_to_hoco = [None] * pread.length
    normal_to_hoco[0] = i_hoco = 0
    for i_normal in range(1, pread.length):
        if pread.seq[i_normal] != pread.seq[i_normal - 1]:
            i_hoco += 1
        normal_to_hoco[i_normal] = i_hoco
    assert i_hoco == len(counts_hoco) - 1, "Inconsistent lengths"
    pread_hoco = ProfiledRead(K=K,
                              id=read_id,
                              seq=seq,
                              counts=[counts_hoco[i_hoco]
                                      for i_hoco in normal_to_hoco])
    return (pread, pread_hoco)
