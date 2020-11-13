import os
from typing import Optional
from fastk import profex
from bits.seq import load_db
from ..type import ProfiledRead


def load_pread(db_fname: str,
               fastk_prefix: str,
               read_id: int,
               K: int) -> Optional[ProfiledRead]:
    if not (isinstance(db_fname, str) and os.path.exists(db_fname)):
        return None
    seq = load_db(db_fname, read_id)[0].seq
    counts = profex(f"{fastk_prefix}.K{K}", read_id)
    assert len(counts) == len(seq) - (K - 1), "Inconsistent lengths"
    return ProfiledRead(K=K,
                        id=read_id,
                        seq=seq,
                        counts=[0] * (K - 1) + counts)
