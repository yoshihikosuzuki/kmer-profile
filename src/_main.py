from . import (Ctype, ErrorModel,
               load_pread, calc_seq_ctx)
from ._const import ERR_PARAMS


def classify_read(read_id: int, fastk_prefix: str, db_fname: str):
    # Load the read with count profile
    pread = load_pread(read_id, fastk_prefix, db_fname)

    # Compute sequence contexts
    emodels = [ErrorModel(*ERR_PARAMS[c]) for c in Ctype]
    pread.ctx = calc_seq_ctx(pread._seq, pread.K, emodels)

    # Compute error probabilities and find walls

    # Find reliable intervals and correct wall counts

    # Classify reliable intervals

    # Classify the rest of the k-mers

    return
