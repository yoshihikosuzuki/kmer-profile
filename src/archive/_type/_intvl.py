from dataclasses import dataclass
from typing import Tuple
from ._error import Etype

@dataclass
class Intvl:
    b: int
    e: int
    cb: int
    ce: int
    pe: Tuple[len(Etype) * (float,)]
    asgn: str
    is_rel: bool
    is_err: bool


@dataclass
class RelIntvl(Intvl):
    ccb: int
    cce: int
    # TODO: prob of error in O after correction?
