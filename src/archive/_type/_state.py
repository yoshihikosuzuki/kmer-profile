from enum import Enum


class State(Enum):
    ERROR  = 0
    HAPLO  = 1
    DIPLO  = 2
    REPEAT = 3


STATES = 'EHDR'
