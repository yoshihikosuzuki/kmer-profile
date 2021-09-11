from .. import STATES
from .._const import S_COLS, S_PRINT_COLS


class StateToColDict(dict):
    def __missing__(self, key):
        return 'gray'


class StateToColPrintDict(dict):
    def __missing__(self, key):
        return key


S_TO_COL = StateToColDict(dict(zip(STATES + STATES.lower(), S_COLS * 2)))
S_TO_COL_PRINT = StateToColPrintDict(dict(zip(STATES + STATES.lower(), S_PRINT_COLS * 2)))


def color_asgn(asgn):
    return ''.join([S_TO_COL_PRINT[s] for s in asgn])
