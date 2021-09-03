from ._type import Etype, ThresT

### ------------------------------ ###
#    Error model parameters
### ------------------------------ ###

_MAX_CLEN = 20

ERR_PARAMS = ((_MAX_CLEN // 1, lambda n: 0.002 * n * n + 0.002),
              (_MAX_CLEN // 2, lambda n: 0.002 * n * n + 0.002),
              (_MAX_CLEN // 3, lambda n: 0.002 * n * n + 0.002))


### ------------------------------ ###
#    Perror thresholds
### ------------------------------ ###

PE_THRES = {ThresT.INIT: {Etype.SELF: 0.001, Etype.OTHERS: 0.05},
            ThresT.FINAL: {Etype.SELF: 1e-5, Etype.OTHERS: 1e-5}}
MAX_N_HC = 5
PTHRES_DIFF_EO = 1e-20
PTHRES_DIFF_REL = 1e-4


### ------------------------------ ###
#    Sequence context names & colors
### ------------------------------ ###

CTX_NAMES = ("HP", "DS", "TS")
ERR_NAMES = ("self", "others")
WALL_NAMES = ("drop", "gain")


### ------------------------------ ###
#    State colors
### ------------------------------ ###

S_COLS = ('red', 'deepskyblue', 'navy', 'gold')
S_PRINT_COLS = ('\x1b[35mE\x1b[0m', '\x1b[32mH\x1b[0m', '\x1b[34mD\x1b[0m', '\x1b[33mR\x1b[0m')
