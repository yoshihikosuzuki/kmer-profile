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

# SELF:   Larger == Smaller thresholds
#         If cin > this value, then the position is not considered due to errors in self
# OTHERS: Larger == Greater thresholds
#         If cin < this value, then the position is not considered due to errors in others
# INIT = for individual positions, FINAL = for pairs
PE_THRES = {ThresT.INIT: {Etype.SELF: 0.001, Etype.OTHERS: 0.05},
            ThresT.FINAL: {Etype.SELF: 1e-5, Etype.OTHERS: 1e-5}}
MAX_N_HC = 5             # Number of bases considered within a single error event
PTHRES_DIFF_EO = 1e-10   # in [0, 1]; Larger == Fewer walls explained by errors in others == More walls
PTHRES_DIFF_REL = 1e-4   # in [0, 1]; Larger == Fewer reliable intervals == More smooth reliable intervals


### ------------------------------ ###
#    Perror thresholds
### ------------------------------ ###

OFFSET = 1000             # Larger == Allow larger count difference between close positions in the same class
# PTHRES_TRANS_REL = 1e-3   # For H < D < R requirment
N_SIGMA_R = 2             # R-cov = D-cov + `N_SIGMA_R`-sigma
R_LOGP = -10.             # Pr{R-interval} = this value, to prioritize other classes with a somewhat high probability
E_PO_BASE = -0.          # Add this value to Pr{E-interval} to avoid over-classification in low-cov regions


### ------------------------------ ###
#    Sequence context names & colors
### ------------------------------ ###

STATE_NAMES = ("Error", "Haplo", "Diplo", "Repeat")
CTX_NAMES = ("HP", "DS", "TS")
ERR_NAMES = ("self", "others")
WALL_NAMES = ("drop", "gain")

# |Ctype| * |Wtype|
CTX_COLS = (("dodgerblue", "coral"),
            ("teal", "firebrick"),
            ("olive", "indigo"))


### ------------------------------ ###
#    State colors
### ------------------------------ ###

S_COLS = ('red', 'deepskyblue', 'navy', 'gold')
S_PRINT_COLS = ('\x1b[35mE\x1b[0m', '\x1b[32mH\x1b[0m', '\x1b[34mD\x1b[0m', '\x1b[33mR\x1b[0m')
