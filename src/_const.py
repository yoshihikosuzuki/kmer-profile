### ------------------------------ ###
#    Error model parameters
### ------------------------------ ###

_MAX_CLEN = 20

ERR_PARAMS = ((_MAX_CLEN // 1, lambda n: 0.002 * n * n + 0.002),
              (_MAX_CLEN // 2, lambda n: 0.002 * n * n + 0.002),
              (_MAX_CLEN // 3, lambda n: 0.002 * n * n + 0.002))


### ------------------------------ ###
#    Sequence context names & colors
### ------------------------------ ###

CTX_NAMES = ("HP", "DS", "TS")
CTX_COLS = (("dodgerblue", "coral"), ("teal", "firebrick"), ("olive", "indigo"))


### ------------------------------ ###
#    State colors
### ------------------------------ ###

S_COLS = ('red', 'deepskyblue', 'navy', 'gold')
S_PRINT_COLS = ('\x1b[35mE\x1b[0m', '\x1b[32mH\x1b[0m', '\x1b[34mD\x1b[0m', '\x1b[33mR\x1b[0m')
