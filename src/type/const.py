STATES = 'EHDR'


class StateToColDict(dict):
    def __missing__(self, key):
        return 'gray'


STATE_TO_COL = StateToColDict(E='red', H='#00cfff', D='#0000c0', R='gold')
for s in STATES:
    STATE_TO_COL[s.lower()] = STATE_TO_COL[s]


class StateToColPrintDict(dict):
    def __missing__(self, key):
        return key


STATE_TO_COL_PRINT = StateToColPrintDict(E='\x1b[35mE\x1b[0m', e='\x1b[35me\x1b[0m',
                                         H='\x1b[32mH\x1b[0m', h='\x1b[32mh\x1b[0m',
                                         D='\x1b[34mD\x1b[0m', d='\x1b[34md\x1b[0m',
                                         R='\x1b[33mR\x1b[0m', r='\x1b[33mr\x1b[0m')
