from typing import Sequence, Tuple, List, Dict
import numpy as np
from scipy.stats import skellam


def viterbi_count_const(data: Sequence[int],
                        depth_prior: Dict[str, int] = {'E': 1, 'H': 20, 'D': 40},
                        verbose: bool = False) -> str:
    """Viterbi algorithm on the HMM over k-mer counts, consisting of:
    - Data: counts
    - States: {E, H, D}
    - Transition: Const(States -> States)
    - Emission: Poisson(State -> Data) with constant depth per state
    """
    def logp_emission(state: str, d: int) -> float:
        nonlocal STATE_TO_DEPTH
        assert d >= 0, "Invalid parameter"
        l = STATE_TO_DEPTH[state]
        return d * np.log(l) - l - sum([np.log(n) for n in range(1, d + 1)])

    states = ['E', 'H', 'D']
    STATE_TO_DEPTH = depth_prior
    logp_transition = {('E', 'E'): 0.99, ('E', 'H'): 0.005, ('E', 'D'): 0.005,
                       ('H', 'E'): 0.001, ('H', 'H'): 0.99, ('H', 'D'): 0.009,
                       ('D', 'E'): 0.001, ('D', 'H'): 0.009, ('D', 'D'): 0.99}
    N = len(data)
    dp = {(i, s): None for i in range(N) for s in states}
    for s in states:
        dp[(0, s)] = logp_emission(s, data[0])
    backtraces = {(i, s): None for i in range(N) for s in states}
    for i in range(1, N):
        for t in states:
            candidates = [dp[(i - 1, s)] + logp_transition[(s, t)]
                          for s in states]
            backtraces[(i, t)] = states[np.argmax(candidates)]
            dp[(i, t)] = logp_emission(t, data[i]) + max(candidates)
        if verbose:
            print(f"@{i} data = {data[i]}, "
                  f"opt state = {states[np.argmax([dp[(i, s)] for s in states])]}")
    opt_states = [states[np.argmax([dp[(N - 1, s)] for s in states])]]
    for i in reversed(range(1, N)):
        opt_states.append(backtraces[(i, opt_states[-1])])
    return ''.join(reversed(opt_states))


def viterbi_count_variable_transition(data: Sequence[int],
                                      depth_prior: Dict[str, int] = {'E': 1, 'H': 20, 'D': 40},
                                      verbose: bool = False) -> str:
    """Viterbi algorithm on the HMM (?) over k-mer counts, consisting of:
    - Data: counts
    - States: {E, H, D}
    - Transition: Skellam(Data[i] - Data[i+1]; State[i], State[i+1])
      with constant depth per state
    - Emission: Poisson(State -> Data) with constant depth per state
    """
    def logp_emission(state: str, d: int) -> float:
        nonlocal STATE_TO_DEPTH
        assert d >= 0, "Invalid parameter"
        l = STATE_TO_DEPTH[state]
        return d * np.log(l) - l - sum([np.log(n) for n in range(1, d + 1)])

    def logp_transition(s: str, t: str, d_s: int, d_t: int) -> float:
        nonlocal STATE_TO_DEPTH
        assert d_s >= 0 and d_t >= 0, "Invalid parameters"
        return skellam.logpmf(d_s - d_t, STATE_TO_DEPTH[s], STATE_TO_DEPTH[t])

    states = ['E', 'H', 'D']
    STATE_TO_DEPTH = depth_prior
    N = len(data)
    dp = {(i, s): None for i in range(N) for s in states}
    for s in states:
        dp[(0, s)] = logp_emission(s, data[0])
    backtraces = {(i, s): None for i in range(N) for s in states}
    for i in range(1, N):
        for t in states:
            candidates = [dp[(i - 1, s)] + logp_transition(s, t, data[i - 1], data[i])
                          for s in states]
            backtraces[(i, t)] = states[np.argmax(candidates)]
            dp[(i, t)] = logp_emission(t, data[i]) + max(candidates)
        if verbose:
            print(f"@{i} data = {data[i]}, "
                  f"opt state = {states[np.argmax([dp[(i, s)] for s in states])]}")
    opt_states = [states[np.argmax([dp[(N - 1, s)] for s in states])]]
    for i in reversed(range(1, N)):
        opt_states.append(backtraces[(i, opt_states[-1])])
    return ''.join(reversed(opt_states))
