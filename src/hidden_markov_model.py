from typing import Sequence, Tuple, List, Dict
import numpy as np
from scipy.stats import skellam


def logp_poisson(d: int, l: float) -> float:
    assert min(d, l) >= 0, "Invalid parameters"
    return d * np.log(l) - l - sum([np.log(n) for n in range(1, d + 1)])


def logp_skellam(d1: int, d2: int, l1: float, l2: float) -> float:
    assert min(d1, d2, l1, l2) >= 0, "Invalid parameters"
    return skellam.logpmf(d1 - d2, l1, l2)


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
        return logp_poisson(d, depth_prior[state])

    states = ['E', 'H', 'D']
    logp_transition = {k: np.log(v) for k, v in
                       {('E', 'E'): 0.99, ('E', 'H'): 0.005, ('E', 'D'): 0.005,
                        ('H', 'E'): 0.001, ('H', 'H'): 0.99, ('H', 'D'): 0.009,
                        ('D', 'E'): 0.001, ('D', 'H'): 0.009, ('D', 'D'): 0.99}.items()}
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
        return logp_poisson(d, depth_prior[state])

    def logp_transition(s: str, t: str, d_s: int, d_t: int) -> float:
        return logp_skellam(d_s, d_t, depth_prior[s] / 24000, depth_prior[t] / 24000)

    states = ['E', 'H', 'D']
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


def viterbi_diff(data: Sequence[int],
                 depth_prior: Dict[str, int] = {'E': 1, 'H': 20, 'D': 40},
                 verbose: bool = False) -> str:
    """Viterbi algorithm on the HMM (?) over k-mer counts, consisting of:
    - Data: counts and count diffs
    - States: {BaseState * BaseState} where BaseState = {E, H, D}
    - Transition: Const(States -> States) where -inf for impossible transitions,
      with constant depth per state
    - Emission: Poisson(State.target -> Data[i]) * Skellam(Data[i - 1] - Data[i], State))
      with constant depth per state
    """
    def logp_emission(state: Tuple[str, str], d_s: int, d_t: int) -> float:
        s, t = state
        l_s, l_t = depth_prior[s], depth_prior[t]
        return logp_skellam(d_s, d_t, l_s / 24000, l_t / 24000) + logp_poisson(d_t, l_t)

    N = len(data)
    base_states = ['E', 'H', 'D']
    states = [(s, t) for s in base_states for t in base_states]
    logp_transition = {(s, t): (-np.inf if s[1] != t[0]
                                else np.log(0.99) if s[0] == s[1] and s[1] == t[1]
                                else np.log(0.01) if s[0] == s[1] and s[1] != t[1]
                                else np.log(0.01) if s[0] == t[1]
                                else np.log(0.01) if s[0] != t[1]
                                else np.log(0.01))
                       for s in states for t in states}
    dp = {(i, s): None for i in range(N) for s in states}
    for s in states:
        dp[(0, s)] = (logp_poisson(data[0], depth_prior[s[1]]) if s[0] == s[1]
                      else -np.inf)
    backtraces = {(i, s): None for i in range(N) for s in states}
    for i in range(1, N):
        for t in states:
            candidates = [dp[(i - 1, s)] + logp_transition[(s, t)]
                          for s in states]
            backtraces[(i, t)] = states[np.argmax(candidates)]
            dp[(i, t)] = logp_emission(t, data[i - 1], data[i]) + max(candidates)
        if verbose:
            print(f"@{i} data = {data[i - 1]} -> {data[i]}, "
                  f"opt state = {states[np.argmax([dp[(i, s)] for s in states])]}")
    opt_states = [states[np.argmax([dp[(N - 1, s)] for s in states])]]
    for i in reversed(range(1, N)):
        opt_states.append(backtraces[(i, opt_states[-1])])
    return ''.join([x[1] for x in reversed(opt_states)])
