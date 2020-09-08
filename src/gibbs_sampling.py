from typing import Sequence, Tuple, List, Dict


def logp_state(i: int,
               s: str,
               data: Sequence[int],
               states: str) -> float:
    """Log probability that the state at position i is `s` given states at the other positions.
    """
    N = len(data)

    h_prev = list(filter(lambda j: states[j] == 'H', range(i)))
    h_prev_depth = h_prev[-1] if len(h_prev) > 0 else 20
    h_next = list(filter(lambda j: states[j] == 'H', range(i + 1, N)))
    h_next_depth = h_next[0] if len(h_next) > 0 else 20

    d_prev = list(filter(lambda j: states[j] == 'D', range(i)))
    d_prev_depth = d_prev[-1] if len(d_prev) > 0 else 40
    d_next = list(filter(lambda j: states[j] == 'D', range(i + 1, N)))
    d_next_depth = d_next[0] if len(d_next) > 0 else 40

    e_prev = list(filter(lambda j: states[j] == 'E', range(i)))
    e_prev_depth = e_prev[-1] if len(e_prev) > 0 else 1
    e_next = list(filter(lambda j: states[j] == 'E', range(i + 1, N)))
    e_next_depth = e_next[0] if len(e_next) > 0 else 1

    
