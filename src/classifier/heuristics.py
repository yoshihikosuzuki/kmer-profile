from typing import NamedTuple, Optional, Sequence, List, Tuple, Dict
from copy import deepcopy
import numpy as np
from scipy.stats import binom
from .hidden_markov_model import logp_poisson

states = 'EHDR'
mean_depths = {'E': 1, 'H': 20, 'D': 40, 'R': 100}


class PositionalCount(NamedTuple):
    pos: int
    count: int


class CountIntvl(NamedTuple):
    start: PositionalCount
    end: PositionalCount


def run_heuristics(counts: Sequence[int],
                   seq: str,
                   max_n_iter: int = 30,
                   th_cap: Optional[int] = 100,
                   n_boundary: int = 3) -> str:
    def is_significant(i: int) -> bool:
        """Check if the change counts[i-1] -> counts[i] is significant."""
        # Naive way
        return abs(counts[i] - counts[i - 1]) >= 5
        # TODO: imple sequence-context-aware way

    def find_nearest(i: int,
                     state: str,
                     intervals: List[CountIntvl],
                     assignments: List[str]) -> Tuple[int, int]:
        """Indice of nearest interval labeled as `state` before and after 
        `intervals[i]`
        """
        p = i - 1
        while p >= 0 and assignments[p] != state:
            p -= 1
        n = i + 1
        while n < len(intervals) and assignments[n] != state:
            n += 1
        return (p, n)

    def calc_neighbor_depth(i, l, state, intervals, assignments):
        """Return maximum depth among `l` nearest intervals of the state."""
        ps = []
        p = i - 1
        for _ in range(l):
            while p >= 0 and assignments[p] != state:
                p -= 1
            if p >= 0:
                ps.append(max(intervals[p].start.count,
                              intervals[p].end.count))
            p -= 1
        ns = []
        n = i + 1
        for _ in range(l):
            while n < len(intervals) and assignments[n] != state:
                n += 1
            if n < len(intervals):
                ns.append(max(intervals[n].start.count,
                              intervals[n].end.count))
            n += 1
        return (max(ps) if len(ps) > 0 else -1,
                max(ns) if len(ns) > 0 else len(intervals))

    def calc_logp(i: int,
                  state: str,
                  intervals: List[CountIntvl],
                  assignments: List[str]) -> float:
        """Compute probablity that the state of `intvl` is `state` by calculating
        smoothness of `intvl` given adjacent intervals of the same state.
        """
        intvl = intervals[i]
        if state == 'E':
            return (logp_poisson(intvl.start.count,
                                 mean_depths[state])
                    + logp_poisson(intvl.end.count,
                                   mean_depths[state]))
        elif state in ('H', 'D'):
            """
            p_depth, n_depth = calc_neighbor_depth(i, n_boundary, 'D',
                                                   intervals,
                                                   assignments)
            assert p_depth >= 0 or n_depth < len(intervals), \
                "No diploid states"
            if p_depth < 0:
                p_depth = n_depth
            elif n_depth >= len(intervals):
                n_depth = p_depth
            """
            p, n = find_nearest(i, state, intervals, assignments)
            if p < 0 and n >= len(intervals):
                return -np.inf
            prev_count = (intervals[p].end.count if p >= 0
                          else intervals[n].start.count)
            next_count = (intervals[n].start.count if n < len(intervals)
                          else intervals[p].end.count)
            return (binom.logpmf(min(intvl.start.count, prev_count),
                                 max(intvl.start.count, prev_count),
                                 0.92)
                    + binom.logpmf(min(intvl.end.count, next_count),
                                   max(intvl.end.count, next_count),
                                   0.92))
        else:   # 'R'
            p_depth, n_depth = calc_neighbor_depth(i, n_boundary, 'D',
                                                   intervals,
                                                   assignments)
            assert p_depth >= 0 or n_depth < len(intervals), \
                "No diploid states"
            if p_depth < 0:
                p_depth = n_depth
            elif n_depth >= len(intervals):
                n_depth = p_depth
            if (p_depth + mean_depths['H'] / 2 <= intvl.start.count
                    or n_depth + mean_depths['H'] / 2 <= intvl.end.count):
                return np.inf
            else:
                return -np.inf

    def update_state(i: int,
                     intervals: List[CountIntvl],
                     assignments: List[str]) -> str:
        # Update state of `intervals[i]` based on the current `assignments`
        max_logp = -np.inf
        max_state = None
        #print(f"@{i} {'-'.join(map(str, intervals[i][2:]))}")
        for state in states:
            logp = calc_logp(i, state, intervals, assignments)
            if logp > max_logp:
                max_logp = logp
                max_state = state
        # if max_state != assignments[i]:
        #    print(f"State updated @{i}: {assignments[i]} -> {max_state}")
        return max_state

    assert len(counts) == len(seq)
    if th_cap is not None:
        counts = [min(c, th_cap) for c in counts]
    # TODO: first adjust continuous gain/loses by homopolymer?

    # Split count profile into a set of "smooth" intervals
    # by cutting at "significant" count gain/lose
    change_points = [(0, counts[0])]
    for i in range(1, len(counts)):
        if is_significant(i):
            # NOTE: None is for Plotly
            change_points += [(i - 1, counts[i - 1]),   # end of the last interval
                              (None, None),
                              (i, counts[i])]   # start of the next interval
    change_points.append((len(counts) - 1, counts[-1]))
    intervals = []
    for i in range(len(change_points) // 3 + 1):
        s, t = change_points[i * 3:i * 3 + 2]
        intervals.append(CountIntvl(start=PositionalCount(*s),
                                    end=PositionalCount(*t)))
    # Initial assignment to the nearest state
    # TODO: consider probability
    assignments = [states[np.argmin([abs(mean_depths[state]
                                         - ((intvl.start.count
                                             + intvl.end.count) / 2))
                                     for state in states])]
                   for intvl in intervals]
    # Update assignments until convergence
    count = 0
    for _ in range(max_n_iter):
        print(".", end='')
        prev_assignments = deepcopy(assignments)
        # TODO: update only intervals affected by the last update
        for i in range(len(intervals)):
            assignments[i] = update_state(i, intervals, assignments)
        if prev_assignments == assignments:
            break
        count += 1
        # print(''.join(assignments))
    if count == max_n_iter:
        print("Not converged")
    return change_points, intervals, ''.join(assignments)
