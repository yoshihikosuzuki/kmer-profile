from typing import Optional, Sequence
from collections import defaultdict
import plotly_light as pl


STATE_TO_COL = {'E': "red", 'H': "green", 'D': "blue", 'R': "yellow"}


def show_transition(counts: Sequence[int],
                    estimated_states: Optional[str] = None,
                    max_count: Optional[int] = None,
                    marker_size: int = 4):
    traces = [pl.make_scatter(x=list(range(len(counts))),
                              y=counts,
                              mode="lines",
                              col="black")]
    if estimated_states is not None:
        state_pos = defaultdict(list)
        for i, state in enumerate(estimated_states):
            assert state in STATE_TO_COL, "Invalid state character"
            state_pos[state].append(i)
        traces += [pl.make_scatter(x=pos_list,
                                   y=[counts[i] for i in pos_list],
                                   mode="markers",
                                   marker_size=marker_size,
                                   col=STATE_TO_COL[state])
                   for state, pos_list in state_pos.items()]
    pl.show(traces,
            pl.make_layout(y_range=(0, max_count)))
