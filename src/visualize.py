from typing import Optional, Sequence, List
from collections import defaultdict
import plotly.graph_objects as go
import plotly_light as pl

STATE_COLS = {'E': "red",
              'H': "green",
              'D': "blue",
              'R': "yellow"}


def gen_traces_profile(counts: Sequence[int],
                       estimated_states: Optional[str] = None,
                       line_col: str = "black",
                       max_count: Optional[int] = None,
                       marker_size: int = 4) -> List[go.Scatter]:
    traces = [pl.make_scatter(x=list(range(len(counts))),
                              y=counts,
                              mode="lines",
                              col=line_col)]
    if estimated_states is not None:
        state_pos = defaultdict(list)
        for i, state in enumerate(estimated_states):
            assert state in STATE_COLS, "Invalid state character"
            state_pos[state].append(i)
        traces += [pl.make_scatter(x=pos_list,
                                   y=[counts[i] for i in pos_list],
                                   mode="markers",
                                   marker_size=marker_size,
                                   col=STATE_COLS[state])
                   for state, pos_list in state_pos.items()]
    return traces


def show_profile(counts: Sequence[int],
                 estimated_states: Optional[str] = None,
                 line_col: str = "black",
                 max_count: Optional[int] = None,
                 marker_size: int = 4):
    pl.show(gen_traces_profile(counts,
                               estimated_states,
                               line_col,
                               max_count,
                               marker_size),
            pl.make_layout(y_range=(0, max_count)))
