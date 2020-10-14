from typing import Optional, Sequence, List
from collections import defaultdict
import plotly.graph_objects as go
import plotly_light as pl
from ..type import STATES, STATE_TO_COL, ProfiledRead


def gen_trace_profiled_read(read: ProfiledRead,
                            K: int,
                            layout: Optional[go.Layout] = None) -> go.Figure:
    trace_counts = pl.make_scatter(
        x=list(range(read.length)),
        y=read.counts,
        text=[f"pos = {i}<br>count = {c}<br>"
              f"k-mer = {read.seq[i - K + 1:i + 1] if i >= K - 1 else '-'}"
              for i, c in enumerate(read.counts)],
        mode="lines",
        col="black",
        name="Counts",
        show_legend=True)
    trace_bases = pl.make_scatter(
        x=list(range(read.length)),
        y=read.counts,
        text=list(read.seq),
        text_pos="top center",
        mode="text",
        text_size=10,
        name="Bases",
        show_legend=True,
        show_init=False)
    traces = [trace_counts, trace_bases]
    if read.states is not None:
        assert len(read.states) == read.length
        state_pos = defaultdict(list)
        for i, s in enumerate(read.states):
            assert s in STATES, "Invalid state character"
            state_pos[s].append(i)
        traces += [pl.make_scatter(
            x=pos_list,
            y=[read.counts[i] for i in pos_list],
            mode="markers",
            marker_size=4,
            col=STATE_TO_COL[state])
            for state, pos_list in state_pos.items()]
    _layout = pl.make_layout(x_title="Position",
                             y_title="Count")
    if layout is None:
        _layout = pl.merge_layout(_layout, layout, overwrite=True)
    fig = go.Figure(data=traces, layout=_layout)
    max_count = max(read.counts)
    fig.update_layout(updatemenus=[
        dict(type="buttons",
             buttons=[dict(label="All",
                           method="relayout",
                           args=[{"yaxis.range": (0, max_count)}]),
                      dict(label="<100",
                           method="relayout",
                           args=[{"yaxis.range": (0, 100)}])])])
    return fig
