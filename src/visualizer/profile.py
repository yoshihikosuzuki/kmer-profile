from typing import Union, Optional, Sequence, List
from collections import defaultdict
import plotly.graph_objects as go
from plotly.basedatatypes import BaseTraceType
import plotly_light as pl
from ..type import STATES, STATE_TO_COL, ProfiledRead


def gen_trace_pread_counts(pread: ProfiledRead,
                           col: str = "black",
                           name: str = "Counts",
                           show_legend: bool = True) -> go.Scatter:
    return pl.make_scatter(
        x=list(range(pread.length)),
        y=pread.counts,
        text=[f"pos = {i}<br>count = {c}<br>"
              f"k-mer = {pread.seq[i - pread.K + 1:i + 1] if i >= pread.K - 1 else '-'}"
              for i, c in enumerate(pread.counts)],
        mode="lines",
        col=col,
        name=name,
        show_legend=show_legend)


def gen_trace_pread_bases(pread: ProfiledRead,
                          col: str = "black",
                          name: str = "Bases",
                          show_legend: bool = True,
                          show_init: bool = False) -> go.Scatter:
    return pl.make_scatter(x=list(range(pread.length)),
                           y=pread.counts,
                           text=list(pread.seq),
                           text_pos="top center",
                           mode="text",
                           name=name,
                           show_legend=show_legend,
                           show_init=show_init)


def gen_trace_pread_states(pread: ProfiledRead,
                           show_legend: bool = True,
                           show_init: bool = False) -> List[go.Scatter]:
    assert len(pread.states) == pread.length
    state_pos = defaultdict(list)
    for i, s in enumerate(pread.states):
        assert s in STATES, "Invalid state character"
        state_pos[s].append(i)
    return [pl.make_scatter(x=pos_list,
                            y=[pread.counts[i] for i in pos_list],
                            mode="markers",
                            marker_size=4,
                            col=STATE_TO_COL[state],
                            name=state,
                            show_legend=show_legend,
                            show_init=show_init)
            for state, pos_list in state_pos.items()]


def gen_fig_preads(traces: Union[BaseTraceType, List[BaseTraceType]],
                   layout: Optional[go.Layout] = None) -> go.Figure:
    """Utility for generating a plot of a single-read profile.

    positional arguments:
      @ traces : Plotly trace object(s) made by:
                   - gen_trace_pread_counts()
                   - gen_trace_pread_bases()
                   - gen_fig_preads()
                 or your custom codes.
    
    optional arguments:
      @ layout : For any additional layouts.
    """
    _layout = pl.make_layout(x_title="Position",
                             y_title="Count")
    if layout is not None:
        _layout = pl.merge_layout(_layout, layout, overwrite=True)
    fig = pl.make_figure(traces, _layout)
    fig.update_layout(updatemenus=[
        dict(type="buttons",
             buttons=[dict(label="<100",
                           method="relayout",
                           args=[{"yaxis.range[0]": 0,
                                  "yaxis.range[1]": 100}])])])
    return fig
