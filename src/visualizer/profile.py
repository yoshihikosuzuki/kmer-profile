from __future__ import annotations
from dataclasses import dataclass, field
from typing import Union, Optional, Sequence, List
from collections import defaultdict
import plotly.graph_objects as go
from plotly.basedatatypes import BaseTraceType
import plotly_light as pl
from ..type import STATES, STATE_TO_COL, ProfiledRead


@dataclass
class ProfiledReadVisualizer:
    show_legend: bool = True
    traces: pl.Traces = field(default_factory=list)

    def __post_init__(self):
        if not isinstance(self.traces, list):
            self.traces = [self.traces]

    def add_trace_counts(self,
                         pread: ProfiledRead,
                         col: str = "black",
                         name: str = "Counts",
                         show_legend: bool = True) -> ProfiledReadVisualizer:
        self.traces.append(
            pl.make_scatter(
                x=list(range(pread.length)),
                y=pread.counts,
                text=[f"pos = {i}<br>count = {c}<br>"
                      f"k-mer = {pread.seq[i - pread.K + 1:i + 1] if i >= pread.K - 1 else '-'}"
                      for i, c in enumerate(pread.counts)],
                mode="lines",
                col=col,
                name=name,
                show_legend=show_legend))
        return self

    def add_trace_bases(self,
                        pread: ProfiledRead,
                        col: str = "black",
                        name: str = "Bases",
                        show_legend: bool = True,
                        show_init: bool = False) -> ProfiledReadVisualizer:
        self.traces.append(
            pl.make_scatter(x=list(range(pread.length)),
                            y=pread.counts,
                            text=list(pread.seq),
                            text_pos="top center",
                            mode="text",
                            name=name,
                            show_legend=show_legend,
                            show_init=show_init))
        return self

    def add_trace_states(self,
                         pread: ProfiledRead,
                         show_legend: bool = True,
                         show_init: bool = False) -> ProfiledReadVisualizer:
        assert len(pread.states) == pread.length
        state_pos = defaultdict(list)
        for i, s in enumerate(pread.states):
            assert s in STATES, "Invalid state character"
            state_pos[s].append(i)
        self.traces += \
            [pl.make_scatter(x=pos_list,
                             y=[pread.counts[i] for i in pos_list],
                             mode="markers",
                             marker_size=4,
                             col=STATE_TO_COL[state],
                             name=state,
                             show_legend=show_legend,
                             show_init=show_init)
             for state, pos_list in state_pos.items()]
        return self

    def show(self,
             layout: Optional[go.Layout] = None,
             return_fig: bool = False) -> Optional[go.Figure]:
        """
        optional arguments:
          @ layout : For any additional layouts.
          @ return_fig : If True, return go.Figure object.
        """
        _layout = pl.make_layout(x_title="Position",
                                 y_title="Count")
        fig = pl.make_figure(self.traces,
                             pl.merge_layout(_layout, layout))
        fig.update_layout(updatemenus=[
            dict(type="buttons",
                 buttons=[dict(label="<100",
                               method="relayout",
                               args=[{"yaxis.range[0]": 0,
                                      "yaxis.range[1]": 100}])])])
        return fig if return_fig else pl.show(fig)
