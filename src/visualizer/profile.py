from __future__ import annotations
from dataclasses import dataclass, field
from typing import Union, Optional, Sequence, List, Tuple
from collections import defaultdict
import plotly.graph_objects as go
from plotly.basedatatypes import BaseTraceType
import plotly_light as pl
from ..type import STATES, STATE_TO_COL, ProfiledRead


@dataclass
class ProfiledReadVisualizer:
    show_legend: bool = True
    use_webgl: bool = True
    traces: pl.Traces = field(default_factory=list)

    def __post_init__(self):
        if not isinstance(self.traces, list):
            self.traces = [self.traces]

    def add_pread(self,
                  pread: ProfiledRead,
                  col: str = "black",
                  name: Optional[str] = None,
                  show_legend: bool = True,
                  show_init_counts: bool = True,
                  show_init_bases: bool = False,
                  show_init_states: bool = False) -> ProfiledReadVisualizer:
        """Utility for simultaneously adding counts, bases, and states when a
        profiled read is available.
        """
        name = name if name is not None else f"Read {pread.id}"
        self.add_counts(pread.counts,
                        pread.seq,
                        pread.K,
                        col,
                        name,
                        show_legend,
                        show_init_counts)
        self.add_bases(pread.seq,
                       pread.counts,
                       col,
                       name,
                       show_legend,
                       show_init_bases)
        if pread.states is not None:
            self.add_states(pread.states,
                            pread.counts,
                            show_legend,
                            show_init_states)
        return self

    def add_counts(self,
                   counts: List[int],
                   seq: Optional[str] = None,
                   K: Optional[int] = None,
                   col: str = "black",
                   name: str = "Counts",
                   show_legend: bool = True,
                   show_init: bool = True) -> ProfiledReadVisualizer:
        x = list(range(len(counts)))
        y = counts
        text = ([f"pos = {i}, count = {c}<br>"
                 f"k-mer = {seq[i - K + 1:i + 1] if i >= K - 1 else '-'}<br>"
                 f"-(k-1) pos = {i-K+1}, +(k-1) pos = {i+K-1}"
                 for i, c in enumerate(counts)] if seq is not None and K is not None
                else None)
        self.traces += \
            [pl.make_scatter(x=x, y=y, text=text,
                             mode="lines",
                             col=col,
                             name=name,
                             show_legend=show_legend,
                             show_init=show_init,
                             use_webgl=self.use_webgl),
             pl.make_scatter(x=x, y=y, text=text,
                             mode="markers",
                             col=col,
                             name=name,
                             show_legend=show_legend,
                             show_init=False,
                             use_webgl=self.use_webgl)]
        return self

    def add_bases(self,
                  seq: str,
                  counts: List[int],
                  col: str = "black",
                  name: str = "Bases",
                  show_legend: bool = True,
                  show_init: bool = False) -> ProfiledReadVisualizer:
        self.traces.append(
            pl.make_scatter(x=list(range(len(seq))),
                            y=counts,
                            text=list(seq),
                            text_pos="top center",
                            mode="text",
                            name=name,
                            show_legend=show_legend,
                            show_init=show_init,
                            use_webgl=self.use_webgl))
        return self

    def add_states(self,
                   states: List[str],
                   counts: List[int],
                   show_legend: bool = True,
                   show_init: bool = False) -> ProfiledReadVisualizer:
        state_pos = defaultdict(list)
        for i, s in enumerate(states):
            assert s in STATES, "Invalid state character"
            state_pos[s].append(i)
        self.traces += \
            [pl.make_scatter(x=pos_list,
                             y=[counts[i] for i in pos_list],
                             mode="markers",
                             marker_size=4,
                             col=STATE_TO_COL[state],
                             name=state,
                             show_legend=show_legend,
                             show_init=show_init,
                             use_webgl=self.use_webgl)
             for state, pos_list in state_pos.items()]
        return self

    def add_intvls(self,
                   intvls: List[Tuple[int, int]],
                   counts: List[int],
                   states: Optional[List[str]] = None,
                   col: str = "black",
                   name: str = "Intervals",
                   show_legend: bool = True,
                   show_init: bool = True) -> ProfiledReadVisualizer:
        self.traces.append(
            pl.make_scatter([x for b, e in intvls for x in [b, e - 1, None]],
                            [counts[x] if x is not None else None
                             for b, e in intvls for x in [b, e - 1, None]],
                            text=[f"intvls[{i}] @{x} ({counts[x]})" if x is not None else None
                                  for i, (b, e) in enumerate(intvls)
                                  for x in [b, e - 1, None]],
                            mode="markers+lines",
                            col=col,
                            name=name,
                            show_legend=show_legend))
        if states is not None:
            self.traces.append(
                pl.make_scatter([x for b, e in intvls for x in [b, e - 1]],
                                [counts[x]
                                 for b, e in intvls for x in [b, e - 1]],
                                col=[STATE_TO_COL[s] for s in states for _ in range(2)]))
        return self

    def add_traces(self,
                   traces: pl.Traces) -> ProfiledReadVisualizer:
        if isinstance(traces, list):
            self.traces += traces
        else:
            self.traces.append(traces)
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
                                 y_title="Count",
                                 x_grid=False,
                                 y_grid=False)
        fig = pl.make_figure(self.traces,
                             pl.merge_layout(_layout, layout))
        fig.update_layout(
            updatemenus=[dict(type="buttons",
                              buttons=[dict(label="<100",
                                            method="relayout",
                                            args=[{"yaxis.range[0]": 0,
                                                   "yaxis.range[1]": 100}])])])
        return fig if return_fig else pl.show(fig)
