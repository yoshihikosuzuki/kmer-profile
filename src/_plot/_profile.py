from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List
from collections import defaultdict
import plotly.graph_objects as go
import plotly_light as pl
from .. import Intvl, ProfiledRead
from ._color import S_TO_COL


@dataclass
class ProfiledReadVisualizer:
    max_count:   Optional[int] = None
    width:       Optional[int] = None
    height:      Optional[int] = None
    show_legend: bool = True
    use_webgl:   bool = False
    traces:      pl.Traces = field(default_factory=list)

    def __post_init__(self):
        if not isinstance(self.traces, list):
            self.traces = [self.traces]

    def add_traces(self,
                   traces: pl.Traces) -> ProfiledReadVisualizer:
        self.traces += traces if isinstance(traces, list) else [traces]
        return self

    def add_pread(self,
                  pread: ProfiledRead,
                  col: str = "black",
                  show_legend: bool = True,
                  show_init_counts: bool = True,
                  show_init_bases: bool = False,
                  show_init_states: bool = False) -> ProfiledReadVisualizer:
        self.add_counts(pread.counts,
                        pread._seq,
                        pread.K,
                        col,
                        show_legend=show_legend,
                        show_init=show_init_counts)
        self.add_bases(pread.seq,
                       pread.counts,
                       show_legend=show_legend,
                       show_init=show_init_bases)
        if pread.states is not None:
            self.add_states(pread.states,
                            pread.counts,
                            show_legend=show_legend,
                            show_init=show_init_states)
        return self

    def add_counts(self,
                   counts: List[int],
                   _seq: Optional[str] = None,
                   K: Optional[int] = None,
                   col: str = "black",
                   name: str = "Counts",
                   show_legend: bool = True,
                   show_init: bool = True) -> ProfiledReadVisualizer:
        x = list(range(len(counts)))
        y = (counts if self.max_count is None
             else [min(self.max_count, c) for c in counts])
        if _seq is None or K is None:
            text = None
        else:
            assert len(_seq) == len(counts) + K - 1
            text = [f"pos = {i} (-(k-1)={i-K+1}, +(k-1)={i+K-1}), count = {c}<br>"
                    f"k-mer = {_seq[i:i + K]}<br>"
                    for i, c in enumerate(counts)]
        return self.add_traces(
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
                             use_webgl=self.use_webgl)])

    def add_bases(self,
                  seq: str,
                  counts: List[int],
                  name: str = "Bases",
                  show_legend: bool = True,
                  show_init: bool = False) -> ProfiledReadVisualizer:
        return self.add_traces(
            pl.make_scatter(x=list(range(len(seq))),
                            y=(counts if self.max_count is None
                               else [min(self.max_count, c) for c in counts]),
                            text=list(seq),
                            text_col="black",
                            text_pos="top center",
                            mode="text",
                            name=name,
                            show_legend=show_legend,
                            show_init=show_init,
                            use_webgl=self.use_webgl))

    def add_states(self,
                   states: List[str],
                   counts: List[int],
                   show_legend: bool = True,
                   show_init: bool = False) -> ProfiledReadVisualizer:
        state_pos = defaultdict(list)
        for i, s in enumerate(states):
            assert s in S_TO_COL, "Invalid state character"
            state_pos[s].append(i)
        return self.add_traces(
            [pl.make_scatter(x=pos_list,
                             y=[(counts[i] if self.max_count is None
                                else min(self.max_count, counts[i]))
                                for i in pos_list],
                             mode="markers",
                             marker_size=4,
                             col=S_TO_COL[state],
                             name=state,
                             show_legend=show_legend,
                             show_init=show_init,
                             use_webgl=self.use_webgl)
             for state, pos_list in state_pos.items()])

    def add_intvls(self,
                   intvls: List[Intvl],
                   use_corrected_counts: bool = False,
                   col: str = "black",
                   name: str = "Intervals",
                   show_asgn: bool = False,
                   show_legend: bool = True,
                   show_init: bool = True) -> ProfiledReadVisualizer:
        x, y, t = [None] * (len(intvls) * 3), [None] * (len(intvls) * 3), [None] * (len(intvls) * 3)
        for i, I in enumerate(intvls):
            for j, (p, c) in enumerate([(I.b, I.cb if not use_corrected_counts else I.ccb),
                                        (I.e - 1, I.ce if not use_corrected_counts else I.cce)]):
                idx = 3 * i + j
                x[idx] = p
                y[idx] = c if self.max_count is None else min(self.max_count, c)
                t[idx] = f"intvls[{i}] @{p} ({c}; pe={I.pe:7f})"

        traces = [pl.make_scatter(x, y, t,
                                  mode="markers+lines",
                                  col=col,
                                  name=name,
                                  show_legend=show_legend,
                                  show_init=show_init)]
        if show_asgn:
            traces.append(
                pl.make_scatter([x for I in intvls for x in [I.b, I.e - 1]],
                                [c if self.max_count is None else min(self.max_count, c)
                                 for I in intvls
                                 for c in [I.cb if not use_corrected_counts else I.ccb,
                                           I.ce if not use_corrected_counts else I.cce]],
                                col=[S_TO_COL[I.asgn] for I in intvls for _ in range(2)]))
        return self.add_traces(traces)

    def show(self,
             max_count_zoom: Optional[int] = 100,
             layout: Optional[go.Layout] = None,
             return_fig: bool = False) -> Optional[go.Figure]:
        """
        optional arguments:
          @ max_count_zoom : If not None, make a button zooming in to [0,`max_count_zoom`].
          @ layout         : Additional layout.
          @ return_fig     : If True, return a `go.Figure` object.
        """
        default_layout = pl.layout(x_title="Position",
                                   y_title=("K-mer count" if self.max_count is None
                                            else f"K-mer count (capped at {self.max_count})"),
                                   x_grid=False,
                                   y_grid=False)
        fixed_layout = pl.make_layout(width=self.width,
                                      height=self.height)
        fig = pl.make_figure(self.traces,
                             pl.merge_layout(default_layout, layout, fixed_layout))
        if max_count_zoom is not None and (self.max_count is None or max_count_zoom < self.max_count):
            fig.update_layout(
                updatemenus=[dict(type="buttons",
                                  buttons=[dict(label=f"<{max_count_zoom}",
                                                method="relayout",
                                                args=[{"yaxis.range[0]": 0,
                                                       "yaxis.range[1]": max_count_zoom + 1}])])])
        return fig if return_fig else pl.show(fig)
