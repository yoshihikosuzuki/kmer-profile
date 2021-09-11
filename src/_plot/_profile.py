from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Union, List, Tuple, Dict
from collections import defaultdict
import plotly.graph_objects as go
import plotly_light as pl
from logzero import logger
from .. import (Ctype, Wtype, STATES, Intvl, ErrorIntvl, ProfiledRead,
                STATE_NAMES, CTX_NAMES, WALL_NAMES, CTX_COLS)
from ._color import S_TO_COL


@dataclass(repr=False)
class PreadVisualizer:
    pread:       ProfiledRead
    max_count:   Optional[int] = None
    title:       Optional[str] = None
    width:       Optional[int] = None
    height:      Optional[int] = None
    use_webgl:   bool = True
    traces:      pl.Traces = field(init=False, default_factory=list)

    pos_list:      List[int] = field(init=False)
    capped_counts: List[int] = field(init=False)

    def cap_cnt(self, cnt: int) -> int:
        return cnt if self.max_count is None else min(cnt, self.max_count)

    def __post_init__(self):
        pread = self.pread
        self.pos_list = list(range(pread.length))
        self.capped_counts = [self.cap_cnt(c) for c in pread.counts]

    def add_traces(self, traces: pl.Traces) -> PreadVisualizer:
        self.traces += traces if isinstance(traces, list) else [traces]
        return self

    def add_counts(self,
                   col: str = "black",
                   name: str = "Counts",
                   show_legend: bool = True,
                   show_init: bool = True) -> PreadVisualizer:
        pread = self.pread
        t = (None if pread._seq is None or pread.K is None else
             [f"pos = {i} (-(k-1)={i-pread.K+1}, +(k-1)={i+pread.K-1}), count = {c}<br>"
              f"k-mer = {pread._seq[i:i + pread.K]}<br>"
              for i, c in enumerate(pread.counts)])
        return self.add_traces(
            [pl.make_scatter(x=self.pos_list,
                             y=self.capped_counts,
                             text=t,
                             mode=mode,
                             col=col,
                             name=name,
                             show_legend=show_legend,
                             show_init=_show_init,
                             use_webgl=self.use_webgl)
             for mode, _show_init in [("lines", show_init),
                                      ("markers", False)]])

    def add_bases(self,
                  col: str = "black",
                  name: str = "Bases",
                  show_legend: bool = True,
                  show_init: bool = False) -> PreadVisualizer:
        return self.add_traces(
            pl.make_scatter(x=self.pos_list,
                            y=self.capped_counts,
                            text=list(self.pread.seq),
                            text_col=col,
                            text_pos="top center",
                            mode="text",
                            name=name,
                            show_legend=show_legend,
                            show_init=show_init,
                            use_webgl=self.use_webgl))

    def add_states(self,
                   marker_size: int = 4,
                   names: Tuple[len(STATES) * (str,)] = STATE_NAMES,
                   show_legend: bool = True,
                   show_init: bool = True) -> PreadVisualizer:
        assert self.pread.states is not None, "No classificatons available"
        state_pos = [list(filter(lambda i: self.pread.states[i] == s, self.pos_list))
                     for s in STATES]
        return self.add_traces(
            [pl.make_scatter(x=pos_list,
                             y=[self.capped_counts[i] for i in pos_list],
                             mode="markers",
                             marker_size=marker_size,
                             col=S_TO_COL[s],
                             name=f"{name}-mers",
                             show_legend=show_legend,
                             show_init=show_init,
                             use_webgl=self.use_webgl)
             for s, name, pos_list in zip(STATES, names, state_pos)
             if len(pos_list) > 0])

    def add_intvls(self,
                   col: str = "black",
                   name: str = "Intervals",
                   show_asgn: bool = False,
                   show_legend: bool = True,
                   show_init: bool = True) -> PreadVisualizer:
        return self._add_intvls(
            [(I.b, I.e - 1, I.cb, I.ce, I.pe, I.asgn) for I in self.pread.intvls],
            col, name, show_asgn, show_legend, show_init)

    def add_rel_intvls(self,
                       col: str = "black",
                       name: str = "Rel Intervals",
                       show_asgn: bool = False,
                       show_legend: bool = True,
                       show_init: bool = True) -> PreadVisualizer:
        return self._add_intvls(
            [(I.b, I.e - 1, I.ccb, I.cce, I.pe, I.asgn) for I in self.pread.rel_intvls],
            col, name, show_asgn, show_legend, show_init)

    def _add_intvls(self,
                    data: Tuple,
                    col: str,
                    name: str,
                    show_asgn: bool,
                    show_legend: bool,
                    show_init: bool) -> PreadVisualizer:
        N = len(data) * 3
        x, y, t = [None] * N, [None] * N, [None] * N
        for i, (b, e, cb, ce, pe, _) in enumerate(data):
            for j, (p, c) in enumerate([(b, cb), (e, ce)]):
                idx = 3 * i + j
                x[idx] = p
                y[idx] = self.cap_cnt(c)
                t[idx] = f"intvls[{i}] @{p} ({c}; pe={pe:7f})"

        traces = [pl.make_scatter(x=x, y=y, text=t,
                                  mode="markers+lines",
                                  col=col,
                                  name=name,
                                  show_legend=show_legend,
                                  show_init=show_init)]
        if show_asgn:
            N = len(data) * 2
            x, y, t = [None] * N, [None] * N, [None] * N
            for i, (b, e, cb, ce, _, s) in enumerate(data):
                for j, (p, c) in enumerate([(b, cb), (e, ce)]):
                    idx = 2 * i + j
                    x[idx] = p
                    y[idx] = c if self.max_count is None else min(self.max_count, c)
                    t[idx] = S_TO_COL[s]
            traces.append(pl.make_scatter(x=x, y=y, col=t))
        return self.add_traces(traces)

    def add_simple_intvls(self,
                          intvls: List[Union[Intvl, ErrorIntvl]],
                          col: str = "black",
                          name: str = "Intervals",
                          show_legend: bool = True,
                          show_init: bool = True) -> pl.BaseTraceType:
        """For simple intervals with only `I.b` and `I.e`."""
        N = len(intvls) * 3
        x, y, t = [None] * N, [None] * N, [None] * N
        for i, I in enumerate(intvls):
            for j, p in enumerate([I.b, I.e - 1]):
                idx = 3 * i + j
                c = self.pread.counts[p]
                x[idx] = p
                y[idx] = self.cap_cnt(c)
                t[idx] = f"intvls[{i}] @{p} ({c})"
        return self.add_traces(
            pl.make_scatter(x=x, y=y, text=t,
                            mode="markers+lines",
                            col=col,
                            name=name,
                            show_legend=show_legend,
                            show_init=show_init))

    def add_depth(self,
                  depths: Dict[str, int],
                  dthres: Dict[str, float] = None,
                  opacity: float = 1,
                  depth_width: float = 2,
                  dthres_col: str = "black",
                  dthres_width: float = 1,
                  name: Optional[str] = None,
                  use_webgl: bool = True,
                  show_legend: bool = True,
                  show_init: bool = True) -> List[pl.BaseTraceType]:
        def _trace_depth(depth: int, col: str, width: float, name: str):
            return pl.make_lines((0, depth, self.pread.length, depth),
                                 col=col,
                                 opacity=opacity,
                                 name=name,
                                 width=width,
                                 use_webgl=use_webgl,
                                 show_legend=show_legend,
                                 show_init=show_init)

        traces = [_trace_depth(depths[s],
                               S_TO_COL[s],
                               depth_width,
                               f"{s} {'' if name is None else name} depth")
                  for s in STATES]
        traces += [_trace_depth(val,
                                dthres_col,
                                dthres_width,
                                f"{s} {'' if name is None else name} threshold")
                   for s, val in dthres.items()]
        return self.add_traces(traces)

    def add_ctx(self,
                min_vals: Tuple[len(Ctype) * (int,)] = (4, 3, 2),
                cols: Tuple[len(Ctype) * (str, str)] = CTX_COLS,
                ctx_names: Tuple[len(Ctype) * (str,)] = CTX_NAMES,
                wall_names: Tuple[len(Wtype) * (str,)] = WALL_NAMES,
                use_webgl: bool = True,
                show_legend: bool = True,
                show_init: bool = True) -> List[pl.BaseTraceType]:
        ctx_by_type = [list(zip(*x)) for x in list(zip(*self.pread.ctx))]
        return self.add_traces(
            list(filter(None,
                        [trace_minus(data, col, f"{ctype} length (>={min_val}) [{wtype}]", min_val,
                                     use_webgl, show_legend, show_init)
                         for ctx, ctype, ctx_cols, min_val in zip(ctx_by_type, ctx_names, cols, min_vals)
                         for data, col, wtype in zip(ctx, ctx_cols, wall_names)])))

    def add_walls(self,
                  col: str = "gold",
                  name: str = "Walls",
                  opacity: float = 1.,
                  show_legend: bool = True,
                  show_init: bool = True) -> pl.BaseTraceType:
        pread = self.pread
        max_count = self.max_count if self.max_count is not None else max(pread.counts)
        return self.add_traces(
            pl.make_lines([(i, 0, i, max_count) for i in pread.walls],
                          text=[f"{pread.counts[i - 1]}@{i - 1}" if i > 0 else ""
                                + f"->{pread.counts[i]}@{i}" if i < pread.length else ""
                                for i in pread.walls],
                          col=col,
                          opacity=opacity,
                          name=name,
                          use_webgl=self.use_webgl,
                          show_legend=show_legend,
                          show_init=show_init))

    def show(self,
             layout: Optional[go.Layout] = None,
             max_count_zoom: Optional[int] = 100,
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
        fixed_layout = pl.make_layout(title=self.title,

                                      width=self.width,
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


def trace_minus(data: List[float],
                col: str,
                name: str,
                min_val: float = 0,
                use_webgl: bool = True,
                show_legend: bool = True,
                show_init: bool = True) -> pl.BaseTraceType:
    _x = {i: -x for i, x in enumerate(data) if x >= min_val}
    _t = [f"{name} = {round(x, 3)}<br>pos = {i}"
          for i, x in enumerate(data) if x >= min_val]
    if len(_x) == 0:
        return None
    return pl.make_hist(_x,
                        text=_t,
                        col=col,
                        use_lines=True,
                        name=name,
                        use_webgl=use_webgl,
                        show_legend=show_legend,
                        show_init=show_init)
