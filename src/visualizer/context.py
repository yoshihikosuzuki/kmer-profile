from typing import Optional, Sequence, List, Tuple, Dict
from logzero import logger
import plotly_light as pl
from ..type import STATE_TO_COL, ProfiledRead, SeqCtx, ErrorModel


def trace_minus(data: Sequence[float],
                col: str,
                name: str,
                min_val: float = 0,
                show_legend: bool = True,
                show_init: bool = True) -> pl.BaseTraceType:
    return pl.make_hist({i: -x for i, x in enumerate(data) if x >= min_val},
                        text=[f"{name} = {round(x, 3)}<br>pos = {i}"
                              for i, x in enumerate(data) if x >= min_val],
                        col=col,
                        use_lines=True,
                        name=name,
                        show_legend=show_legend,
                        show_init=show_init)


def trace_ctx(pread: ProfiledRead,
              min_vals: Tuple[int, int] = (4, 3, 2),
              show_legend: bool = True,
              show_init: bool = True) -> List[pl.BaseTraceType]:
    assert len(min_vals) == len(pread.ctx)
    return [trace_minus(data, col, f"{ctx.emodel.name} length (>={min_val}) [{_type}]", min_val,
                        show_legend, show_init)
            for ctx, min_val in zip(pread.ctx, min_vals)
            for data, col, _type in zip(ctx.lens, ctx.emodel.cols, ("drop", "gain"))]


def trace_pe(pread: ProfiledRead,
             min_val: float = 1e-5,
             show_legend: bool = True,
             show_init: bool = True) -> List[pl.BaseTraceType]:
    return [trace_minus([x * 100 for x in pread.pe["self"][_type]],
                        col, f"%Pr{{error in self}} [{_type}]", min_val,
                        show_legend, show_init)
            for _type, col in zip(("drop", "gain"), ("deepskyblue", "deeppink"))]


def trace_depth(pread: ProfiledRead,
                depths: Dict[str, int],
                thres: Dict[str, float] = None,
                opacity: float = 1,
                col_thres: str = "tomato",
                width_thres: float = 2,
                name: Optional[str] = None,
                show_legend: bool = True,
                show_init: bool = True) -> List[pl.BaseTraceType]:
    def _trace_depth(depth: int, col: str, width: float, name: str):
        return pl.make_lines((0, depth, pread.length, depth),
                             col=col,
                             opacity=opacity,
                             name=name,
                             width=width,
                             show_legend=show_legend,
                             show_init=show_init)

    traces = [_trace_depth(depths[s],
                           STATE_TO_COL[s],
                           1,
                           f"{s} {'' if name is None else name} depth")
              for s in ('H', 'D')]
    if thres is not None:
        traces += [_trace_depth(val,
                                col_thres,
                                width_thres,
                                f"{s} {'' if name is None else name} threshold")
                   for s, val in thres.items()]
    return traces


def trace_ns(pread: ProfiledRead,
             max_count: Optional[int] = None,
             show_legend: bool = True,
             show_init: bool = True) -> pl.BaseTraceType:
    if max_count is None:
        max_count = max(pread.counts)
    return pl.make_lines([(i, 0, i, max_count) for i, ns in enumerate(pread.ns) if ns],
                         text=[f"{pread.counts[i - 1]}@{i - 1}->{pread.counts[i]}@{i}"
                               for i, ns in enumerate(pread.ns) if ns],
                         col="goldenrod",
                         opacity=0.5,
                         name="Non-smooth points",
                         show_legend=show_legend,
                         show_init=show_init)
