from typing import Optional, Sequence, List, Tuple, Dict
import plotly_light as pl
from .. import STATES, ProfiledRead, Ctype, Etype, Wtype
from .._const import CTX_NAMES, ERR_NAMES, WALL_NAMES
from ._color import S_TO_COL


def trace_depth(pread: ProfiledRead,
                depths: Dict[str, int],
                dthres: Dict[str, float] = None,
                opacity: float = 1,
                depth_width: float = 2,
                dthres_col: str = "red",
                dthres_width: float = 1,
                name: Optional[str] = None,
                use_webgl: bool = True,
                show_legend: bool = True,
                show_init: bool = True) -> List[pl.BaseTraceType]:
    def _trace_depth(depth: int, col: str, width: float, name: str):
        return pl.make_lines((0, depth, pread.length, depth),
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
    return traces


def trace_minus(data: Sequence[float],
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


def trace_ctx(pread: ProfiledRead,
              min_vals: Tuple[len(Ctype) * (int,)] = (4, 3, 2),
              cols: Tuple[len(Ctype) * (str, str)] = (("dodgerblue", "coral"),
                                                      ("teal", "firebrick"),
                                                      ("olive", "indigo")),
              use_webgl: bool = True,
              show_legend: bool = True,
              show_init: bool = True) -> List[pl.BaseTraceType]:
    ctx_by_type = [list(zip(*x)) for x in list(zip(*pread.ctx))]
    return list(filter(None,
                       [trace_minus(data, col, f"{ctype} length (>={min_val}) [{wtype}]", min_val,
                                    use_webgl, show_legend, show_init)
                        for ctx, ctype, ctx_cols, min_val in zip(ctx_by_type, CTX_NAMES, cols, min_vals)
                        for data, col, wtype in zip(ctx, ctx_cols, WALL_NAMES)]))


# def trace_pe(pread: ProfiledRead,
#              min_val: float = 1e-5,
#              cols: Tuple[len(Wtype) * (str,)] = ("deepskyblue", "deeppink"),
#              use_webgl: bool = True,
#              show_legend: bool = True,
#              show_init: bool = True) -> List[pl.BaseTraceType]:
#     return [trace_minus([x * 100 for x in pread.pe[Etype.SELF][w.value]],
#                         col, f"%Pr{{error in self}} [{wtype}]", min_val,
#                         use_webgl, show_legend, show_init)
#             for w, wtype, col in zip(Wtype, WALL_NAMES, cols)]


def trace_wall(pread: ProfiledRead,
               max_count: Optional[int] = None,
               col: str = "gold",
               opacity: float = 0.6,
               use_webgl: bool = True,
               show_legend: bool = True,
               show_init: bool = True) -> pl.BaseTraceType:
    if max_count is None:
        max_count = max(pread.counts)
    return pl.make_lines([(i, 0, i, max_count) for i in pread.walls],
                         text=[f"{pread.counts[i - 1]}@{i - 1}" if i > 0 else ""
                               + f"->{pread.counts[i]}@{i}" if i < pread.length else ""
                               for i in pread.walls],
                         col=col,
                         opacity=opacity,
                         name="Walls",
                         use_webgl=use_webgl,
                         show_legend=show_legend,
                         show_init=show_init)
