from typing import Union, Optional, Sequence, List
from collections import defaultdict, Counter
import plotly.graph_objects as go
import plotly_light as pl
from ..type import RelCounter


def gen_fig_count_dist(counts: Union[Union[Counter, RelCounter],
                                     List[Union[Counter, RelCounter]]],
                       names: Union[str, List[str]],
                       cols: Optional[Union[str, List[str]]] = None,
                       relative: bool = False,
                       layout: Optional[go.Layout] = None) -> go.Figure:
    if not isinstance(counts, list):
        counts = [counts]
    if not isinstance(names, list):
        names = [names]
    assert len(counts) == len(names), "Inconsistent lengths"
    if cols is not None:
        if not isinstance(cols, list):
            cols = [cols]
        assert len(counts) == len(cols), "Inconsistent lengths"
    traces = []
    for i, (count, name) in enumerate(zip(counts, names)):
        if relative and not isinstance(count, RelCounter):
            count = RelCounter(count)
        traces.append(pl.make_hist(count if not relative else count.relative(),
                                   bin_size=1,
                                   col=cols[i] if cols is not None else None,
                                   opacity=1 if i == 0 else 0.7,
                                   name=name,
                                   show_legend=True))
    _layout = pl.make_layout(x_title="K-mer count",
                             y_title=("Frequency" if not relative
                                      else "Relative frequency [%]"),
                             barmode="overlay")
    if layout is None:
        _layout = pl.merge_layout(_layout, layout, overwrite=True)
    return go.Figure(data=traces, layout=_layout)
