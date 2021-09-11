from __future__ import annotations
from dataclasses import dataclass, field
from typing import Union, Optional
from collections import Counter
import plotly.graph_objects as go
import plotly_light as pl
from bits.util import RelCounter


@dataclass
class CountHistVisualizer:
    width:         Optional[int] = 700
    height:        Optional[int] = 500
    relative:      bool = False
    show_legend:   bool = False
    use_histogram: bool = False
    barmode:       str = "overlay"
    
    traces:        pl.Traces = field(default_factory=list)

    def __post_init__(self):
        if not isinstance(self.traces, list):
            self.traces = [self.traces]

    def add_trace(self,
                  count_freqs: Union[Counter, RelCounter],
                  col: Optional[str] = None,
                  opacity: float = 1,
                  name: Optional[str] = None) -> CountHistVisualizer:
        """
        positional arguments:
          @ count_freqs : Count distribution.

        optional arguments:
          @ col         : Of histogram.
          @ opacity     : Of histogram color.
                          (1 for primary and 0.7 for secondary are recommended.)
          @ name        : For plot.
        """
        self.traces.append(
            pl.make_hist((lambda x: x.relative() if self.relative else x)
                         (RelCounter(count_freqs)),
                         bin_size=1,
                         col=col,
                         opacity=opacity,
                         name=name,
                         show_legend=self.show_legend,
                         use_histogram=self.use_histogram))
        return self

    def show(self,
             layout: Optional[go.Layout] = None,
             return_fig: bool = False) -> Optional[go.Figure]:
        """
        optional arguments:
          @ layout     : Any additional layouts.
          @ return_fig : If True, return go.Figure object.
        """
        _layout = pl.make_layout(width=self.width,
                                 height=self.height,
                                 x_title="K-mer count",
                                 y_title=("Frequency" if not self.relative
                                          else "Relative frequency [%]"),
                                 barmode=self.barmode)
        fig = pl.make_figure(self.traces,
                             pl.merge_layout(_layout, layout))
        return fig if return_fig else pl.show(fig)
