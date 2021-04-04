import argparse
from os.path import isfile
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, List, Dict
from logzero import logger
import plotly.graph_objects as go
import plotly_light as pl
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from bits.util import RelCounter
import fastk
from .type import ProfiledRead
from .classifier import load_pread
from .visualizer import CountDistVisualizer, ProfiledReadVisualizer

app = dash.Dash(__name__,
                external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])


@dataclass(repr=False, eq=False)
class Cache:
    """Cache for data that are needed to be shared over multiple operations."""
    args: argparse.Namespace = None
    cdv: Optional[CountDistVisualizer] = None
    prv: Optional[ProfiledReadVisualizer] = None
    pread: Optional[ProfiledRead] = None
    states: Optional[Dict] = None


cache = Cache()


def reset_axes(fig: go.Figure) -> go.Layout:
    """Return the layout of the figure with reset axes."""
    return pl.merge_layout(fig["layout"] if "layout" in fig else None,
                           pl.make_layout(x_range=None, y_range=None))

### ----------------------------------------------------------------------- ###
###                        k-mer count distribution                         ###
### ----------------------------------------------------------------------- ###


@app.callback(
    Output('fig-dist', 'figure'),
    [Input('submit-dist', 'n_clicks'),
     Input('submit-profile', 'n_clicks')],
    [State('read-id', 'value'),
     State('max-count-dist', 'value'),
     State('fig-dist', 'figure')]
)
def update_count_dist(_n_clicks_dist: int,
                      _n_clicks_profile: int,
                      read_id: Optional[str],
                      max_count: Optional[str],
                      fig: go.Figure) -> go.Figure:
    """Update the aggregated k-mer count distribution."""
    global cache
    ctx = dash.callback_context
    if not max_count:
        raise PreventUpdate
    max_count = int(max_count)
    if (not ctx.triggered
            or ctx.triggered[0]["prop_id"] == "submit-dist.n_clicks"):
        # Global k-mer count distribution
        cache.cdv = CountDistVisualizer(relative=True)
        cache.cdv.add_trace(fastk.histex(cache.args.fastk_prefix,
                                         max_count=max_count),
                            col="gray",
                            opacity=1,
                            name="Global")
        return cache.cdv.show(layout=reset_axes(fig),
                              return_fig=True)
    elif ctx.triggered[0]["prop_id"] == "submit-profile.n_clicks":
        # Single-read k-mer count distribution
        if not read_id:
            raise PreventUpdate
        read_id = int(read_id)
        prof = fastk.profex(cache.args.fastk_prefix, read_id)
        return (deepcopy(cache.cdv)
                .add_trace(RelCounter([min(c, max_count) for c in prof]),
                           col="turquoise",
                           opacity=0.7,
                           name=f"Read {read_id}")
                .show(layout=reset_axes(fig),
                      return_fig=True))
    raise PreventUpdate


### ----------------------------------------------------------------------- ###
###                           k-mer count profile                           ###
### ----------------------------------------------------------------------- ###


@app.callback(
    Output('fig-profile', 'figure'),
    [Input('submit-profile', 'n_clicks')],
    [State('read-id', 'value'),
     State('max-count-prof', 'value'),
     State('class-init', 'value'),
     State('fig-profile', 'figure')]
)
def update_kmer_profile(_n_clicks_profile: int,
                        read_id: Optional[str],
                        max_count: Optional[str],
                        class_init: List[str],
                        fig: go.Figure) -> go.Figure:
    """Update the count profile plot."""
    global cache
    ctx = dash.callback_context
    if not ctx.triggered or not read_id:
        raise PreventUpdate
    read_id = int(read_id)
    max_count = int(max_count) if max_count else None
    if ctx.triggered[0]["prop_id"] == "submit-profile.n_clicks":
        # Draw a k-mer count profile from scratch
        cache.pread = load_pread(read_id,
                                 cache.args.fastk_prefix,
                                 cache.args.seq_fname)
        cache.pread.states = (None if cache.states is None
                              else 'E' * (cache.pread.K - 1) + cache.states[read_id])
        if cache.pread is None:
            raise PreventUpdate
        cache.prv = (ProfiledReadVisualizer(max_count=max_count)
                     .add_pread(cache.pread, show_init_states="SHOW" in class_init))
        return (cache.prv.show(layout=reset_axes(fig),
                               return_fig=True))
    raise PreventUpdate


### ----------------------------------------------------------------------- ###
###                   Layout, Command-line arguments, etc.                  ###
### ----------------------------------------------------------------------- ###


def main():
    global cache
    parse_args()
    app.layout = html.Div(children=[
        html.Div([html.Button(id='submit-dist',
                              n_clicks=0,
                              children='Draw k-mer count distribution'),
                  " [OPTIONS]",
                  " Max count = ",
                  dcc.Input(id='max-count-dist',
                            value='100',
                            type='number')]),
        dcc.Graph(id='fig-dist',
                  figure=go.Figure(
                      layout=pl.make_layout(width=800,
                                            height=400)),
                  config=dict(toImageButtonOptions=dict(format="svg"))),
        html.Div(["Read ID: ",
                  dcc.Input(id='read-id',
                            value='',
                            type='number')]),
        html.Div([html.Button(id='submit-profile',
                              n_clicks=0,
                              children='Draw k-mer count profile'),
                  " [OPTIONS]",
                  " Max count = ",
                  dcc.Input(id='max-count-prof',
                            value='',
                            type='number'),
                  dcc.Checklist(id='class-init',
                                options=[{'label': 'Show classifications from the beginning',
                                          'value': 'SHOW'}],
                                value=[])]),
        dcc.Graph(id='fig-profile',
                  figure=go.Figure(
                      layout=pl.make_layout(width=1800,
                                            height=500)),
                  config=dict(toImageButtonOptions=dict(format="svg")))
    ])
    app.run_server(port=int(cache.args.port_number),
                   debug=cache.args.debug_mode)


def parse_args() -> argparse.Namespace:
    global cache
    parser = argparse.ArgumentParser(
        description="K-mer count profile visualizer")
    parser.add_argument(
        "fastk_prefix",
        type=str,
        help="Prefix of FastK's output files.")
    parser.add_argument(
        "-s",
        "--seq_fname",
        type=str,
        default=None,
        help="Name of the input file for FastK. Must be .db/dam/fast[a|q]. [None]")
    parser.add_argument(
        "-c",
        "--class_fname",
        type=str,
        default=None,
        help="K-mer classification result file name. [None]")
    parser.add_argument(
        "-p",
        "--port_number",
        type=int,
        default=8050,
        help="Port number to run the server. [8050]")
    parser.add_argument(
        "-d",
        "--debug_mode",
        action="store_true",
        help="Run a Dash server in a debug mode.")
    cache.args = args = parser.parse_args()
    for fname in [f"{args.fastk_prefix}.hist",
                  f"{args.fastk_prefix}.prof",
                  args.seq_fname,
                  args.class_fname]:
        assert fname is None or isfile(fname), f"{fname} does not exist"
    if args.class_fname is not None:
        cache.states = {}
        with open(args.class_fname, 'r') as f:
            for line in f:
                read_id, states = line.strip().split('\t')
                cache.states[int(read_id)] = states


if __name__ == '__main__':
    main()
