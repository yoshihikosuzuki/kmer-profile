import argparse
from os.path import splitext
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
from bits.seq import load_db
from bits.util import RelCounter
import fastk
from .type import ProfiledRead
from .classifier.heuristics import run_heuristics
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
    pread_hoco: Optional[ProfiledRead] = None
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
def update_count_dist(n_clicks_dist: int,
                      n_clicks_profile: int,
                      read_id: int,
                      max_count: int,
                      fig: go.Figure) -> go.Figure:
    """Update the aggregated k-mer count distribution."""
    global cache
    ctx = dash.callback_context
    max_count = int(max_count)
    if (not ctx.triggered
            or ctx.triggered[0]["prop_id"] == "submit-dist.n_clicks"):
        # Global k-mer count distribution
        cache.cdv = CountDistVisualizer(relative=True)
        cache.cdv.add_trace(fastk.histex(cache.args.fastk_prefix,
                                         max_count=max_count),
                            col="gray",
                            opacity=1,
                            name=("Global" if cache.args.fastk_prefix_hoco is None
                                  else "Global (normal)"))
        if cache.args.fastk_prefix_hoco is not None:
            cache.cdv.add_trace(fastk.histex(cache.args.fastk_prefix_hoco,
                                             max_count=max_count),
                                col="darkorange",
                                opacity=0.7,
                                name="Global (hoco)")
        return cache.cdv.show(layout=reset_axes(fig),
                              return_fig=True)
    elif ctx.triggered[0]["prop_id"] == "submit-profile.n_clicks":
        # Single-read k-mer count distribution
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


def pullback_hoco(hoco_profile: List[int],
                  normal_seq: str) -> List[int]:
    """Project back hoco profile onto normal space."""
    assert hoco_profile[0] == 0, "Must have (K-1) 0-counts"
    pb_profile = [None] * len(normal_seq)
    i_normal = i_hoco = 0
    pb_profile[i_normal] = hoco_profile[i_hoco]
    for i_normal in range(1, len(normal_seq)):
        if normal_seq[i_normal] != normal_seq[i_normal - 1]:
            i_hoco += 1
        pb_profile[i_normal] = hoco_profile[i_hoco]
    assert i_hoco == len(hoco_profile) - 1, "Inconsistent lengths"
    return pb_profile


@app.callback(
    Output('fig-profile', 'figure'),
    [Input('submit-profile', 'n_clicks')],
    [State('read-id', 'value'),
     State('class-init', 'value'),
     State('fig-profile', 'figure')]
)
def update_kmer_profile(n_clicks_profile: int,
                        read_id: int,
                        class_init: List[str],
                        fig: go.Figure) -> go.Figure:
    """Update the count profile plot."""
    global cache
    ctx = dash.callback_context
    if read_id != "":
        read_id = int(read_id)
    if not ctx.triggered:
        raise PreventUpdate
    if ctx.triggered[0]["prop_id"] == "submit-profile.n_clicks":
        # Draw a k-mer count profile from scratch
        seq = load_db(cache.args.db_fname, read_id)[0].seq
        prof = fastk.profex(cache.args.fastk_prefix,
                            read_id,
                            cache.args.k)
        cache.pread = ProfiledRead(seq=seq,
                                   id=read_id,
                                   K=cache.args.k,
                                   counts=prof,
                                   states=None if cache.states is None else 'E' * (cache.args.k - 1) + cache.states[read_id])
        if cache.pread is None:
            raise PreventUpdate
        cache.prv = (ProfiledReadVisualizer()
                     .add_trace_counts(cache.pread,
                                       name=("Profile" if cache.args.fastk_prefix_hoco is None
                                             else "Normal")))
        if cache.args.fastk_prefix_hoco is not None:
            prof_hoco = pullback_hoco(fastk.profex(cache.args.fastk_prefix_hoco,
                                                   read_id,
                                                   cache.args.k),
                                      seq)
            cache.pread_hoco = ProfiledRead(seq=seq,
                                            id=read_id,
                                            K=cache.args.k,
                                            counts=prof_hoco)
            if cache.pread_hoco is None:
                raise PreventUpdate
            cache.prv.add_trace_counts(cache.pread_hoco,
                                       col=cache.args.color_hoco,
                                       name="Hoco")
        if cache.states is not None:
            cache.prv.add_trace_states(cache.pread,
                                       show_init="SHOW" in class_init)
        return (cache.prv.add_trace_bases(cache.pread)
                .show(layout=reset_axes(fig),
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
                  config=dict(toImageButtonOptions=dict(format=cache.args.download_as))),
        html.Div(["Read ID: ",
                  dcc.Input(id='read-id',
                            value='',
                            type='number')]),
        html.Div([html.Button(id='submit-profile',
                              n_clicks=0,
                              children='Draw k-mer count profile'),
                  dcc.Checklist(id='class-init',
                                options=[{'label': 'Show classifications from the beginning',
                                          'value': 'SHOW'}],
                                value=[])]),
        dcc.Graph(id='fig-profile',
                  figure=go.Figure(
                      layout=pl.make_layout(width=1800,
                                            height=500)),
                  config=dict(toImageButtonOptions=dict(format=cache.args.download_as)))
    ])
    app.run_server(port=int(cache.args.port_number),
                   debug=cache.args.debug_mode)


def parse_args() -> argparse.Namespace:
    global cache
    parser = argparse.ArgumentParser(
        description="Visualizations for k-mer analysis")
    parser.add_argument(
        "db_fname",
        type=str,
        help="DAZZ_DB file name.")
    parser.add_argument(
        "-k",
        type=int,
        default=40,
        help="The value of K for K-mers. [40]")
    parser.add_argument(
        "-f",
        "--fastk_prefix",
        type=Optional[str],
        default=None,
        help="Prefix of FastK outputs. [prefix of `db_fname`]")
    parser.add_argument(
        "-g",
        "--fastk_prefix_hoco",
        type=Optional[str],
        default=None,
        help="Prefix of FastK outputs for homopolymer compressed datasets. [None]")
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
        "-i",
        "--download_as",
        type=str,
        default="svg",
        help="File format of the image downloaed via the icon. [svg]")
    parser.add_argument(
        "-d",
        "--debug_mode",
        action="store_true",
        help="Run a Dash server in a debug mode.")
    cache.args = parser.parse_args()
    if cache.args.fastk_prefix is None:
        cache.args.fastk_prefix = splitext(cache.args.db_fname)[0]
    if cache.args.class_fname is not None:
        cache.states = {}
        with open(cache.args.class_fname, 'r') as f:
            for line in f:
                read_id, states = line.strip().split('\t')
                cache.states[int(read_id)] = states


if __name__ == '__main__':
    main()
