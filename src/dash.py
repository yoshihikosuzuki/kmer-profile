import argparse
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional
import plotly.io as pio
import plotly.graph_objects as go
import plotly_light as pl
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from .type import ProfiledRead
from .io import load_histex, load_pread
from .classifier.heuristics import run_heuristics
from .visualizer import CountDistVisualizer, ProfiledReadVisualizer

pio.templates.default = 'plotly_white'
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
        # Draw the global k-mer count distribution from all reads
        count_global = load_histex(cache.args.fastk_prefix,
                                   max_count=max_count)
        count_global_hoco = load_histex(cache.args.fastk_prefix_hoco,
                                        max_count=max_count)
        if count_global is None or count_global_hoco is None:
            raise PreventUpdate
        cache.cdv = (CountDistVisualizer(relative=True)
                     .add_trace(count_global,
                                col="gray",
                                opacity=1,
                                name="Global (normal)")
                     .add_trace(count_global_hoco,
                                col=cache.args.color_hoco,
                                opacity=0.7,
                                name="Global (hoco)"))
        return cache.cdv.show(layout=reset_axes(fig),
                              return_fig=True)
    elif ctx.triggered[0]["prop_id"] == "submit-profile.n_clicks":
        read_id = int(read_id)
        pread = load_pread(cache.args.db_fname,
                           cache.args.fastk_prefix,
                           read_id,
                           cache.args.k)
        if pread is None:
            raise PreventUpdate
        return (deepcopy(cache.cdv)
                .add_trace(pread.count_freqs(max_count),
                           col="turquoise",
                           opacity=0.7,
                           name=f"Read {read_id}")
                .show(layout=reset_axes(fig),
                      return_fig=True))
    raise PreventUpdate


### ----------------------------------------------------------------------- ###
###                           k-mer count profile                           ###
### ----------------------------------------------------------------------- ###


@ app.callback(
    Output('fig-profile', 'figure'),
    [Input('submit-profile', 'n_clicks'),
     Input('submit-classify', 'n_clicks')],
    [State('read-id', 'value'),
     State('fig-profile', 'figure')]
)
def update_kmer_profile(n_clicks_profile: int,
                        n_clicks_classify: int,
                        read_id: int,
                        fig: go.Figure) -> go.Figure:
    """Update the count profile plot."""
    global cache
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    if ctx.triggered[0]["prop_id"] == "submit-profile.n_clicks":
        # Draw a k-mer count profile from scratch
        cache.pread = load_pread(cache.args.db_fname,
                                 cache.args.fastk_prefix,
                                 int(read_id),
                                 cache.args.k,
                                 cache.args.fastk_prefix_hoco)
        if cache.pread is None:
            raise PreventUpdate
        if cache.args.fastk_prefix_hoco is not None:
            cache.preead, cache.pread_hoco = cache.pread
            if cache.pread is None or cache.pread_hoco is None:
                raise PreventUpdate
        cache.prv = (ProfiledReadVisualizer()
                     .add_trace_counts(cache.pread,
                                       name="Normal"))
        if cache.args.fastk_prefix_hoco is not None:
            cache.prv.add_trace_counts(cache.pread_hoco,
                                       col=cache.args.color_hoco,
                                       name="Hoco") \
                .add_trace_bases(cache.pread_hoco)
        return cache.prv.show(layout=reset_axes(fig),
                              return_fig=True)
    elif ctx.triggered[0]["prop_id"] == "submit-classify.n_clicks":
        if cache.prv is None:
            raise PreventUpdate
        # run_heuristics(cache.read, K)
        return (deepcopy(cache.prv)
                .add_trace_states(cache.pread)
                .show(layout=reset_axes(fig),
                      return_fig=True))
    raise PreventUpdate


### ----------------------------------------------------------------------- ###
###                   Layout, Command-line arguments, etc.                  ###
### ----------------------------------------------------------------------- ###


def main():
    # TODO: check availavility of commands required afterwards
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
                              children='Draw k-mer count profile')]),
        html.Div([html.Button(id='submit-classify',
                              n_clicks=0,
                              children='Classify k-mers')]),
        # TODO: "download_html" button?
        dcc.Graph(id='fig-profile',
                  figure=go.Figure(
                      layout=pl.make_layout(width=1800,
                                            height=500)),
                  config=dict(toImageButtonOptions=dict(format=cache.args.download_as)))
    ])
    app.run_server(debug=cache.args.debug_mode)


def parse_args() -> argparse.Namespace:
    global cache
    parser = argparse.ArgumentParser(
        description="Visualizations for k-mer analysis")
    parser.add_argument(
        "db_fname",
        type=str,
        help="DAZZ_DB file name.")
    parser.add_argument(
        "fastk_prefix",
        type=str,
        help="Prefix of the FastK output files.")
    parser.add_argument(
        "fastk_prefix_hoco",
        nargs='?',
        type=Optional[str],
        default=None,
        help="Prefix of the FastK output files for HoCo profiles.")
    parser.add_argument(
        "-k",
        type=int,
        default=40,
        help="The value of K for K-mers.")
    parser.add_argument(
        "-f",
        "--download_as",
        type=str,
        default="svg",
        help="File format of the image downloaed via the icon.")
    parser.add_argument(
        "-c",
        "--color_hoco",
        type=str,
        default="darkorange",
        help="Color for hoco plots.")
    parser.add_argument(
        "-d",
        "--debug_mode",
        action="store_true",
        help="Run a Dash server in a debug mode.")
    cache.args = parser.parse_args()


if __name__ == '__main__':
    main()
