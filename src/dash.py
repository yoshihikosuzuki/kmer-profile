import argparse
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
from bits.util import RelCounter
from .type import ProfiledRead
from .classifier.heuristics import run_heuristics
from .visualizer import CountDistVisualizer, ProfiledReadVisualizer

pio.templates.default = 'plotly_white'
app = dash.Dash(__name__,
                external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])


@dataclass(repr=False, eq=False)
class Cache:
    """Cache for data that are needed to be shared over multiple operations."""
    count_global: Optional[RelCounter] = None
    read: Optional[ProfiledRead] = None


cache = Cache()

K = 40   # TODO: change to an input

### ----------------------------------------------------------------------- ###
###                        k-mer count distribution                         ###
### ----------------------------------------------------------------------- ###


@app.callback(
    Output('fig-dist', 'figure'),
    [Input('submit-dist', 'n_clicks'),
     Input('submit-profile', 'n_clicks')],
    [State('db-fname', 'value'),
     State('read-id', 'value'),
     State('max-count-dist', 'value'),
     State('fig-dist', 'figure')]
)
def update_count_dist(n_clicks_dist: int,
                      n_clicks_profile: int,
                      db_fname: str,
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
        cache.count_global = load_count_dist(db_fname, max_count)
        if cache.count_global is None:
            raise PreventUpdate
        return gen_fig_count_dist(cache.count_global,
                                  "All reads",
                                  "gray",
                                  relative=True,
                                  layout=pl.merge_layout(
                                      fig["layout"] if "layout" in fig else None,
                                      pl.make_layout(x_range=None, y_range=None)))
    elif ctx.triggered[0]["prop_id"] == "submit-profile.n_clicks":
        read = load_kmer_profile(db_fname, int(read_id))
        if read is None:
            raise PreventUpdate
        return gen_fig_count_dist([cache.count_global,
                                   read.count_freqs(max_count)],
                                  ["All reads",
                                   f"Read {read.id}"],
                                  ["gray",
                                   "turquoise"],
                                  relative=True,
                                  layout=pl.merge_layout(
                                      fig["layout"] if "layout" in fig else None,
                                      pl.make_layout(x_range=None, y_range=None)))
    raise PreventUpdate


### ----------------------------------------------------------------------- ###
###                           k-mer count profile                           ###
### ----------------------------------------------------------------------- ###


@app.callback(
    Output('fig-profile', 'figure'),
    [Input('submit-profile', 'n_clicks'),
     Input('submit-classify', 'n_clicks')],
    [State('db-fname', 'value'),
     State('read-id', 'value'),
     State('fig-profile', 'figure')]
)
def update_kmer_profile(n_clicks_profile: int,
                        n_clicks_classify: int,
                        db_fname: str,
                        read_id: int,
                        fig: go.Figure) -> go.Figure:
    """Update the count profile plot."""
    global cache
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    if ctx.triggered[0]["prop_id"] == "submit-profile.n_clicks":
        # Draw a k-mer count profile from scratch
        cache.read = load_kmer_profile(db_fname, int(read_id))
        if cache.read is None:
            raise PreventUpdate
        return gen_fig_preads(cache.read, K,
                              layout=pl.merge_layout(
                                  fig["layout"] if "layout" in fig else None,
                                  pl.make_layout(x_range=None, y_range=None)))
    elif ctx.triggered[0]["prop_id"] == "submit-classify.n_clicks":
        if cache.read is None:
            raise PreventUpdate
        # run_heuristics(cache.read, K)
        return gen_fig_preads(cache.read, K,
                              layout=fig["layout"] if "layout" in fig else None)
    raise PreventUpdate


### ----------------------------------------------------------------------- ###
###                   Layout, Command-line arguments, etc.                  ###
### ----------------------------------------------------------------------- ###


def main():
    args = parse_args()
    app.layout = html.Div(children=[
        html.Div(["DB file: ",
                  dcc.Input(id='db-fname',
                            value=args.input_db,
                            type='text')]),
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
                  config=dict(toImageButtonOptions=dict(format=args.download_as))),
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
        dcc.Graph(id='fig-profile',
                  figure=go.Figure(
                      layout=pl.make_layout(width=1800,
                                            height=500)),
                  config=dict(toImageButtonOptions=dict(format=args.download_as)))
    ])
    app.run_server(debug=args.debug_mode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualizations for k-mer analysis")
    parser.add_argument(
        "-i",
        "--input_db",
        type=str,
        default="",
        help="Input DAZZ_DB file name.")
    parser.add_argument(
        "-f",
        "--download_as",
        type=str,
        default="svg",
        help="File format of the image downloaed via the icon.")
    parser.add_argument(
        "-d",
        "--debug_mode",
        action="store_true",
        help="Run a Dash server in a debug mode.")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
