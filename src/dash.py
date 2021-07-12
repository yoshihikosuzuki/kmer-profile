import argparse
from os import getcwd
from os.path import isfile
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from logzero import logger
import plotly.graph_objects as go
import plotly_light as pl
from flask import Flask, send_from_directory
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from bits.seq import load_fastq
from bits.util import RelCounter
import fastk
from .type import ProfiledRead
from .classifier import load_pread
from .visualizer import CountDistVisualizer, ProfiledReadVisualizer

server = Flask(__name__)
app = dash.Dash(__name__,
                external_stylesheets=[
                    'https://codepen.io/chriddyp/pen/bWLwgP.css'],
                server=server)


@server.route("/download/<path:fname>")
def download_file(fname: str):
    """Called when `https://localhost:<port_number>/download/<fname>` is accessed.
    Download a file named `fname` in the running directory."""
    return send_from_directory(getcwd(), fname, as_attachment=True)


def build_download_button(fname: str, name: str) -> html.Form:
    """Generate a button for downloading a file named `fname`."""
    return html.Form(action=f"download/{fname}",
                     method="get",
                     children=[html.Button(children=[name])])


@dataclass(repr=False, eq=False)
class Cache:
    """Cache for data that are needed to be shared over multiple operations."""
    args: argparse.Namespace = None
    cdv: Optional[CountDistVisualizer] = None
    pread: Optional[ProfiledRead] = None
    prv: Optional[ProfiledReadVisualizer] = None
    tpread: Optional[ProfiledRead] = None
    tprv: Optional[ProfiledReadVisualizer] = None


cache = Cache()


def reset_axes(fig: go.Figure) -> go.Layout:
    """Return the layout of the figure with reset axes."""
    return pl.merge_layout(fig["layout"] if "layout" in fig else None,
                           pl.make_layout(x_range=None, y_range=None))

### ----------------------------------------------------------------------- ###
###                        k-mer count distribution                         ###
### ----------------------------------------------------------------------- ###


@app.callback(
    [Output('fig-dist', 'figure'),
     Output('download-dist', 'children')],
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
                      fig: go.Figure) -> Tuple[go.Figure, html.Form]:
    """Update the aggregated k-mer count distribution."""
    def _show_fig(_cdv, out_fname: str = "kmer_hist.html") -> Tuple[go.Figure, html.Form]:
        new_fig = _cdv.show(layout=reset_axes(fig),
                            return_fig=True)
        pl.show(new_fig, out_html=out_fname, do_not_display=True)
        return [new_fig,
                build_download_button(out_fname, "Download HTML")]

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
        return _show_fig(cache.cdv)
    elif ctx.triggered[0]["prop_id"] == "submit-profile.n_clicks":
        # Single-read k-mer count distribution
        if not read_id:
            raise PreventUpdate
        read_id = int(read_id)
        prof = fastk.profex(cache.args.fastk_prefix, read_id)
        return _show_fig(deepcopy(cache.cdv)
                         .add_trace(RelCounter([min(c, max_count) for c in prof]),
                                    col="turquoise",
                                    opacity=0.7,
                                    name=f"Read {read_id}"))
    raise PreventUpdate


### ----------------------------------------------------------------------- ###
###                           k-mer count profile                           ###
### ----------------------------------------------------------------------- ###


@app.callback(
    [Output('fig-profile', 'figure'),
     Output('fig-true-profile', 'figure'),
     Output('download-profile', 'children')],
    [Input('submit-profile', 'n_clicks')],
    [State('read-id', 'value'),
     State('max-count-prof', 'value'),
     State('class-init', 'value'),
     State('fig-profile', 'figure'),
     State('fig-true-profile', 'figure')]
)
def update_kmer_profile(_n_clicks_profile: int,
                        read_id: Optional[str],
                        max_count: Optional[str],
                        class_init: List[str],
                        fig: go.Figure,
                        tfig: go.Figure) -> Tuple[go.Figure, go.Figure, html.Form]:
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
        cache.pread.states = (None if cache.args.class_fname is None
                              else load_fastq(cache.args.class_fname, read_id)[0].qual)
        if cache.pread is None:
            raise PreventUpdate

        cache.prv = (ProfiledReadVisualizer(max_count=max_count)
                     .add_pread(cache.pread, show_init_states="SHOW" in class_init))
        new_fig = cache.prv.show(layout=reset_axes(fig),
                                 return_fig=True)

        # True profile
        if cache.args.truth_class_fname is not None:
            cache.tpread = deepcopy(cache.pread)
            cache.tpread.states = load_fastq(cache.args.truth_class_fname, read_id)[0].qual
            cache.tprv = (ProfiledReadVisualizer(max_count=max_count)
                          .add_pread(cache.tpread, show_init_states="SHOW" in class_init))
            new_tfig = cache.tprv.show(layout=reset_axes(tfig),
                                       return_fig=True)

        pl.show(new_fig, out_html="kmer_prof.html", do_not_display=True)
        return [new_fig,
                new_tfig,
                build_download_button("kmer_prof.html", "Download HTML")]
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
                              children='Draw histogram',
                              style={"float": "left"}),
                  html.Div(id="download-dist",
                           children=[],
                           style={"float": "left"}),
                  "   [OPTIONS] Max count = ",
                  dcc.Input(id='max-count-dist',
                            value='100',
                            type='number')]),
        dcc.Graph(id='fig-dist',
                  figure=go.Figure(
                      layout=pl.make_layout(width=800,
                                            height=400)),
                  config=dict(toImageButtonOptions=dict(format=cache.args.img_format))),
        html.Div(["Read ID: ",
                  dcc.Input(id='read-id',
                            value='',
                            type='number')]),
        html.Div([html.Button(id='submit-profile',
                              n_clicks=0,
                              children='Draw profile',
                              style={"float": "left"}),
                  html.Div(id="download-profile",
                           children=[],
                           style={"float": "left"}),
                  " [OPTIONS]",
                  " Max count = ",
                  dcc.Input(id='max-count-prof',
                            value='',
                            type='number'),
                  dcc.Checklist(id='class-init',
                                options=[{'label': 'Show classifications from the beginning',
                                          'value': 'SHOW'}],
                                value=(["SHOW"] if cache.args.class_fname is not None else []))]),
        dcc.Graph(id='fig-profile',
                  figure=go.Figure(
                      layout=pl.make_layout(width=1800,
                                            height=500)),
                  config=dict(toImageButtonOptions=dict(format=cache.args.img_format))),
        (dcc.Graph(id='fig-true-profile',
                  figure=go.Figure(
                      layout=pl.make_layout(width=1800,
                                            height=500)),
                  config=dict(toImageButtonOptions=dict(format=cache.args.img_format)))
         if cache.args.truth_class_fname is not None else None)
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
        help="Prefix of FastK's output files. Both `<fastk_prefix>.hist` and `<fastk_prefix>.prof` must exist.")
    parser.add_argument(
        "-s",
        "--seq_fname",
        type=str,
        default=None,
        help="Name of the input file for FastK containing reads. Must be .db/dam/fast[a|q]. Used for displaying baes in profile plot [None]")
    parser.add_argument(
        "-c",
        "--class_fname",
        type=str,
        default=None,
        help="File name of K-mer classification result. [None]")
    parser.add_argument(
        "-t",
        "--truth_class_fname",
        type=str,
        default=None,
        help="File name of ground truth of K-mer classification result. [None]")
    parser.add_argument(
        "-p",
        "--port_number",
        type=int,
        default=8050,
        help="Port number of localhost to run the server. [8050]")
    parser.add_argument(
        "-f",
        "--img_format",
        type=str,
        default="svg",
        help="Format of plot images you can download with camera icon. ['svg']")
    parser.add_argument(
        "-d",
        "--debug_mode",
        action="store_true",
        help="Run a Dash server in a debug mode.")
    cache.args = args = parser.parse_args()
    for fname in [f"{args.fastk_prefix}.hist",
                  f"{args.fastk_prefix}.prof",
                  args.seq_fname,
                  args.class_fname,
                  args.truth_class_fname]:
        assert fname is None or isfile(fname), f"{fname} does not exist"


if __name__ == '__main__':
    main()
