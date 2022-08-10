import argparse
from os import getcwd
from os.path import isfile
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, List, Tuple
import plotly.graph_objects as go
import plotly_light as pl
from flask import Flask, send_from_directory
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from bits.seq import load_fastq
from bits.util import RelCounter
import fastk
from . import ProfiledRead, CountHistVisualizer, PreadVisualizer, load_pread


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
    """Cached data shared across multiple operations.

    Variables:
        - args  <argparse.Namespace>           : Command line arguments.
        - has_class <bool>: .class file is provided or not.
        - has_tclass <bool>: Ground-truth .class file is provided or not.
        - cdv   <Optional[CountHistVisualizer]>: Count histogram plot object.
        - pread <Optional[ProfiledRead])>      : Count profile object.
        - prv   <Optional[PreadVisualizer]>    : Count profile plot object.
        - tpead <Optional[ProfiledRead]>       : Ground-truth count profile object.
        - tprv  <Optional[PreadVisualizer]>    : Ground-truth count profile plot object.
    """
    args:       argparse.Namespace = None
    has_class:  bool = False
    has_tclass: bool = False
    cdv:        Optional[CountHistVisualizer] = None
    pread:      Optional[ProfiledRead] = None
    prv:        Optional[PreadVisualizer] = None
    tpread:     Optional[ProfiledRead] = None
    tprv:       Optional[PreadVisualizer] = None


cache = Cache()


def reset_axes(fig: go.Figure) -> go.Layout:
    """
    Return the layout of the figure with reset axes.

    Args:
        fig (go.Figure): _description_

    Returns:
        go.Layout: _description_
    """
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
        cache.cdv = CountHistVisualizer(relative=True, show_legend=True)
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
                                    col="deepskyblue",
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
                                 cache.args.class_fname)
        if cache.pread is None:
            raise PreventUpdate
        cache.pread.states = (load_fastq(cache.args.class_fname, read_id).qual[cache.pread.K - 1:] if cache.has_class
                              else None)
        cache.prv = (PreadVisualizer(cache.pread, max_count=max_count, use_webgl=True)
                     .add_counts()
                     .add_states(show_init="SHOW" in class_init))
        new_fig = cache.prv.show(layout=reset_axes(fig),
                                 return_fig=True)
        pl.show(new_fig, out_html="kmer_prof.html", do_not_display=True)

        # True profile
        if cache.has_tclass:
            cache.tpread = deepcopy(cache.pread)
            cache.tpread.states = load_fastq(cache.args.truth_class_fname, read_id).qual[cache.pread.K - 1:]
            cache.tprv = (PreadVisualizer(cache.tpread, max_count=max_count, use_webgl=True)
                          .add_counts()
                          .add_states(show_init="SHOW" in class_init))
            new_tfig = cache.tprv.show(layout=reset_axes(tfig),
                                       return_fig=True)
        else:
            new_tfig = dash.no_update

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
    graph_config = dict(showTips=False,
                        displaylogo=False,
                        toImageButtonOptions=dict(format=cache.args.img_format))
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
                  figure=go.Figure(layout=pl.layout(title="K-mer count histogram",
                                                    width=800, height=400)),
                  config=dict(showTips=False,
                              displaylogo=False,
                              toImageButtonOptions=dict(format=cache.args.img_format))),
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
                                value=(["SHOW"] if cache.has_class or cache.has_tclass else []))]),
        dcc.Graph(id='fig-profile',
                  figure=go.Figure(layout=pl.layout(title="Read profile",
                                                    width=1800, height=500)),
                  config=graph_config,
                  style={"display": "none"} if not cache.has_class and cache.has_tclass else {}),
        dcc.Graph(id='fig-true-profile',
                  figure=go.Figure(layout=pl.layout(title="Ground Truth",
                                                    width=1800, height=500)),
                  config=graph_config,
                  style={"display": "none"} if not cache.has_tclass else {})
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
        "-c",
        "--class_fname",
        type=str,
        default=None,
        help="Either a k-mer classification file by ClassPro (.class) or a sequence file (.db, .dam, or .fast[a|q]). [None]")
    parser.add_argument(
        "-t",
        "--truth_class_fname",
        type=str,
        default=None,
        help="A ground-trugh k-mer classification file. [None]")
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
                  args.class_fname,
                  args.truth_class_fname]:
        assert fname is None or isfile(fname), f"{fname} does not exist"
    if args.class_fname is not None and args.class_fname.endswith(".class"):
        cache.has_class = True
    if args.truth_class_fname is not None:
        cache.has_tclass = True


if __name__ == '__main__':
    main()
