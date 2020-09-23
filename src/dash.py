import argparse
from dataclasses import dataclass
from typing import Any, Optional
import plotly.io as pio
import plotly.graph_objects as go
import plotly_light as pl
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from .type import StateThresholds, RelCounter, ProfiledRead
from .io import load_count_dist, load_kmer_profile
from .visualizer import gen_traces_profile

pio.templates.default = 'plotly_white'
app = dash.Dash(__name__,
                external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])


@dataclass(repr=False, eq=False)
class Cache:
    """Cache for data that are needed to be shared over multiple operations."""
    th_global: Optional[StateThresholds] = None
    count_global: Optional[RelCounter] = None
    trace_hist_global: Optional[go.Bar] = None
    read: Optional[ProfiledRead] = None
    trace_profile: Optional[go.Scatter] = None
    bases_shown: bool = False   # If True, need to remove bases on plot when relayouted

### ----------------------------------------------------------------------- ###
###                     constants and global variables                      ###
### ----------------------------------------------------------------------- ###


THRESHOLD_COLS = {'error_haplo': "coral",
                  'haplo_diplo': "navy",
                  'diplo_repeat': "mediumvioletred"}
cache = Cache()

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
        ret = load_count_dist(db_fname, max_count)
        if ret is None:
            raise PreventUpdate
        cache.count_global, cache.th_global = ret
        cache.trace_hist_global = pl.make_hist(cache.count_global.relative(),
                                               bin_size=1,
                                               col="gray",
                                               name="All reads",
                                               show_legend=True)
        threshold_lines = [pl.make_line(count, 0, count, 1,
                                        yref="paper",
                                        width=2,
                                        col=THRESHOLD_COLS[name],
                                        layer="above")
                           for name, count in cache.th_global._asdict().items()]
        return go.Figure(data=cache.trace_hist_global,
                         layout=pl.merge_layout(pl.make_layout(shapes=threshold_lines),
                                                fig["layout"]))
    elif ctx.triggered[0]["prop_id"] == "submit-profile.n_clicks":
        read = load_kmer_profile(db_fname, int(read_id))
        if read is None:
            raise PreventUpdate
        trace_hist_read = pl.make_hist(read.count_freqs(max_count).relative(),
                                       bin_size=1,
                                       col="turquoise",
                                       opacity=0.7,
                                       name=f"Read {read_id}",
                                       show_legend=True)
        return go.Figure(data=[cache.trace_hist_global, trace_hist_read],
                         layout=fig["layout"])


### ----------------------------------------------------------------------- ###
###                           k-mer count profile                           ###
### ----------------------------------------------------------------------- ###


@app.callback(
    Output('fig-profile', 'figure'),
    [Input('submit-profile', 'n_clicks'),
     Input('fig-profile', 'relayoutData')],
    [State('db-fname', 'value'),
     State('read-id', 'value'),
     State('max-count-profile', 'value'),
     State('fig-profile', 'figure')]
)
def update_kmer_profile(n_clicks: int,
                        relayout_data: Any,
                        db_fname: str,
                        read_id: int,
                        max_count: int,
                        fig: go.Figure) -> go.Figure:
    """Update the count profile plot."""
    global cache
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    if ctx.triggered[0]["prop_id"] == "submit-profile.n_clicks":
        # Draw a k-mer count profile from scratch
        read = load_kmer_profile(db_fname, int(read_id))
        if read is None:
            raise PreventUpdate
        cache.read = read
        cache.bases_shown = False
        if not isinstance(max_count, int):
            max_count = max(cache.read.counts)
        cache.trace_profile = gen_traces_profile(cache.read.counts)[0]
        threshold_lines = ([pl.make_line(0, count, 1, count,
                                         xref="paper",
                                         col=THRESHOLD_COLS[name],
                                         layer="below")
                            for name, count in cache.th_global._asdict().items()]
                           if cache.th_global is not None else None)
        return go.Figure(data=cache.trace_profile,
                         layout=pl.merge_layout(pl.make_layout(y_range=(0, max_count),
                                                               shapes=threshold_lines),
                                                fig["layout"]))
    elif ctx.triggered[0]["prop_id"] == "fig-profile.relayoutData":
        if len(fig["data"]) == 0:
            raise PreventUpdate
        xmin, xmax = map(int, fig["layout"]["xaxis"]["range"])
        xmin = max(0, xmin)
        xmax = min(cache.read.length, xmax)
        if xmax - xmin < 300:
            # Show bases if the plotting region is shorter than 300 bp
            cache.bases_shown = True
            print(list(cache.read.seq[xmin:xmax]))
            trace_bases = pl.make_scatter(x=list(range(xmin, xmax)),
                                          y=cache.read.counts[xmin:xmax],
                                          text=list(cache.read.seq[xmin:xmax]),
                                          text_pos="top center",
                                          mode="text")
            return go.Figure(data=[trace_bases, cache.trace_profile],
                             layout=fig["layout"])
        elif cache.bases_shown:
            # Remove the drawn bases if we left the desired plotting region
            cache.bases_shown = False
            return go.Figure(data=cache.trace_profile,
                             layout=fig["layout"])
        else:
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
                                            height=400,
                                            x_title="K-mer count",
                                            y_title="Relative frequency [%]",
                                            barmode="overlay")),
                  config=dict(toImageButtonOptions=dict(format=args.download_as))),
        html.Div(["Read ID: ",
                  dcc.Input(id='read-id',
                            value='',
                            type='number')]),
        html.Div([html.Button(id='submit-profile',
                              n_clicks=0,
                              children='Draw k-mer count profile'),
                  " [OPTIONS]",
                  " Max count = ",
                  dcc.Input(id='max-count-profile',
                            value='',
                            type='number')]),
        dcc.Graph(id='fig-profile',
                  figure=go.Figure(
                      layout=pl.make_layout(width=1800,
                                            height=500,
                                            x_title="Position",
                                            y_title="Count",
                                            y_grid=False)),
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
