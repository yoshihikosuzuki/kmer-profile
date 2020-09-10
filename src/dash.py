import os
import argparse
from collections import Counter
from typing import Any, Optional, Tuple, List
from logzero import logger
import plotly.io as pio
import plotly.graph_objects as go
from plotly.basedatatypes import BaseTraceType
import plotly_light as pl
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State, ALL
from dash.exceptions import PreventUpdate
from BITS.util.proc import run_command
from .io import REPkmerResult, CountProfile, load_count_dist, load_kmer_profile

pio.templates.default = 'plotly_white'
app = dash.Dash(__name__,
                external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])

### ----------------------------------------------------------------------- ###
###                     constants and global variables                      ###
### ----------------------------------------------------------------------- ###

# Colors
THRESHOLD_COLS = {'error_haplo': "coral",
                  'haplo_diplo': "navy",
                  'diplo_repeat': "mediumvioletred"}
CLASS_COLS = {'E': "red",
              'H': "green",
              'D': "blue"}

# Cache for k-mer count frequencies/profiles
repkmer_result: Optional[REPkmerResult] = None
trace_hist_all: Optional[go.Bar] = None
trace_profile: Optional[go.Scatter] = None
bases_shown = False   # If True, need to remove bases on plot when relayouted


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
    global repkmer_result, trace_hist_all
    ctx = dash.callback_context
    max_count = int(max_count)
    if (not ctx.triggered
            or ctx.triggered[0]["prop_id"] == "submit-dist.n_clicks"):
        # Draw the global k-mer count distribution from all reads
        repkmer_result = load_count_dist(db_fname, max_count)
        if repkmer_result is None:
            raise PreventUpdate
        trace_hist_all = pl.make_hist(repkmer_result.count_rel_freqs,
                                      bin_size=1,
                                      col="gray",
                                      name="All reads",
                                      show_legend=True)
        threshold_lines = [pl.make_line(count, 0, count, 1,
                                        yref="paper",
                                        width=2,
                                        col=THRESHOLD_COLS[name],
                                        layer="above")
                           for name, count in repkmer_result.thresholds.items()]
        return go.Figure(data=trace_hist_all,
                         layout=(pl.make_layout(shapes=threshold_lines)
                                 .update(fig["layout"])))
    elif ctx.triggered[0]["prop_id"] == "submit-profile.n_clicks":
        read_id = int(read_id)
        profile = load_kmer_profile(db_fname, read_id)
        if profile is None:
            raise PreventUpdate
        trace_hist_read = pl.make_hist(profile.count_rel_freqs(max_count=max_count),
                                       bin_size=1,
                                       col="turquoise",
                                       opacity=0.7,
                                       name=f"Read {read_id}",
                                       show_legend=True)
        return go.Figure(data=[trace_hist_all, trace_hist_read],
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
    global trace_profile, bases_shown
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    if ctx.triggered[0]["prop_id"] == "submit-profile.n_clicks":
        # Draw a k-mer count profile from scratch
        read_id = int(read_id)
        profile = load_kmer_profile(db_fname, read_id)
        if profile is None:
            raise PreventUpdate
        bases_shown = False
        if not isinstance(max_count, int):
            max_count = max(profile.counts)
        trace_profile = pl.make_scatter(x=list(range(len(profile.counts))),
                                        y=profile.counts,
                                        mode="lines",
                                        col="black")
        threshold_lines = ([pl.make_line(0, count, 1, count,
                                         xref="paper",
                                         col=THRESHOLD_COLS[name],
                                         layer="below")
                            for name, count in repkmer_result.thresholds.items()]
                           if repkmer_result is not None else None)
        return go.Figure(data=trace_profile,
                         layout=(pl.make_layout().update(fig["layout"])
                                 .update(pl.make_layout(y_range=(0, max_count),
                                                        shapes=threshold_lines),
                                         overwrite=True)))
    elif ctx.triggered[0]["prop_id"] == "fig-profile.relayoutData":
        if len(fig["data"]) == 0:
            raise PreventUpdate
        xmin, xmax = map(int, fig["layout"]["xaxis"]["range"])
        xmin = max(0, xmin)
        xmax = min(len(profile.counts), xmax)
        if xmax - xmin < 300:
            # Show bases if the plotting region is shorter than 300 bp
            bases_shown = True
            trace_bases = pl.make_scatter(x=list(range(xmin, xmax)),
                                          y=profile.counts[xmin:xmax],
                                          text=profile.bases[xmin:xmax],
                                          text_pos="top center",
                                          mode="text")
            return go.Figure(data=[trace_bases, trace_profile],
                             layout=fig["layout"])
        elif bases_shown:
            # Remove the drawn bases if we left the desired plotting region
            bases_shown = False
            return go.Figure(data=trace_profile,
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
