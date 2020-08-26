import os
import argparse
from collections import Counter
from typing import List
import plotly_light as pl
import plotly.graph_objects as go
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import dash
import plotly.io as pio
from BITS.util.proc import run_command


pio.templates.default = 'plotly_white'
app = dash.Dash(__name__,
                external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])


### ----------------------------------------------------------------------- ###
###                        k-mer count distribution                         ###
### ----------------------------------------------------------------------- ###


def load_count_dist(db_fname: str,
                    max_count: int) -> Counter:
    """Load global k-mer count frequencies in the database."""
    if not os.path.exists(db_fname):
        return Counter()
    count_freqs = Counter()
    command = f"REPkmer -x{max_count} {db_fname}"
    lines = run_command(command).strip().split('\n')
    for i, line in enumerate(lines):
        if line.strip() == "K-mer Histogram":
            lines = lines[i:]
            break
    for line in lines:
        data = line.strip().split()
        if len(data) != 3:
            continue
        kmer_count, freq, percent = data
        kmer_count = kmer_count[:-1]
        if kmer_count[-1] == '+':
            kmer_count = kmer_count[:-1]
        count_freqs[int(kmer_count)] = int(freq)
    return count_freqs


@app.callback(
    Output('kmer-count-dist', 'figure'),
    [Input('submit-count-freq', 'n_clicks')],
    [State('db-fname', 'value'),
     State('max-count', 'value')]
)
def update_count_dist(n_clicks: int,
                      db_fname: str,
                      max_count: int) -> go.Figure:
    """Update the count distribution."""
    if n_clicks == 0:
        return go.Figure()
    max_count = int(max_count)
    count_freqs = load_count_dist(db_fname, max_count)
    return go.Figure(data=pl.make_hist(count_freqs,
                                       bin_size=1),
                     layout=pl.make_layout(x_title="K-mer count",
                                           y_title="Frequency"))


### ----------------------------------------------------------------------- ###
###                           k-mer count profile                           ###
### ----------------------------------------------------------------------- ###


def load_kmer_profile(db_fname: str,
                      read_id: int) -> List[int]:
    """Load k-mer count profile of a single read."""
    if not os.path.exists(db_fname):
        return []
    counts = []
    command = f"KMlook {db_fname} {read_id}"
    for line in run_command(command).strip().split('\n')[3:]:
        data = line.strip().split()
        if len(data) != 4:
            continue
        pos, base, count, base_type = data
        counts.append(count)
    return counts


@app.callback(
    Output('kmer-profile-graph', 'figure'),
    [Input('submit-kmer-profile', 'n_clicks')],
    [State('db-fname', 'value'),
     State('read-id', 'value')]
)
def update_kmer_profile(n_clicks: int,
                        db_fname: str,
                        read_id: str) -> go.Figure:
    """Update the count profile plot."""
    if n_clicks == 0:
        return go.Figure()
    read_id = int(read_id)
    counts = load_kmer_profile(db_fname, read_id)
    return go.Figure(data=pl.make_scatter(x=list(range(len(counts))),
                                          y=counts,
                                          mode="lines",
                                          col="black"),
                     layout=pl.make_layout(x_title="Position",
                                           y_title="Count",
                                           x_grid=False,
                                           y_grid=False))


### ----------------------------------------------------------------------- ###
###                           Layout, Options, etc.                         ###
### ----------------------------------------------------------------------- ###


def main():
    args = parse_args()
    app.layout = html.Div(children=[
        html.Div(["DB file: ",
                  dcc.Input(id='db-fname',
                            value=args.input_db,
                            type='text')]),
        html.Div(["Max count [for k-mer count distribution]: ",
                  dcc.Input(id='max-count',
                            value='1000',
                            type='number')]),
        html.Button(id='submit-count-freq',
                    n_clicks=0,
                    children='Draw k-mer count distribution'),
        dcc.Graph(id='kmer-count-dist',
                  config=dict(toImageButtonOptions=dict(format=args.download_as))),
        html.Div(["Read ID: ",
                  dcc.Input(id='read-id',
                            value='',
                            type='number')]),
        html.Button(id='submit-kmer-profile',
                    n_clicks=0,
                    children='Draw k-mer profile'),
        dcc.Graph(id='kmer-profile-graph',
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
