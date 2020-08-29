import os
import argparse
from collections import Counter
from typing import Any, Tuple, List
import plotly.io as pio
import plotly.graph_objects as go
import plotly_light as pl
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State, ALL
from dash.exceptions import PreventUpdate
from BITS.util.proc import run_command


pio.templates.default = 'plotly_white'
app = dash.Dash(__name__,
                external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])


### ----------------------------------------------------------------------- ###
###                        k-mer count distribution                         ###
### ----------------------------------------------------------------------- ###


# Classification thresholds
ERROR_HAPLO, HAPLO_DIPLO, DIPLO_REPEAT = None, None, None


def load_count_dist(db_fname: str,
                    max_count: int) -> Counter:
    """Load global k-mer count frequencies in the database."""
    global ERROR_HAPLO, HAPLO_DIPLO, DIPLO_REPEAT
    count_freqs = Counter()
    if not os.path.exists(db_fname):
        return count_freqs
    command = f"REPkmer -x{max_count} {db_fname}"
    lines = run_command(command).strip().split('\n')
    for i, line in enumerate(lines):
        if line.strip().startswith("Error"):
            # Load thresholds for classification
            _, _, eh, _, _, _, hd, _, _, _, dr, _, _ = line.strip().split()
            ERROR_HAPLO = int(eh)
            HAPLO_DIPLO = int(hd)
            DIPLO_REPEAT = int(dr)
        if line.strip() == "K-mer Histogram":
            lines = lines[i:]
            break
    for line in lines:
        data = line.strip().split()
        if len(data) != 3:
            continue
        kmer_count, freq, _ = data
        kmer_count = kmer_count[:-1]
        if kmer_count[-1] == '+':
            kmer_count = kmer_count[:-1]
        count_freqs[int(kmer_count)] = int(freq)
    return count_freqs


# Global cache for global k-mer count frequencies
global_dist_trace = None


@app.callback(
    Output('kmer-count-dist', 'figure'),
    [Input('submit-count-freq', 'n_clicks'),
     Input('submit-kmer-profile', 'n_clicks')],
    [State('db-fname', 'value'),
     State('read-id', 'value'),
     State('max-count-dist', 'value'),
     State('kmer-count-dist', 'figure')]
)
def update_count_dist(n_clicks_dist: int,
                      n_clicks_profile: int,
                      db_fname: str,
                      read_id: int,
                      max_count: int,
                      fig: go.Figure) -> go.Figure:
    """Update the count distribution."""
    global global_dist_trace
    ctx = dash.callback_context
    max_count = int(max_count)
    if (not ctx.triggered
            or ctx.triggered[0]["prop_id"] == "submit-count-freq.n_clicks"):
        count_freqs = load_count_dist(db_fname, max_count)
        tot_freq = sum(list(count_freqs.values()))
        count_freqs = {k: v / tot_freq * 100 for k, v in count_freqs.items()}
        global_dist_trace = pl.make_hist(count_freqs,
                                         bin_size=1,
                                         col="gray",
                                         name="All k-mers",
                                         show_legend=True)
        threshold_lines = [pl.make_line(ERROR_HAPLO, 0, ERROR_HAPLO, 1,
                                        yref="paper",
                                        width=3,
                                        col="coral",
                                        layer="above"),
                           pl.make_line(HAPLO_DIPLO, 0, HAPLO_DIPLO, 1,
                                        yref="paper",
                                        width=3,
                                        col="navy",
                                        layer="above"),
                           pl.make_line(DIPLO_REPEAT, 0, DIPLO_REPEAT, 1,
                                        yref="paper",
                                        width=3,
                                        col="mediumvioletred",
                                        layer="above")]
        layout = pl.make_layout(width=800,
                                height=400,
                                x_title="K-mer count",
                                y_title="Relative frequency [%]",
                                shapes=threshold_lines)
        layout["barmode"] = "overlay"
        return go.Figure(data=global_dist_trace,
                         layout=layout)
    elif ctx.triggered[0]["prop_id"] == "submit-kmer-profile.n_clicks":
        read_id = int(read_id)
        # TODO: reuse count data loaded for profile?
        _, counts = load_kmer_profile(db_fname, read_id)
        counts = list(filter(lambda x: x > 0, counts))
        count_freqs = Counter([min(count, max_count) for count in counts])
        tot_freq = sum(list(count_freqs.values()))
        count_freqs = {k: v / tot_freq * 100 for k, v in count_freqs.items()}
        return go.Figure(data=[global_dist_trace,
                               pl.make_hist(count_freqs,
                                            bin_size=1,
                                            col="turquoise",
                                            opacity=0.7,
                                            name=f"Read {read_id}",
                                            show_legend=True)],
                         layout=fig["layout"])


### ----------------------------------------------------------------------- ###
###                           k-mer count profile                           ###
### ----------------------------------------------------------------------- ###


def load_kmer_profile(db_fname: str,
                      read_id: int) -> Tuple[List[str], List[int]]:
    """Load k-mer count profile of a single read."""
    if not os.path.exists(db_fname):
        return [], []
    command = f"KMlook {db_fname} {read_id}"
    lines = run_command(command).strip().split('\n')
    assert lines[2].startswith('len'), "Invalid KMlook output"
    read_length = int(lines[2].strip().split()[2])
    bases, counts = [''] * read_length, [0] * read_length
    for line in lines[3:]:
        data = line.strip().split()
        if data[0][-1] == ':':
            if data[2] == 'Z':
                pos, base = data[:2]
                count = 0
            else:
                pos, base, count = data[:3]
            pos = int(pos[:-1])
            bases[pos] = base
            counts[pos] = int(count)
    return bases, counts


# Global cache for k-mer profile plot
bases, counts = [], []
bases_shown = False   # to judge if we should remove labels when relayouted


@app.callback(
    Output('kmer-profile-graph', 'figure'),
    [Input('submit-kmer-profile', 'n_clicks'),
     Input('kmer-profile-graph', 'relayoutData')],
    [State('db-fname', 'value'),
     State('read-id', 'value'),
     State('max-count-profile', 'value'),
     State('kmer-profile-graph', 'figure')]
)
def update_kmer_profile(n_clicks: int,
                        relayout_data: Any,
                        db_fname: str,
                        read_id: int,
                        max_count: int,
                        fig: go.Figure) -> go.Figure:
    """Update the count profile plot."""
    global bases, counts, bases_shown
    ctx = dash.callback_context
    if not ctx.triggered:
        return go.Figure(layout=pl.make_layout(x_title="Position",
                                               y_title="Count"))
    if ctx.triggered[0]["prop_id"] == "submit-kmer-profile.n_clicks":
        # Draw a new plot from scratch
        read_id = int(read_id)
        bases, counts = load_kmer_profile(db_fname, read_id)
        bases_shown = False
        if not isinstance(max_count, int):
            max_count = max(counts)
        threshold_lines = [pl.make_line(0, ERROR_HAPLO, 1, ERROR_HAPLO,
                                        xref="paper",
                                        col="coral",
                                        layer="below"),
                           pl.make_line(0, HAPLO_DIPLO, 1, HAPLO_DIPLO,
                                        xref="paper",
                                        col="navy",
                                        layer="below"),
                           pl.make_line(0, DIPLO_REPEAT, 1, DIPLO_REPEAT,
                                        xref="paper",
                                        col="mediumvioletred",
                                        layer="below")]
        return go.Figure(data=pl.make_scatter(x=list(range(len(counts))),
                                              y=counts,
                                              mode="lines",
                                              col="black"),
                         layout=pl.make_layout(width=1600,
                                               height=400,
                                               x_title="Position",
                                               y_title="Count",
                                               y_range=(0, max_count),
                                               y_grid=False,
                                               shapes=threshold_lines))
    elif ctx.triggered[0]["prop_id"] == "kmer-profile-graph.relayoutData":
        if fig is None or len(fig["data"]) == 0:
            raise PreventUpdate
        xmin, xmax = map(int, fig["layout"]["xaxis"]["range"])
        xmin = max(0, xmin)
        xmax = min(len(counts), xmax)
        if xmax - xmin < 300:
            # Show bases if the plotting region is shorter than 300 bp
            bases_shown = True
            return go.Figure(data=([pl.make_scatter(x=list(range(xmin, xmax)),
                                                    y=counts[xmin:xmax],
                                                    text=bases[xmin:xmax],
                                                    text_pos="top center",
                                                    mode="text")]
                                   + fig["data"]),
                             layout=fig["layout"])
        elif bases_shown:
            # Remove the drawn bases if we left the desired plotting region
            bases_shown = False
            return go.Figure(data=pl.make_scatter(x=list(range(len(counts))),
                                                  y=counts,
                                                  mode="lines",
                                                  col="black"),
                             layout=fig["layout"])
        else:
            raise PreventUpdate


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
        html.Div([html.Button(id='submit-count-freq',
                              n_clicks=0,
                              children='Draw k-mer count distribution'),
                  " [OPTIONS]",
                  " Max count = ",
                  dcc.Input(id='max-count-dist',
                            value='100',
                            type='number')]),
        dcc.Graph(id='kmer-count-dist',
                  config=dict(toImageButtonOptions=dict(format=args.download_as))),
        html.Div(["Read ID: ",
                  dcc.Input(id='read-id',
                            value='',
                            type='number')]),
        html.Div([html.Button(id='submit-kmer-profile',
                              n_clicks=0,
                              children='Draw k-mer profile'),
                  " [OPTIONS]",
                  " Max count = ",
                  dcc.Input(id='max-count-profile',
                            value='',
                            type='number')]),

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
