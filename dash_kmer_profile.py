import os
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
app.layout = html.Div(children=[
    dcc.Graph(id='kmer-profile-graph'),
    html.Div(["DB file name: ",
              dcc.Input(id='db-fname',
                        value='',
                        type='text')]),
    html.Br(),
    html.Div(["Read ID: ",
              dcc.Input(id='read-id',
                        value='1',
                        type='number')]),
    html.Br(),
    html.Button(id='submit-button-state',
                n_clicks=0,
                children='Draw k-mer profile'),
])


@app.callback(
    Output('kmer-profile-graph', 'figure'),
    [Input('submit-button-state', 'n_clicks')],
    [State('db-fname', 'value'),
     State('read-id', 'value')]
)
def update_kmer_profile(n_clicks: int,
                        db_fname: str,
                        read_id: str) -> go.Figure:
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


if __name__ == '__main__':
    app.run_server(debug=True)
