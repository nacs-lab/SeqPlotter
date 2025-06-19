# Import packages
import dash
from dash import Dash, html, dcc, callback, Input, Output, State, Patch, ALL, MATCH, ctx, no_update, clientside_callback, ClientsideFunction
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import struct
import plotly.express as px
import base64
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
import json
from pathlib import Path

# Initialize the app
app = Dash(title='Seq Plotter', prevent_initial_callbacks="initial_duplicate", external_stylesheets=[dbc.themes.BOOTSTRAP], use_pages=True)

app.layout = dbc.Container([
    dbc.Row([html.Div('Sequence Plotter', className='text-center display-3 fw-bold mb-4')]),
    dbc.Row([dbc.Col([dbc.Nav([dbc.NavItem(dbc.NavLink("Home", active='exact', href="/")),
             dbc.NavItem(dbc.NavLink("Help", active='exact', href="/help")),
    ], pills=True, fill=True, justified=True)], width={'size': 4, 'offset': 4})]),
    html.Hr(className="my-4 mx-auto w-75 border border-2 border-secondary"),
    html.Div(id ='uuid-display', children='', className='text-center display-6 mb-2'),
    dash.page_container,
    dcc.Interval(id='get-uuid', interval=1, n_intervals=0, max_intervals=0),
    dcc.Store(id='uuid', storage_type='local')
], fluid=True)

clientside_callback(
    ClientsideFunction(
        namespace='clientside',
        function_name='get_uuid'
    ),
    Output('uuid', 'data'),
    Input('get-uuid', 'n_intervals'),
    State('uuid', 'data')
)

clientside_callback(
    ClientsideFunction(
        namespace='clientside',
        function_name='display_uuid'
    ),
    Output('uuid-display', 'children'),
    Input('page-load', 'n_intervals'),
    State('uuid', 'data')
)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
