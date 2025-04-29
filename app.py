# Import packages
import dash
from dash import Dash, html, dcc, callback, Input, Output, State, Patch, ALL, MATCH, ctx, no_update
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
    dash.page_container
], fluid=True)
# Run the app
if __name__ == '__main__':
    app.run(debug=True)
