import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

dash.register_page(__name__, path='/help')

file_path = './pages/help.md'

with open(file_path, 'r') as file:
    file_content = file.read()

layout = dbc.Container([
    html.Div('Documentation', className='text-center display-4'),
    dcc.Markdown(file_content, className='mt-4', dangerously_allow_html=True),
])