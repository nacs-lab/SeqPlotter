# Import packages
from dash import Dash, html, dcc, callback, Input, Output, State, Patch, ALL, MATCH, ctx, no_update
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import struct
import plotly.express as px
import base64
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
from pathlib import Path

def get_latest_file(directory, extension):
    path = Path(directory)
    files = list(path.glob(f'*.{extension}'))
    if not files:
        return None
    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    return latest_file

def dict_to_dash_elem(d, spacing=0):
    res = []
    indent = 20 * spacing  # Indentation for nested elements
    indent_str = str(int(indent)) + 'px'
    for key, value in d.items():
        if isinstance(value, dict):
            new_child = [html.Summary(key + " :", style={"paddingLeft": indent_str})]
            new_child.extend(dict_to_dash_elem(value, spacing + 1))
            new_elem = html.Details(children=new_child)
            res.append(new_elem)
        else:
            new_child = [html.Div([html.Strong(key), ' : ' + str(value), html.Br()], style={"paddingLeft": indent_str})]
            res.extend(new_child)
    return res

def process_data(bytes_data):
    #Dump output to file in the following format
    #[nchns: 4B][[chn_name: null-terminated string]
    #[npts: 4B][
    #          [[times (int64): 8B][values (double): 8B][pulse_ids (uint32): 4B]] x npts]
    #          x nchns]

    # Amended format for multiple sequences
    # seq_idx is 1 for first basic sequence and 2, etc. for additional ones
    # [nseqs (uint32): 4B][[seq_name: null-terminated string][seq_idx (uint32): 4B][nchns: 4B][[chn_name: null-terminated string]
    #              [npts: 4B][
    #                        [[times (int64): 8B][values (double): 8B][pulse_ids (uint32): 4B]] x npts]
    #              x nchns] x nseqs]

    seq_cache = {}
    seq_map = {} # Map from sequence name to index
    fileContent = bytes_data
    location = 0
    nseqs = int.from_bytes(fileContent[location:(location + 4)], byteorder='little')
    location += 4
    for i in range(nseqs):
        seq_info = dict()
        split_array = fileContent[location:].split(b'\x00', 1)
        seq_name = split_array[0].decode('utf-8')
        location = location + len(split_array[0]) + 1
        seq_info["seq_idx"] = int.from_bytes(fileContent[location:(location + 4)], byteorder='little')
        location += 4
        seq_name = seq_name + f" (BSeq {i}: {seq_info['seq_idx']})"
        seq_info["name"] = seq_name
        seq_map[seq_name] = i
        nchns = int.from_bytes(fileContent[location:(location + 4)], byteorder='little')

        chn_infos = []
        location +=4
        chn_map = {} # Map from channel name to index

        for chn_num in range(nchns):
            chn_info = dict()
            split_array = fileContent[location:].split(b'\x00', 1)
            name = split_array[0].decode('utf-8')
            chn_info["name"] = name
            chn_map[name] = chn_num
            location = location + len(split_array[0]) + 1
            npts = int.from_bytes(fileContent[location:(location + 4)], byteorder='little')
            location += 4
            chn_info["npts"] = npts
            chn_info["ts"] = []
            chn_info["vals"] = []
            chn_info["ids"] = []
            for idx in range(npts):
                res = struct.unpack('<qdI',fileContent[location:(location + 20)])
                location += 20
                chn_info["ts"].append(res[0])
                chn_info["vals"].append(res[1])
                chn_info["ids"].append(res[2])
            chn_info["ts"] = np.array(chn_info["ts"])
            chn_info["vals"] = np.array(chn_info["vals"])
            chn_info["ids"] = np.array(chn_info["ids"])
            chn_infos.append(chn_info)
        seq_info['outputs'] = chn_infos
        seq_info['chn_map'] = chn_map
        seq_cache[seq_name] = seq_info
    # Get debug info
    # Dump backtrace to file in the following format
    # [has_bt_info: 1B][nfilenames (uint32): 4B][[filename: nul-terminated
    #                   string] x nfilenames]
    #                   [nnames (uint32): 4B][[name: nul-terminated string] x
    #                   nnames]
    #                   [nobjs (uint32): 4B][[nframes (uint32):
    #                   4B][[filename_id (uint32): 4B][name_id (uint32):
    #                   4B][line_num (uint32): 4B] x nframes] x nobjs]

    # Amended format for multiple sequences
    # NOTE: pulse_ids are 0 indexed
    # Dump backtrace to file in the following format
    # [has_bt_info: 1B][[bt_idx (uint32): 4B] x nseqs][n_bts (uint32): 4B][[nfilenames (uint32): 4B][[filename: nul-terminated
    #                   string] x nfilenames]
    #                   [nnames (uint32): 4B][[name: nul-terminated string] x
    #                   nnames]
    #                   [nobjs (uint32): 4B][[nframes (uint32):
    #                   4B][[filename_id (uint32): 4B][name_id (uint32):
    #                   4B][line_num (uint32): 4B] x nframes] x nobjs] x n_bts]
    # bt_idx is zero indexed
    bt_info = {}
    has_bt = int.from_bytes(fileContent[location:location + 1], byteorder='little')
    location += 1
    if has_bt != 0:
        bt_idxs = []
        for i in range(nseqs):
            bt_idx = int.from_bytes(fileContent[location:location + 4], byteorder='little')
            location += 4
            bt_idxs.append(bt_idx)
        bt_info["bt_idxs"] = bt_idxs
        nbts = int.from_bytes(fileContent[location:location + 4], byteorder='little')
        location += 4
        all_bts = []
        for i in range(nbts):
            this_tot_bt = {}
            nfilenames = int.from_bytes(fileContent[location:location + 4], byteorder='little')
            location += 4
            all_filenames = []
            for _ in range(nfilenames):
                split_array = fileContent[location:].split(b'\x00', 1)
                filename = split_array[0].decode('utf-8')
                all_filenames.append(filename)
                location += len(split_array[0]) + 1
            this_tot_bt["filenames"] = all_filenames
            nnames = int.from_bytes(fileContent[location:location + 4], byteorder='little')
            location += 4
            all_names = []
            for _ in range(nnames):
                split_array = fileContent[location:].split(b'\x00', 1)
                name = split_array[0].decode('utf-8')
                all_names.append(name)
                location += len(split_array[0]) + 1
            this_tot_bt["names"] = all_names
            nobjs = int.from_bytes(fileContent[location:location + 4], byteorder='little')
            location += 4
            bts = [] # Backtrace for all objects
            for _ in range(nobjs):
                this_bt = []
                nframes = int.from_bytes(fileContent[location:location + 4], byteorder='little')
                location += 4
                for _ in range(nframes):
                    filename_id = int.from_bytes(fileContent[location:location + 4], byteorder='little')
                    location += 4
                    name_id = int.from_bytes(fileContent[location:location + 4], byteorder='little')
                    location += 4
                    line_num = int.from_bytes(fileContent[location:location + 4], byteorder='little')
                    location += 4
                    this_bt.append([filename_id, name_id, line_num])
                bts.append(this_bt)
            this_tot_bt["backtraces"] = bts
            all_bts.append(this_tot_bt)
        bt_info["bt_infos"] = all_bts
    return [seq_cache, bt_info, seq_map]

def create_default_figure(title = ''):
    default_figure = make_subplots(specs=[[{"secondary_y": True}]])
    default_figure.update_layout(
        title=title,
        xaxis=dict(
            title="Time (ms)",
            rangeslider=dict(visible=True,
                             yaxis = dict(rangemode='auto'))
        ),
        yaxis=dict(
            title="Value",
            fixedrange=False
        ),
        yaxis2=dict(
            title="Frequency (Hz)",
            fixedrange=False
        ),
        legend_title="Legend",
        font=dict(size=14)
    )
    default_figure.add_trace(go.Scatter(x=[], y=[], mode='markers', name='Selected',
                         marker=dict(opacity=0.75, color='yellow', size=12),
                         showlegend=False))
    return default_figure

def create_new_block(id_num, chn_names, title, seq_name):
    """Create a new block with a figure and channel selector."""
    return html.Div([
        dcc.Dropdown(
            id={'type': 'chn_selector', 'index': id_num},
            options=[name for name in chn_names],
            placeholder="Select a channel",
            multi=True
        ),
        dcc.Graph(id={'type': 'data_plotter', 'index': id_num}, figure=create_default_figure(title), mathjax=True),
        #html.Div([
        #    html.Div("Min X: ", style={'display': 'inline-block', 'margin-right': '10px'}),
        #    dcc.Input(id={'type': 'min_X', 'index': id_num}, type='number', value='', style={'display': 'inline-block', 'width': '10%'}, debounce=True),
        #    html.Div("Max X: ", style={'display': 'inline-block', 'margin-right': '10px'}),
        #    dcc.Input(id={'type': 'max_X', 'index': id_num}, type='number', value='', style={'display': 'inline-block', 'width': '10%'}, debounce=True)
        #]),
        dbc.Row([
            dbc.Col(html.Pre(id={'type': 'figure_info', 'index': id_num}, children='Click on a point for more information', style={'whiteSpace': 'pre_wrap'})),
            dbc.Col(html.Pre(id={'type': 'figure_info2', 'index': id_num}, children='Click on a point for more information2', style={'whiteSpace': 'pre_wrap'}))
        ]
        ),
        dcc.Store(id={'type': 'chns_storage', 'index': id_num}, storage_type='memory'),
        dcc.Store(id={'type': 'seq_name', 'index': id_num}, data=seq_name, storage_type='memory')
    ])


# Initialize the app
app = Dash(prevent_initial_callbacks="initial_duplicate", external_stylesheets=[dbc.themes.BOOTSTRAP])

add_figure_elem = html.Div([dcc.Dropdown(
            id='sequence_selector',
            options=[],
            placeholder="Select a sequence"
            ),
            dbc.Button('Add Figure for this Sequence', id='add-figure-btn', n_clicks=0, color="primary", className="mb-3 mt-3"),])
file_uploader = dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=False
    )
uploaded_file = html.Div()
latest_file_uploader = html.Div([
    html.Div("Enter path:", style={'display': 'inline-block', 'margin-right': '10px'}),
    dcc.Input(id='file_upload_path', type='text', value='', style={'display': 'inline-block', 'width': '50%'}),
    html.Button('Load Latest File', id='load-latest-file-btn', n_clicks=0, style={'display': 'inline-block'})
])
msg = html.Div()
seq_cache = dcc.Store(id='seq_cache', storage_type='memory')
figure_container = html.Div(id='figure-container', children=[])
#reset_block_signal = dcc.Store(id='reset_block_signal', data=0, storage_type='memory') # Used to reset blocks

# test_dict = {}
# test_dict['a'] = dict()
# test_dict['b'] = [4,6,8]
# test_dict['a']['b'] = dict()
# test_dict['a']['b']['c'] = 3.14159
# test_dict['a']['c'] = 7.589e-6
# # print(dict_to_dash_elem(test_dict))

# # test_elem = html.Div(id='test_elem', children=dict_to_dash_elem(test_dict))
# test_elem = html.Div(children=dict_to_dash_elem(test_dict))

# App layout
app.layout = dbc.Container([
    dbc.Row([html.Pre('My First App with Data\n', style={'whiteSpace': 'pre_wrap'})]),
    dbc.Row([file_uploader]),
    dbc.Row([uploaded_file]),
    dbc.Row([latest_file_uploader]),
    dbc.Row([add_figure_elem]),
    dbc.Row([figure_container]),
    msg,
    seq_cache
])

@callback(
    Output(figure_container, 'children', allow_duplicate=True),
    Input('add-figure-btn', 'n_clicks'),
    State('sequence_selector', 'value'),
    State(seq_cache, 'data'),
    prevent_initial_call=True
)
def add_figure(n_clicks, seq_name, data):
    if data is None:
        return []
    seq_cache = data[0]
    if seq_name not in seq_cache:
        return []
    seq_info = seq_cache[seq_name]
    nchns = len(seq_info['outputs'])
    chn_names = [seq_info['outputs'][i]["name"] for i in range(nchns)]
    if n_clicks > 0:
        block = Patch()
        block.append(create_new_block(n_clicks, chn_names, seq_name, seq_name))
        return block
    return []

@callback(
    Output(seq_cache, 'data'),
    Output(uploaded_file, 'children'),
    Output(figure_container, 'children', allow_duplicate=True),
    Output({'type': 'chns_storage', 'index': ALL}, 'data', allow_duplicate=True),
    Output({'type': 'seq_name', 'index': ALL}, 'data'),
    Input(file_uploader, 'contents'),   
    Input(file_uploader, 'filename'),
    Input('load-latest-file-btn', 'n_clicks'),
    State('file_upload_path', 'value'),
    State(figure_container, 'children'),
    prevent_initial_call=True
)
def upload_file(contents, filename, n_clicks, file_path, cur_figs):
    nfigs = len(cur_figs)
    none_array = [None for _ in range(nfigs)]
    triggered_id = ctx.triggered_id
    if triggered_id == 'upload-data':
        if contents is None:
            return None, "Please upload a file.", no_update, no_update, no_update
        # strip off the prefix
        content_type, content_string = contents.split(',')
        decoded_bytes = base64.b64decode(content_string)
        res = process_data(decoded_bytes)
        #print(decoded_bytes)
        return res, "Uploaded: " + filename, [], none_array, none_array
    if triggered_id == 'load-latest-file-btn':
        fullpath = get_latest_file(file_path, 'seq')
        if fullpath is None:
            return None, "No file with extension .seq found in the specified directory.", no_update, no_update, no_update
        fullpath = str(fullpath)
        with open(fullpath, 'rb') as f:
            bytes_data = f.read()
        res = process_data(bytes_data)
        return res, "Loaded latest file: " + fullpath, [], none_array, none_array
    return None, "Please upload a file.", no_update, no_update, no_update

@callback(
    Output('sequence_selector', 'options'),
    Input(seq_cache, 'data')
)
def update_sequence_selector(data):
    if data is None or len(data) == 0:
        return []
    seq_cache = data[0]  # Get the sequence cache
    options = [seq_name for seq_name in seq_cache.keys()]
    return options

# @callback(
#     Output({'type': 'chn_selector', 'index': ALL}, 'options'),
#     Input(storage, 'data'),
#     State({'type': 'chn_selector', 'index': ALL}, 'value')
# )
# def update_chn_selector(chn_infos, chn_selector_values):
#     if chn_infos is None:
#         return [[] for _ in chn_selector_values]
#     this_chn_infos = chn_infos[0]
#     if len(chn_infos[0]) == 0:
#         return [[] for _ in chn_selector_values]
#     nchns = len(this_chn_infos)
#     options = [this_chn_infos[i]["name"] for i in range(nchns)]
#     return [options for _ in chn_selector_values]

@callback(
    Output({'type': 'data_plotter', 'index': MATCH}, 'figure', allow_duplicate=True),
    Output({'type': 'chns_storage', 'index': MATCH}, 'data', allow_duplicate=True),
    Input({'type': 'chn_selector', 'index': MATCH}, 'value'),
    State(seq_cache, 'data'),
    State({'type': 'chns_storage', 'index': MATCH}, 'data'),
    State({'type': 'seq_name', 'index': MATCH}, 'data'),
    prevent_initial_call=True
)
def update_graph(selected_channel, seq_cache, chns_storage, seq_name):
    if chns_storage is None:
        curr_channels = []
    else:
        curr_channels = chns_storage
    if selected_channel is None:
        return create_default_figure(), []
    fig = Patch()

    # Figure out channels to remove and add
    removed_channels = []
    added_channels = []
    for chn in curr_channels:
        if chn not in selected_channel:
            removed_channels.append(chn)
    for chn in selected_channel:
        if chn not in curr_channels:
            added_channels.append(chn)
    if len(removed_channels) > 0:
        # We need to remove channels. Typically only one channel is removed, but just in case...
        removal_indices = []
        for chn_to_remove in removed_channels:
            removal_indices.append(curr_channels.index(chn_to_remove))
        removal_indices = np.array(removal_indices)
        # Now sort removal indices in descending order in place.
        removal_indices[::-1].sort()
        for idx in removal_indices:
            del fig['data'][idx + 1] # Additional +1 for the highlight trace
    
    # Now add elements
    if len(added_channels) > 0:
        this_seq = seq_cache[0][seq_name]
        chn_map = this_seq["chn_map"]
        chn_infos = this_seq["outputs"]
        for channel in added_channels:
            this_channel = int(chn_map[channel])
            chn_info = chn_infos[this_channel]
            x = np.array(chn_info['ts']) * 1e-9
            y = np.array(chn_info['vals'])
            plot_dict = {
                'x': x,
                'y': y,
                'mode': 'lines+markers',
                'name': channel,
                'type': 'scatter',
                'xaxis': 'x',
                'customdata': np.array(chn_info['ids'])
            }
            if np.max(y) < 1e6:
                plot_dict['yaxis'] = 'y'
            else:
                plot_dict['yaxis'] = 'y2'
            #trace = go.Scatter(x=x, y=y, mode='lines+markers', name=channel)
            fig['data'].append(plot_dict)
    #fig['layout']['font']['size'] = 14
        #fig['layout']['title']['text'] = ''
        #fig['layout']['xaxis']['title']['text'] = 'Time (ms)'
        #fig['layout']['yaxis']['title']['text'] = 'Value'
        #fig['layout']['legend']['title']['text'] = 'Legend'
    return fig, selected_channel

@callback(
    Output({'type': 'data_plotter', 'index': MATCH}, 'figure', allow_duplicate=True),
    Output({'type': 'figure_info', 'index': MATCH}, 'children'),
    Input({'type': 'data_plotter', 'index': MATCH}, 'clickData'),
    State(seq_cache, 'data'),
    State({'type': 'seq_name', 'index': MATCH}, 'data'),
    prevent_initial_call=True
)
def print_backtrace_info(point, data, seq_name):
    if point is None:
        return no_update, "Click on a point to see more information."
    if data is None or (not data[1]):
        return no_update, "No backtrace info available."
    fig = Patch()
    point = point['points']
    x = point[0]['x']
    y = point[0]['y']
    fig['data'][0]['x'] = [x]
    fig['data'][0]['y'] = [y]
    if y < 1e6:
        fig['data'][0]['yaxis'] = 'y'
    else:
        fig['data'][0]['yaxis'] = 'y2'
    pulse_id = point[0]['customdata']
    res = 'Pulse id: ' + str(pulse_id)
    this_bt_idx = data[1]["bt_idxs"][data[2][seq_name]]
    if pulse_id == 2**32 - 1:
        res = res + " (Default value)\n"
    elif pulse_id > len(data[1]['bt_infos'][this_bt_idx]['backtraces']) or pulse_id < 0:
        res = res + " Missing backtrace info\n"
    else:
        # Get backtrace info
        # bt_info["bt_idxs"]
        res = res + '\n'
        this_bt = data[1]['bt_infos'][this_bt_idx]['backtraces'][pulse_id]
        nframes = len(this_bt)
        for i in range(nframes):
            filename_id = this_bt[i][0]
            name_id = this_bt[i][1]
            line_num = this_bt[i][2]
            if filename_id >= len(data[1]['bt_infos'][this_bt_idx]['filenames']) or name_id >= len(data[1]['bt_infos'][this_bt_idx]['names']):
                res = res + f"Frame {i}: Missing info\n"
            else:
                filename = data[1]['bt_infos'][this_bt_idx]['filenames'][filename_id]
                name = data[1]['bt_infos'][this_bt_idx]['names'][name_id]
                res = res + f"Frame {i}: {filename}:{name}:{line_num} \n"
    return fig, res

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
