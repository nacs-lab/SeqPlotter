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
from datetime import datetime
from flask import request
import os

dash.register_page(__name__, path='/')

data_dir = './data/'

app = dash.get_app()

def get_latest_file(directory, extension):
    path = Path(directory)
    files = list(path.glob(f'*.{extension}'))
    if not files:
        return None
    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    return latest_file

def dict_to_dash_elem(d, spacing=0, id=0, fig_id=0):
    res = []
    indent = 20 * spacing  # Indentation for nested elements
    indent_str = str(int(indent)) + 'px'
    colors = []
    for key, value in d.items():
        if isinstance(value, dict) and (('value' not in value) or ('type' not in value)):
            elem, id, int_colors = dict_to_dash_elem(value, spacing + 1, id, fig_id)
            color = None
            if "red" in int_colors:
                color = 'red'
                new_child = [html.Summary(html.Strong([key + " :"], style={'color': color}), id={'type': 'modified_value_sum', 'index': id, 'fig_id': fig_id}, style={"paddingLeft": indent_str})]
            elif "blue" in int_colors:
                color = 'blue'
                new_child = [html.Summary(html.Strong([key + " :"], style={'color': color}), id={'type': 'config_value_sum', 'index': id, 'fig_id': fig_id}, style={"paddingLeft": indent_str})]
            else:
                new_child = [html.Summary(key + " :", id={'type': 'default_value_sum', 'index': id, 'fig_id': fig_id}, style={"paddingLeft": indent_str})]
            id = id + 1
            new_child.extend(elem)
            new_elem = html.Details(children=new_child)
            res.append(new_elem)
            colors.append(color)
        else:
            color = None
            if value['type'] == 1:
                if value['value'] != value['config_value']:
                    new_child = [html.Div([html.Strong(key, style={'color': 'red'}), ' : ', html.Span([str(value['config_value'])], style={'color': 'blue'}), html.Strong(['\u21D2', str(value['value'])], style={'color': 'red'}), html.Br()],
                                          id={'type': 'modified_value', 'index': id, 'fig_id': fig_id}, style={"paddingLeft": indent_str})]
                    color = 'red'
                else:
                    # From config
                    new_child = [html.Div([html.Strong(key, style={'color': 'blue'}), ' : ' + str(value['value']), html.Br()],
                                          id={'type': 'config_value', 'index': id, 'fig_id': fig_id}, style={"paddingLeft": indent_str})]
                    color = 'blue'
            elif value['type'] == 2 and (value['value'] != value['old_value']):
                # Overwritten non-config value
                new_child = [html.Div([html.Strong(key, style={'color': 'red'}), ' : ' + str(value['old_value']), html.Strong(['\u21D2', str(value['value'])], style={'color': 'red'}), html.Br()],
                                          id={'type': 'modified_value', 'index': id, 'fig_id': fig_id}, style={"paddingLeft": indent_str})]
                color = 'red'
            elif value['type'] == 3:
                # Overwritten config value
                new_child = [html.Div([html.Strong(key, style={'color': 'red'}), ' : ', html.Span([str(value['config_value'])], style={'color': 'blue'}), html.Strong(['\u21D2', str(value['old_value'])]), html.Strong(['\u21D2', str(value['value'])], style={'color': 'red'}), html.Br()],
                                          id={'type': 'modified_value', 'index': id, 'fig_id': fig_id}, style={"paddingLeft": indent_str})]
                color = 'red'
            else:
                # Non-config default value taken
                new_child = [html.Div([html.Strong(key), ' : ' + str(value['value']), html.Br()],
                                      id={'type': 'default_value', 'index': id, 'fig_id': fig_id}, style={"paddingLeft": indent_str})]
            id = id + 1
            res.extend(new_child)
            colors.append(color)
    return res, id, colors

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
        has_params = int.from_bytes(fileContent[location:location + 1], byteorder='little')
        location += 1
        if has_params:
            split_array = fileContent[location:].split(b'\x00', 1)
            json_str = split_array[0].decode('utf-8')
            location = location + len(split_array[0]) + 1
            # print(repr(json_str))
            seq_info['params'] = json.loads(json_str)
        else:
            seq_info['params'] = {}
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

def create_new_block(id_num, chn_names, title, seq_name, params):
    """Create a new block with a figure and channel selector."""
    if params:
        param_elem, _, _ = dict_to_dash_elem(params, 0, 0, id_num)
    else:
        param_elem = html.Div(children='No parameters loaded.')
    return [
        dcc.Dropdown(
            id={'type': 'chn_selector', 'index': id_num},
            options=[name for name in chn_names],
            placeholder="Select a channel",
            multi=True
        ),
        dcc.Loading(dcc.Graph(id={'type': 'data_plotter', 'index': id_num}, figure=create_default_figure(title), mathjax=True), delay_show=200),
        #html.Div([
        #    html.Div("Min X: ", style={'display': 'inline-block', 'margin-right': '10px'}),
        #    dcc.Input(id={'type': 'min_X', 'index': id_num}, type='number', value='', style={'display': 'inline-block', 'width': '10%'}, debounce=True),
        #    html.Div("Max X: ", style={'display': 'inline-block', 'margin-right': '10px'}),
        #    dcc.Input(id={'type': 'max_X', 'index': id_num}, type='number', value='', style={'display': 'inline-block', 'width': '10%'}, debounce=True)
        #]),
        dbc.Row([
            dbc.Col([html.Pre(id={'type': 'figure_info', 'index': id_num}, children='Click on a point for more information.', style={'whiteSpace': 'pre_wrap'}),
                    # html.Pre(id={'type': 'add_bt_info', 'index': id_num}, children='', style={'whiteSpace': 'pre_wrap'}),
                    dcc.Store(id={'type': 'add_bt_info', 'index': id_num}, storage_type='memory'),
                    dbc.Checklist(options=[{'label': 'Show full backtrace', 'value': 1}], value=[], id={'type': 'show_full_bt', 'index': id_num}, switch=True)
                    ], width=7),
            dbc.Col([
                dbc.Row([
                    dbc.Col([dbc.Checklist(options = [{'label': 'Show overwritten values', 'value': 1}], value = [1], id={'type': 'overwrite_params_selector', 'fig_id': id_num}, switch=True)]),
                    dbc.Col([dbc.Checklist(options = [{'label': 'Show config values', 'value': 1}], value = [1], id={'type': 'config_params_selector', 'fig_id': id_num}, switch=True)]),
                    dbc.Col([dbc.Checklist(options = [{'label': 'Show default values', 'value': 1}], value = [1], id={'type': 'default_params_selector', 'fig_id': id_num}, switch=True)])
                ]),
                dbc.Row([dcc.Loading(children=[
                    html.Div([html.Pre(id={'type': 'figure_info2', 'index': id_num}, children=param_elem, style={'whiteSpace': 'pre_wrap'})          
                    ]),
                    html.Div(id={'type': 'dummy-loading-param', 'fig_id': id_num}, children='dummy', style={'display': 'none'})
                    ], delay_show=200)
                ], style={'height': '100vh'})]
            , width=5)
        ]
        ),
        dcc.Store(id={'type': 'chns_storage', 'index': id_num}, storage_type='memory'),
        dcc.Store(id={'type': 'seq_name', 'index': id_num}, data=seq_name, storage_type='memory')
    ]

delete_files_modal = html.Div(
    [
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Delete Files")),
                # dbc.Row([
                #         dbc.Col([html.Div('Profiles loaded from ' + profile_dir, className='h-100 w-100 text-wrap text-center mb-2')])
                #     ]),
                dbc.ModalBody([
                    dcc.Dropdown(
                        id='delete-file-selector',
                        options=[],
                        placeholder="Files to delete",
                        multi=True,
                        className='mb-2'
                    ),
                ]),
                dbc.ModalFooter([
                    dbc.Button("Delete", id='delete-files', n_clicks=0, color='danger'),
                    dbc.Button("Cancel", id='cancel-delete-files', n_clicks=0)
                ]),
            ],
            id="delete-files-dialog",
            is_open=False,
        ),
    ]
)


add_figure_elem = [dbc.Col(dcc.Dropdown(
            id='sequence_selector',
            options=[],
            placeholder="Select a sequence",
            className="mb-3 mt-3"
            )),
            dbc.Col(dbc.Button('Add Figure for this Sequence', id='add-figure-btn', n_clicks=0, color="primary", className="mb-3 mt-3"))]
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
            'borderWidth': '2px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=False,
        className='my-2'
    )

uploaded_file = html.Div()
# latest_file_uploader = dbc.Row([
#     dbc.Col([
#         dbc.Input(id='file_upload_path', placeholder='Enter path', type='text', value='', className='mt-2')]
#     , width=8),
#     dbc.Col([dbc.Button('Load Latest File', id='load-latest-file-btn', n_clicks=0, className='d-flex align-items-center mt-2')], width=2)
# ])
seq_cache = dcc.Store(id='seq_cache', storage_type='memory')
figs_container = dcc.Tabs(id='figs-container', value='', children=[])
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
layout = dbc.Container([
    dbc.Row([html.P("Drag and drop a .seq file into the below dotted box or enter a path to load in the latest .seq file from a directory. The box can also be clicked to choose a file.",className='text-body my-2')]),
    dbc.Row([file_uploader]),
    dbc.Row([uploaded_file]),
    dbc.Row([dbc.Col(dcc.Dropdown(
            id='seq_selector',
            options=[],
            placeholder="Select a file"
        ), width = 8), dbc.Col([dbc.Button('Load Latest File', id='load-latest-file-btn', n_clicks=0, className='d-flex align-items-center mt-2')], width=2),
        dbc.Col([dbc.Button('Delete Files', id='delete-files-btn', n_clicks=0, className='d-flex align-items-center mt-2')], width=2)]),
    # latest_file_uploader,
    dbc.Row(add_figure_elem),
    dbc.Row(dbc.Col(html.Div([figs_container], style={"height": "100vh", "overflowY": "auto"}))),
    delete_files_modal,
    dcc.Interval(id='page-load', interval=100, n_intervals=0, max_intervals=0),
    dcc.Interval(id='seq_selection_updater', interval=1000, n_intervals=0),
    seq_cache
], fluid=True)

@callback(
    Output('delete-files-dialog', 'is_open', allow_duplicate=True),
    Output('delete-file-selector', 'options'),
    Input("delete-files-btn", 'n_clicks'),
    Input('cancel-delete-files', 'n_clicks'),
    State('uuid', 'data'),
    prevent_initial_call = True
)
def open_delete_files_modal(open_btn, cancel_btn, uuid):
    triggered_id = ctx.triggered_id
    if (triggered_id == 'cancel-delete-files'):
        return 0, no_update
    elif (triggered_id == 'delete-files-btn'):
        dir_name = data_dir + str(uuid) + '/'
        dir_path = Path(dir_name)
        seq_files = list(dir_path.glob("*.seq"))
        seq_strings = [str(file.name) for file in seq_files]
        return 1, seq_strings
    else:
        return no_update, no_update, no_update
    
@callback(
    Output('delete-files-dialog', 'is_open', allow_duplicate=True),
    Input('delete-files', 'n_clicks'),
    State('delete-file-selector', 'value'),
    State('uuid', 'data'),
    prevent_initial_call = True
)
def delete_files(n_clicks, files_to_delete, uuid):
    if n_clicks > 0 and files_to_delete is not None and len(files_to_delete) > 0:
        dir_name = data_dir + str(uuid) + '/'
        for fname in files_to_delete:
            fullpath = os.path.join(dir_name, fname)
            if os.path.exists(fullpath):
                os.remove(fullpath)
        return 0
    return no_update

@app.server.route('/upload', methods=['POST'])
def upload_binary():
    binary_data = request.get_data()
    print(f"Received {len(binary_data)} bytes")
    # Strip off filename
    split_array = binary_data.split(b'\x00', 1)
    fname = split_array[0].decode('utf-8')
    # Strip off uuid
    split_array2 = split_array[1].split(b'\x00', 1)
    uuid = split_array2[0].decode('utf-8')
    decoded_bytes = split_array2[1]
    dir_name = data_dir + str(uuid)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    save_fname = current_time + '_' + fname
    fullpath = os.path.join(dir_name, save_fname)
    with open(fullpath, 'wb') as f:
        f.write(decoded_bytes)
    return "Received", 200

@callback(
    Output('seq_selector', 'options'),
    Input('seq_selection_updater', 'n_intervals'),
    State('uuid', 'data')
)
def update_seq_selector_options(n, uuid):
    if uuid is not None:
        dir_path = Path(data_dir + str(uuid) + '/')
        seq_files = list(dir_path.glob("*.seq"))
        seq_strings = [str(file.name) for file in seq_files]
        return seq_strings
    return no_update

@callback(
    Output(figs_container, 'children', allow_duplicate=True),
    Output(figs_container, 'value'),
    Input('add-figure-btn', 'n_clicks'),
    State('sequence_selector', 'value'),
    State(seq_cache, 'data'),
    prevent_initial_call=True
)
def add_figure(n_clicks, seq_name, data):
    if data is None:
        return no_update, no_update
    seq_cache = data[0]
    if seq_name not in seq_cache:
        return no_update, no_update
    if n_clicks > 0:
        seq_info = seq_cache[seq_name]
        nchns = len(seq_info['outputs'])
        chn_names = [seq_info['outputs'][i]["name"] for i in range(nchns)]
        fig_id = f'{seq_name}_{n_clicks}'
        params = seq_info['params']
        block = create_new_block(n_clicks, chn_names, seq_name, seq_name, params)
        patch = Patch()
        patch.append(dcc.Tab(label=seq_name, id={'type': 'fig_tab', 'index': n_clicks, 'seq': seq_name}, value=fig_id, children=block))

        return patch, fig_id
    return no_update, no_update

# @callback(
#     Output(figure_content, 'children', allow_duplicate=True),
#     Input(figs_container, 'value'),
#     State(fig_cache, 'data'),
#     prevent_initial_call=True
# )
# def update_figure_content(fig_id, fig_cache):
#     if fig_cache is None or fig_id is None or fig_id not in fig_cache:
#         return no_update
#     return fig_cache[fig_id]

@callback(
    Output(seq_cache, 'data'),
    Output(uploaded_file, 'children'),
    Output(figs_container, 'children', allow_duplicate=True),
    Output({'type': 'chns_storage', 'index': ALL}, 'data', allow_duplicate=True),
    Output({'type': 'seq_name', 'index': ALL}, 'data'),
    Output({'type': 'add_bt_info', 'index': ALL}, 'data', allow_duplicate=True),
    Input(file_uploader, 'contents'),   
    Input(file_uploader, 'filename'),
    Input('seq_selector', 'value'),
    Input('load-latest-file-btn', 'n_clicks'),
    # State('file_upload_path', 'value'),
    State(figs_container, 'children'),
    State('uuid', 'data'),
    prevent_initial_call=True
)
def upload_file(contents, filename, seq_selector_name, n_clicks, cur_figs, uuid):
    nfigs = len(cur_figs)
    none_array = [None for _ in range(nfigs)]
    triggered_id = ctx.triggered_id
    if triggered_id == 'upload-data':
        if contents is None:
            return None, "Please upload a file.", no_update, no_update, no_update, no_update
        # strip off the prefix
        content_type, content_string = contents.split(',')
        decoded_bytes = base64.b64decode(content_string)
        res = process_data(decoded_bytes)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        #print(decoded_bytes)
        # Save
        dir_name = data_dir + str(uuid)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        save_fname = current_time + '_' + filename
        fullpath = os.path.join(dir_name, save_fname)
        with open(fullpath, 'wb') as f:
            f.write(decoded_bytes)
        return res, "Uploaded: " + filename, [], none_array, none_array, none_array
    elif triggered_id == 'seq_selector':
        fname = data_dir + str(uuid) + '/' + seq_selector_name
        with open(fname, 'rb') as f:
            bytes_data = f.read()
        res = process_data(bytes_data)
        return res, "Loaded file: " + fname, [], none_array, none_array, none_array
    elif triggered_id == 'load-latest-file-btn':
        file_path = data_dir + str(uuid) + '/'
        fullpath = get_latest_file(file_path, 'seq')
        if fullpath is None:
            return None, "No file with extension .seq found in the specified directory.", no_update, no_update, no_update, no_update
        fullpath = str(fullpath)
        with open(fullpath, 'rb') as f:
            bytes_data = f.read()
        res = process_data(bytes_data)
        return res, "Loaded latest file: " + fullpath, [], none_array, none_array, none_array
    return None, "Please upload a file.", no_update, no_update, no_update, no_update

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
    Output({'type': 'figure_info', 'index': MATCH}, 'children', allow_duplicate=True),
    Output({'type': 'add_bt_info', 'index': MATCH}, 'data', allow_duplicate=True),
    Input({'type': 'data_plotter', 'index': MATCH}, 'clickData'),
    State(seq_cache, 'data'),
    State({'type': 'seq_name', 'index': MATCH}, 'data'),
    prevent_initial_call=True
)
def print_backtrace_info(point, data, seq_name):
    if point is None:
        return no_update, "Click on a point to see more information.", ''
    if data is None or (not data[1]):
        return no_update, "No backtrace info available.", ''
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
    pulse_id_str = 'Pulse id: ' + str(pulse_id)
    this_bt_idx = data[1]["bt_idxs"][data[2][seq_name]]
    if pulse_id == 2**32 - 1:
        first_line = pulse_id_str + " (Default value)\n"
        other_lines = ''
    elif pulse_id > len(data[1]['bt_infos'][this_bt_idx]['backtraces']) or pulse_id < 0:
        first_line = pulse_id_str + " Missing backtrace info\n"
        other_lines = ''
    else:
        # Get backtrace info
        # bt_info["bt_idxs"]
        pulse_id_str = pulse_id_str + '\n'
        res = ''
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
        split_arr = res.split('\n', 1)
        first_line = pulse_id_str + split_arr[0]
        other_lines = split_arr[1] if len(split_arr) > 1 else ''
    return fig, first_line, [first_line, other_lines]

@callback(
    Output({'type': 'figure_info', 'index': MATCH}, 'children', allow_duplicate=True),
    Input({'type': 'show_full_bt', 'index': MATCH}, 'value'),
    Input({'type': 'add_bt_info', 'index': MATCH}, 'data'),
    prevent_initial_call=True
)
def show_full_bt_info(show_full_bt, full_bt_info):
    if full_bt_info is None:
        return "No backtrace info available."
    if show_full_bt is None or len(show_full_bt) == 0:
        return full_bt_info[0]
    else:
        return full_bt_info[0] + '\n' + full_bt_info[1]

clientside_callback(
    ClientsideFunction(
        namespace='clientside',
        function_name='update_param_display'
    ),
    Output({'type': 'config_value', 'index': ALL, 'fig_id': MATCH}, 'style'),
    Output({'type': 'config_value_sum', 'index': ALL, 'fig_id': MATCH}, 'style'),
    Output({'type': 'dummy-loading-param', 'fig_id': MATCH}, 'children', allow_duplicate=True),
    Input({'type': 'config_params_selector', 'fig_id': MATCH}, 'value'),
    State({'type': 'config_value', 'index': ALL, 'fig_id': MATCH}, 'style'),
    State({'type': 'config_value_sum', 'index': ALL, 'fig_id': MATCH}, 'style'),
    prevent_initial_call=True
)

# @callback(
#     Output({'type': 'config_value', 'index': ALL, 'fig_id': MATCH}, 'style'),
#     Output({'type': 'config_value_sum', 'index': ALL, 'fig_id': MATCH}, 'style'),
#     Output({'type': 'dummy-loading-param', 'fig_id': MATCH}, 'children', allow_duplicate=True),
#     Input({'type': 'config_params_selector', 'fig_id': MATCH}, 'value'),
#     State({'type': 'config_value', 'index': ALL, 'fig_id': MATCH}, 'children'),
#     State({'type': 'config_value_sum', 'index': ALL, 'fig_id': MATCH}, 'children'),
#     prevent_initial_call=True
# )
# def show_or_hide_config_params(config_selector, config_values, config_values_sum):
#     patches = []
#     for _ in range(len(config_values)):
#         patch = Patch()
#         if config_selector is None or len(config_selector) == 0:
#             patch['display'] = 'none'
#         else:
#             patch['display'] = 'block'
#         patches.append(patch)
#     patches_sum = []
#     for _ in range(len(config_values_sum)):
#         patch = Patch()
#         if config_selector is None or len(config_selector) == 0:
#             patch['display'] = 'none'
#         else:
#             patch['display'] = 'list-item'
#         patches_sum.append(patch)
#     return patches, patches_sum,  str(np.random.random(1)[0])

clientside_callback(
    ClientsideFunction(
        namespace='clientside',
        function_name='update_param_display'
    ),
    Output({'type': 'modified_value', 'index': ALL, 'fig_id': MATCH}, 'style'),
    Output({'type': 'modified_value_sum', 'index': ALL, 'fig_id': MATCH}, 'style'),
    Output({'type': 'dummy-loading-param', 'fig_id': MATCH}, 'children', allow_duplicate=True),
    Input({'type': 'overwrite_params_selector', 'fig_id': MATCH}, 'value'),
    State({'type': 'modified_value', 'index': ALL, 'fig_id': MATCH}, 'style'),
    State({'type': 'modified_value_sum', 'index': ALL, 'fig_id': MATCH}, 'style'),
    prevent_initial_call=True
)

clientside_callback(
    ClientsideFunction(
        namespace='clientside',
        function_name='update_param_display'
    ),
    Output({'type': 'default_value', 'index': ALL, 'fig_id': MATCH}, 'style'),
    Output({'type': 'default_value_sum', 'index': ALL, 'fig_id': MATCH}, 'style'),
    Output({'type': 'dummy-loading-param', 'fig_id': MATCH}, 'children', allow_duplicate=True),
    Input({'type': 'default_params_selector', 'fig_id': MATCH}, 'value'),
    State({'type': 'default_value', 'index': ALL, 'fig_id': MATCH}, 'style'),
    State({'type': 'default_value_sum', 'index': ALL, 'fig_id': MATCH}, 'style'),
    prevent_initial_call=True
)