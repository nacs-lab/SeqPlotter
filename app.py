# Import packages
from dash import Dash, html, dcc, callback, Input, Output, State, Patch, ALL, MATCH, ctx
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import struct
import plotly.express as px
import base64
import plotly.graph_objs as go
from pathlib import Path

def get_latest_file(directory, extension):
    path = Path(directory)
    files = list(path.glob(f'*.{extension}'))
    if not files:
        return None
    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    return latest_file

def process_data(bytes_data):
    #Dump output to file in the following format
    #[nchns: 4B][[chn_name: null-terminated string]
    #[npts: 4B][
    #          [[times (int64): 8B][values (double): 8B][pulse_ids (uint32): 4B]] x npts]
    #          x nchns]

    fileContent = bytes_data
    location = 0
    nchns = int.from_bytes(fileContent[0:4], byteorder='little')

    chn_infos = []
    location = 4
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
    # Get debug info
    # Dump backtrace to file in the following format
    # [has_bt_info: 1B][nfilenames (uint32): 4B][[filename: nul-terminated
    #                   string] x nfilenames]
    #                   [nnames (uint32): 4B][[name: nul-terminated string] x
    #                   nnames]
    #                   [nobjs (uint32): 4B][[nframes (uint32):
    #                   4B][[filename_id (uint32): 4B][name_id (uint32):
    #                   4B][line_num (uint32): 4B] x nframes] x nobjs]
    bt_info = {}
    has_bt = int.from_bytes(fileContent[location:location + 1], byteorder='little')
    location += 1
    if has_bt != 0:
        nfilenames = int.from_bytes(fileContent[location:location + 4], byteorder='little')
        location += 4
        all_filenames = []
        for _ in range(nfilenames):
            split_array = fileContent[location:].split(b'\x00', 1)
            filename = split_array[0].decode('utf-8')
            all_filenames.append(filename)
            location += len(split_array[0]) + 1
        bt_info["filenames"] = all_filenames
        nnames = int.from_bytes(fileContent[location:location + 4], byteorder='little')
        location += 4
        all_names = []
        for _ in range(nnames):
            split_array = fileContent[location:].split(b'\x00', 1)
            name = split_array[0].decode('utf-8')
            all_names.append(name)
            location += len(split_array[0]) + 1
        bt_info["names"] = all_names
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
        bt_info["backtraces"] = bts

    return chn_infos, chn_map, bt_info

def create_default_figure():
    default_figure = make_subplots(specs=[[{"secondary_y": True}]])
    default_figure.update_layout(
        title="",
        xaxis=dict(
            title="Time (ms)",
        ),
        yaxis=dict(
            title="Value",
        ),
        yaxis2=dict(
            title="Frequency (Hz)"
        ),
        legend_title="Legend",
        font=dict(size=14)
    )
    return default_figure

def create_new_block(id_num, chn_names):
    """Create a new block with a figure and channel selector."""
    return html.Div([
        dcc.Dropdown(
            id={'type': 'chn_selector', 'index': id_num},
            options=[name for name in chn_names],
            placeholder="Select a channel",
            multi=True
        ),
        dcc.Graph(id={'type': 'data_plotter', 'index': id_num}, figure=create_default_figure(), mathjax=True),
        html.Pre(id={'type': 'figure_info', 'index': id_num}, children='Click on a point for more information', style={'whiteSpace': 'pre_wrap'}),
        dcc.Store(id={'type': 'chns_storage', 'index': id_num}, storage_type='memory')
    ])


# Initialize the app
app = Dash(prevent_initial_callbacks="initial_duplicate")

add_figure_btn = html.Button('Add Figure', id='add-figure-btn', n_clicks=0)
chn_selector = dcc.Dropdown(id={'type': 'chn_selector', 'index': -1},options=[], placeholder="Select a channel", multi=True)
plotted_data = dcc.Graph(id={'type': 'data_plotter', 'index': -1},figure=create_default_figure(), mathjax=True)
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
storage = dcc.Store(id='storage', storage_type='memory')
chns_storage = dcc.Store(id={'type': 'chns_storage', 'index': -1}, storage_type='memory')
figure_container = html.Div(id='figure-container', children=[])
figure_info = html.Pre(id={'type': 'figure_info', 'index': -1}, children='Click on a point for more information', style={'whiteSpace': 'pre_wrap'})

# App layout
app.layout = [
    html.Pre('My First App with Data\n', style={'whiteSpace': 'pre_wrap'}),
    file_uploader,
    uploaded_file,
    latest_file_uploader,
    add_figure_btn,
    chn_selector,
    plotted_data,
    figure_info,
    figure_container,
    msg,
    storage,
    chns_storage
]

@callback(
    Output(figure_container, 'children'),
    Input(add_figure_btn, 'n_clicks')
)
def add_figure(n_clicks):
    if n_clicks > 0:
        block = Patch()
        block.append(create_new_block(n_clicks, []))
        return block
    return []

@callback(
    Output(storage, 'data'),
    Output(uploaded_file, 'children'),
    Input(file_uploader, 'contents'),   
    Input(file_uploader, 'filename'),    
    Input('load-latest-file-btn', 'n_clicks'),
    State('file_upload_path', 'value'),
    prevent_initial_call=True
)
def upload_file(contents, filename, n_clicks, file_path):
    triggered_id = ctx.triggered_id
    if triggered_id == 'upload-data':
        if contents is None:
            return None, "Please upload a file."
        # strip off the prefix
        content_type, content_string = contents.split(',')
        decoded_bytes = base64.b64decode(content_string)
        chn_infos, chn_map, bt_info = process_data(decoded_bytes)
        #print(decoded_bytes)
        return [chn_infos, chn_map, bt_info], "Uploaded: " + filename
    if triggered_id == 'load-latest-file-btn':
        fullpath = get_latest_file(file_path, 'seq')
        if fullpath is None:
            return None, "No file with extension .seq found in the specified directory."
        fullpath = str(fullpath)
        with open(fullpath, 'rb') as f:
            bytes_data = f.read()
        chn_infos, chn_map, bt_info = process_data(bytes_data)
        return [chn_infos, chn_map, bt_info], "Loaded latest file: " + fullpath,
    return None, "Please upload a file."

@callback(
    Output({'type': 'chn_selector', 'index': ALL}, 'options'),
    Input(storage, 'data'),
    State({'type': 'chn_selector', 'index': ALL}, 'value')
)
def update_chn_selector(chn_infos, chn_selector_values):
    if chn_infos is None:
        return [[] for _ in chn_selector_values]
    this_chn_infos = chn_infos[0]
    if len(chn_infos[0]) == 0:
        return [[] for _ in chn_selector_values]
    nchns = len(this_chn_infos)
    options = [this_chn_infos[i]["name"] for i in range(nchns)]
    return [options for _ in chn_selector_values]

@callback(
    Output({'type': 'data_plotter', 'index': MATCH}, 'figure'),
    Output({'type': 'chns_storage', 'index': MATCH}, 'data'),
    Input({'type': 'chn_selector', 'index': MATCH}, 'value'),
    State(storage, 'data'),
    State({'type': 'chns_storage', 'index': MATCH}, 'data'),
    prevent_initial_call=True
)
def update_graph(selected_channel, tot_chn_infos, chns_storage):
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
            del fig['data'][idx]
    
    # Now add elements
    if len(added_channels) > 0:
        chn_map = tot_chn_infos[1]
        chn_infos = tot_chn_infos[0]
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
    Output({'type': 'figure_info', 'index': MATCH}, 'children'),
    Input({'type': 'data_plotter', 'index': MATCH}, 'clickData'),
    State(storage, 'data'),
)
def print_backtrace_info(point, data):
    if point is None:
        return "Click on a point to see more information."
    if data is None or (not data[2]):
        return "No backtrace info available."
    point = point['points']
    pulse_id = point[0]['customdata']
    res = 'Pulse id: ' + str(pulse_id)
    if pulse_id == 2**32 - 1:
        res = res + " (Default value)\n"
    elif pulse_id > len(data[2]['backtraces']) or pulse_id < 0:
        res = res + " Missing backtrace info\n"
    else:
        # Get backtrace info
        res = res + '\n'
        this_bt = data[2]['backtraces'][pulse_id]
        nframes = len(this_bt)
        for i in range(nframes):
            filename_id = this_bt[i][0]
            name_id = this_bt[i][1]
            line_num = this_bt[i][2]
            if filename_id >= len(data[2]['filenames']) or name_id >= len(data[2]['names']):
                res = res + f"Frame {i}: Missing info\n"
            else:
                filename = data[2]['filenames'][filename_id]
                name = data[2]['names'][name_id]
                res = res + f"Frame {i}: {filename}:{name}:{line_num} \n"
    return res

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
