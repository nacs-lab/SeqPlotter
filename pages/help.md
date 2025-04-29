This is version **1.0.1**.

# Overview

The sequence plotter is a browser-based platform for plotting experimental sequences. By making use of the browser, we can make use of the many already existing packages for convenient rendering of plots and also intuitive interactivity. To that end we make use of
- **Dash**: Dash is a convenient Python based package utilizing **flask** for serving an app, **React** for its backend interactivity and **plotly** for utilities designed for visualizing graphs and tables. Dash also has many extensions for use together with **bootstrap**, for instance, for better styling.

## Uploading a File

The plotter requires uploading a file (in our own `.seq` format), from which it gathers all of the information about the sequence(s). This file supports multiple sequences (which can be several basic sequences in a sequence with branching or the sequences of a scan), and also supports (optionally) the inclusion of backtrace information and parameter information. After uploading a file, one can select the automatically populated sequence selector and choose a particular sequence to take a look at. Once the sequence is chosen, click "Add Figure for this Sequence" to create a figure to probe the sequence and also reveal additional sequence-specific information. Note that multiple figures can be opened for a particular sequence at your own convenience.

## Visualizing a Sequence

Once a figure is opened, the interface will create tabs which can be used to select your sequence of choice. The graph can be populated with user selected channels. The channel selector is automatically populated with all known channels and is searchable. The figure below automatically has two axes, a left and right axis. The left axis is used for all values below a million (i.e. NiDAQ, DDS/AWG amplitude channels and TTLs). The right axis is used for all values above a million (mainly frequencies). The figure can be zoomed into both using tools on the graph and the slider below the graph (for time axis zooming).

## Debugging a Sequence

Below the graph are two columns of information. The left column consists of backtrace information (i.e. which line of code is responsible for a selected point on the graph). The backtrace does not fully print but can be completely print by toggling the "Show full backtrace" slider. The right column consists of all parameter information stored in `s.C` from MATLAB in a nested tree structure. The source of `s.C` parameters can arise from definitions in your sequence or from `expConfig.m`. Those from the config are shown in blue and can be toggled with the "Show config values" slider. From the MATLAB, one can specify a reference ("default") sequence to compare this sequence to, and information from such a default sequence is stored in the `.seq` file. If the parameter values differ from the default sequence, the parameter will be shown in red and can be toggled on or off with the "Show overwritten values" slider. Lastly, values not changed from the default sequence will be in black and can be toggled on or off with the "Show default values" slider. Note that if no reference sequence is specified, there will be no overwritten values, and thus no value in red. If the reference sequence does not define a particular variable, a placeholder "?" is used for its value.

# Backend Details