function generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        var r = crypto.getRandomValues(new Uint8Array(1))[0] % 16, v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

window.dash_clientside = Object.assign({}, window.dash_clientside, {
    clientside: {
        update_param_display: function(config_selector, current_styles, current_styles_sum) {
            const bVisible = (config_selector && config_selector.length > 0);
    
            // Ensure both style arrays are valid lists
            const styles1 = Array.isArray(current_styles) ? current_styles : [];
            const styles2 = Array.isArray(current_styles_sum) ? current_styles_sum : [];
    
            const output1 = styles1.map(style => ({
                ...style,
                display: bVisible ? "block" : "none"
            }));
    
            const output2 = styles2.map(style => ({
                ...style,
                display: bVisible ? "list-item" : "none"
            }));
    
            return [output1, output2, Math.random()];
        },
        // Output('uuid', 'data'),
        // Input('get-uuid', 'n_intervals'),
        // State('uuid', 'data')
        get_uuid: function(n_intervals, uuid) {
            if (!uuid) {
                return generateUUID();
            }
            return window.dash_clientside.no_update;
        },
        // Output('uuid-display', 'children'),
        // Input('page-load', 'n_intervals'),
        // State('uuid', 'data')
        display_uuid: function(n_intervals, uuid) {
            if (uuid) {
                return 'My UUID: ' + uuid;
            }
            return 'No UUID';
        },
    }
});