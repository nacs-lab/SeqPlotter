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
        }
    }
});