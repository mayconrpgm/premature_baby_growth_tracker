# Create tabs for different measurements
weight_tab, length_tab, hc_tab = st.tabs(["Weight", "Length", "Head Circumference"])

# Mapping for metrics
metric_configs = {
    "weight": {
        "display_name": "Weight",
        "unit": "kg",
        "db_name": "Weight_kg",
        "format": ":.3f",
        "data_col": "weight"
    },
    "length": {
        "display_name": "Length",
        "unit": "cm",
        "db_name": "Length_cm",
        "format": ":.1f",
        "data_col": "length"
    },
    "hc": {
        "display_name": "Head Circumference",
        "unit": "cm",
        "db_name": "Head_Circumference_cm",
        "format": ":.1f",
        "data_col": "hc"
    }
}

# Create the charts in each tab
for tab, metric_key in [
    (weight_tab, "weight"),
    (length_tab, "length"),
    (hc_tab, "hc")
]:
    with tab:
        config = metric_configs[metric_key]
        st.header(f"{config['display_name']} Growth Chart: {st.session_state.sex}")
        selected_sex = sex_map[st.session_state.sex]
        debug_print("Debug - Chart Info:", {
            'sex': selected_sex,
            'metric': config['db_name']
        })
        
        # Filter data for this metric
        chart_data = data[
            (data['sex'] == selected_sex) & 
            (data['param'] == config['db_name']) & 
            (data['ga'] <= 64)
        ]
        
        if not chart_data.empty:
            create_full_chart(chart_data, config, metric=metric_key)
