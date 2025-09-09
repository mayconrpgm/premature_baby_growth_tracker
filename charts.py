"""Chart creation and visualization functions for growth charts."""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression

def debug_print(*args, **kwargs):
    """Prints debug information if debug mode is enabled."""
    if st.session_state.debug_mode:
        st.write(*args, **kwargs)

def pma_to_decimal_weeks(weeks, days):
    """Converts PMA from (weeks, days) to decimal weeks."""
    return weeks + days / 7

def create_full_chart(chart_data, config, metric=None, patient_data=None):
    """Creates a growth chart with specified metric and configuration.
    
    Args:
        chart_data: DataFrame containing the reference data
        config: Dictionary containing chart configuration
        metric: Optional string specifying the metric ('weight', 'length', or 'hc')
        patient_data: Optional DataFrame containing patient measurements for this metric
    
    Returns:
        Plotly figure object
    """
    # Use provided metric or extract from config
    debug_print("Debug - Creating chart for metric:", metric)

    # Create Figure with light theme
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
    )

    # Define curves based on display mode
    if st.session_state.display_mode == 'Percentiles':
        curves = ['p3', 'p10', 'p50', 'p90', 'p97']
        labels = ['3rd', '10th', '50th', '90th', '97th']
        colors = ['red', 'orange', 'green', 'orange', 'red']
    else:  # Z-Scores
        curves = ['z-3', 'z-2', 'z-1', 'z0', 'z1', 'z2', 'z3']
        labels = ['-3', '-2', '-1', '0', '1', '2', '3']
        colors = ['darkred', 'red', 'orange', 'green', 'orange', 'red', 'darkred']

    debug_print("Debug - Chart data columns:", chart_data.columns.tolist())
    debug_print("Debug - Curve columns check:", {
        'expected_curves': curves,
        'available_curves': [c for c in curves if c in chart_data.columns],
        'sample_values': {c: chart_data[c].head().tolist() if c in chart_data.columns else 'Not found' 
                         for c in curves}
    })

    # Add reference curves
    for curve, label, color in zip(curves, labels, colors):
        if curve in chart_data.columns and chart_data[curve].notnull().any():
            fig.add_trace(go.Scatter(
                x=chart_data['ga'],
                y=chart_data[curve],
                mode='lines',
                name=f"{label}",
                line=dict(color=color, width=2, dash='dash' if curve not in ['p50', 'z0'] else 'solid'),
                hoverinfo='skip'
            ))

    # Add patient data if available
    if patient_data is not None and not patient_data.empty:
        patient_df = patient_data.copy()
        
        if not patient_df.empty:
            patient_df['pma_decimal'] = patient_df.apply(
                lambda row: pma_to_decimal_weeks(row['pma_weeks'], row['pma_days']), 
                axis=1
            )
            
            debug_print("Debug - Patient data for plotting:", patient_df)
            
            fig.add_trace(go.Scatter(
                x=patient_df['pma_decimal'],
                y=patient_df[config['data_col']],
                mode='markers+lines',
                name='Patient',
                marker=dict(color='blue', size=10, symbol='circle'),
                line=dict(color='blue', width=1),
                hovertemplate='PMA: %{customdata[0]}w %{customdata[1]}d<br>Value: %{y:.2f}<extra></extra>',
                customdata=patient_df[['pma_weeks', 'pma_days']]
            ))

            # Add trendline if there are at least 2 points
            if len(patient_df) >= 2:
                model = LinearRegression()
                X = patient_df[['pma_decimal']]
                y = patient_df[config['data_col']]
                model.fit(X, y)
                
                last_pma = patient_df['pma_decimal'].max()
                projection_pma = np.array([[last_pma], [last_pma + 4]])
                projection_values = model.predict(projection_pma)

                fig.add_trace(go.Scatter(
                    x=projection_pma.flatten(),
                    y=projection_values,
                    mode='lines',
                    name='Trend Projection',
                    line=dict(color='purple', width=2, dash='dot'),
                    hoverinfo='skip'
                ))

    # Add birth GA line
    birth_ga_decimal = pma_to_decimal_weeks(st.session_state.birth_ga_weeks, st.session_state.birth_ga_days)
    fig.add_vline(
        x=birth_ga_decimal,
        line_width=2,
        line_dash="dash",
        line_color="grey",
        annotation_text="Birth GA",
        annotation_position="top left"
    )

    # Get y-axis title based on metric
    y_axis_titles = {
        "weight": "Weight (kg)",
        "length": "Length (cm)",
        "hc": "Head Circumference (cm)"
    }
    y_axis_title = y_axis_titles[metric]

    # Update layout
    fig.update_layout(
        xaxis_title="Postmenstrual Age (weeks)",
        yaxis_title=y_axis_title,
        legend_title="Legend",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(range=[24, 64], showgrid=True, dtick=1),
        yaxis=dict(dtick=1),
        legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99,
            orientation="h"
        ),
        height=800
    )

    # Display the chart
    st.plotly_chart(fig, use_container_width=True)
    
    return fig  # Return the figure for potential reuse

def create_chart_tab(tab, metric_key, metric_configs, data):
    """Creates a chart for a given metric in the specified tab.
    
    Args:
        tab: Streamlit tab object
        metric_key: String key for the metric ('weight', 'length', or 'hc')
        metric_configs: Dictionary of metric configurations
        data: DataFrame containing reference data
    """
    config = metric_configs[metric_key]
    
    # Map interface values to dataframe values
    sex_map = {"Male": "Boy", "Female": "Girl"}
    
    # Filter data for this metric
    selected_sex = sex_map[st.session_state.sex]
    metric_data = data[(data['sex'] == selected_sex) & 
                      (data['param'] == config['db_name']) & 
                      (data['ga'] <= 64)]
    
    if not metric_data.empty:
        # Create figure with the metric key using imported function
        create_full_chart(metric_data, config, metric=metric_key)
    else:
        st.info(f"No {config['display_name']} data available for the selected criteria.")