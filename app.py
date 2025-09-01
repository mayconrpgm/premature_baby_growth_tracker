
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from fpdf import FPDF
import base64
from sklearn.linear_model import LinearRegression
import math

# --- Page Configuration ---
st.set_page_config(
    page_title="INTERGROWTH-21st Premature Growth Tracker",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Data Loading and Caching ---
@st.cache_data(show_spinner=True)
def load_data():
    """Loads, processes, and caches the INTERGROWTH-21st data."""
    try:
        df_long = pd.read_csv('intergrowth_combined_data.csv')
        st.write("Debug - Colunas originais:", df_long.columns.tolist())
        st.write("Debug - Primeiras linhas:", df_long.head())
        
        # Rename columns for clarity and consistency
        df_long.rename(columns={
            'sex': 'sex',
            'metric': 'param',
            'measurement_type': 'curve',
            'age_weeks': 'ga',
            'measurement_value': 'percentile',
            'value': 'measurement'
        }, inplace=True)

        # Map centile values to our expected column names and calculate z-scores
        centile_map = {
            '3rd': 'p3',
            '5th': 'p5',
            '10th': 'p10',
            '50th': 'p50',
            '90th': 'p90',
            '95th': 'p95',
            '97th': 'p97'
        }
        
        # Map percentiles to z-scores
        z_score_map = {
            '-3': 'z-3',  # Approximately -2.58
            '-2': 'z-2',  # Approximately -1.88
            '-1': 'z-1', # Approximately -1.28
            '0': 'z0',  # Mean
            '1': 'z1',  # Approximately 1.28
            '2': 'z2',   # Approximately 1.88
            '3': 'z3',   # Approximately 2.58
        }
        
        # Create two rows for each measurement, one for percentile and one for z-score
        df_percentiles = df_long.copy()
        df_zscores = df_long.copy()
        
        df_percentiles['curve'] = df_percentiles['percentile'].map(centile_map)
        df_zscores['curve'] = df_zscores['percentile'].map(z_score_map)
        
        # Combine the dataframes and remove rows where mapping resulted in None
        df_long = pd.concat([df_percentiles, df_zscores])
        df_long = df_long.dropna(subset=['curve'])

        st.write("Debug - Valores únicos:", {
            'sex': df_long['sex'].unique(),
            'param': df_long['param'].unique(),
            'curve': df_long['curve'].unique()
        })

        # Pivot the table from long to wide format
        df_wide = df_long.pivot_table(
            index=['sex', 'param', 'ga'],
            columns='curve',
            values='measurement'
        ).reset_index()

        # Clean up column names after pivot
        df_wide.columns.name = None
        
        st.write("Debug - Colunas após pivot:", df_wide.columns.tolist())
        st.write("Debug - Primeiras linhas após pivot:", df_wide.head())
        
        # Convert gestational age weeks to numeric
        df_wide['ga'] = pd.to_numeric(df_wide['ga'], errors='coerce')
        df_wide.dropna(subset=['ga'], inplace=True)
        
        return df_wide
    except FileNotFoundError:
        st.error("Error: `intergrowth_combined_data.csv` not found. Please make sure the data file is in the same directory as the app.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred while processing the data file: {e}")
        st.info("Please ensure the CSV has columns like 'sex', 'metric', 'measurement_type', 'age_weeks', and 'value'.")
        return pd.DataFrame()

data = load_data()

# --- Helper Functions ---
def pma_to_decimal_weeks(weeks, days):
    """Converts PMA from (weeks, days) to decimal weeks."""
    return weeks + days / 7

def decimal_weeks_to_pma(decimal_weeks):
    """Converts decimal weeks to PMA (weeks, days)."""
    weeks = int(decimal_weeks)
    days = int(round((decimal_weeks - weeks) * 7))
    return weeks, days

def calculate_chronological_age(birth_date, measurement_date):
    """Calculates chronological age in days."""
    if isinstance(birth_date, datetime) and isinstance(measurement_date, datetime):
        return (measurement_date - birth_date).days
    return 0

def calculate_pma_from_date(birth_ga_decimal, birth_date, measurement_date):
    """Calculates PMA in decimal weeks from dates."""
    chrono_age_days = calculate_chronological_age(birth_date, measurement_date)
    return birth_ga_decimal + (chrono_age_days / 7)

def get_z_score_and_percentile(pma_decimal, value, sex, metric, df):
    """
    Calculates the Z-score and estimated percentile for a given measurement.
    This is a simplified estimation by interpolating between the nearest curves.
    """
    if df.empty or pma_decimal is None or value is None:
        return np.nan, np.nan

    filtered_df = df[(df['sex'] == sex) & (df['param'] == metric)]
    if filtered_df.empty:
        return np.nan, np.nan

    # Find the closest GA in the data
    closest_ga_row = filtered_df.iloc[(filtered_df['ga'] - pma_decimal).abs().argsort()[:1]]
    if closest_ga_row.empty:
        return np.nan, np.nan

    # Get mean (p50) and standard deviation from the closest row
    mean = closest_ga_row['p50'].iloc[0]
    # A simple way to estimate SD: (p97 - p3) / 4, assuming normality
    p97 = closest_ga_row['p97'].iloc[0]
    p3 = closest_ga_row['p3'].iloc[0]
    
    if pd.isna(mean) or pd.isna(p97) or pd.isna(p3) or (p97 - p3) == 0:
        return np.nan, np.nan

    sd = (p97 - p3) / (2 * 1.88) # Z for 97th is ~1.88, for 3rd is ~-1.88

    z_score = (value - mean) / sd
    
    # Percentile from Z-score (using standard normal distribution CDF)
    percentile = 0.5 * (1 + math.erf(z_score / np.sqrt(2))) * 100
    
    return z_score, percentile


# --- PDF Export Class ---
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, st.session_state.get('pdf_title', 'Growth Chart Report'), 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
        self.set_x(10)
        self.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 0, 'L')

    def summary_table(self, header, data):
        self.set_font('Arial', 'B', 10)
        col_widths = [25, 30, 25, 25, 25, 25]
        for i, item in enumerate(header):
            self.cell(col_widths[i], 7, item, 1, 0, 'C')
        self.ln()
        self.set_font('Arial', '', 9)
        for row in data:
            for i, item in enumerate(row):
                self.cell(col_widths[i], 6, str(item), 1, 0, 'C')
            self.ln()

def create_download_link(val, filename, link_text):
    """Generates a download link for a file."""
    b64 = base64.b64encode(val)
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}">{link_text}</a>'

# --- Session State Initialization ---
if 'patient_data' not in st.session_state:
    st.session_state.patient_data = pd.DataFrame(columns=['pma_weeks', 'pma_days', 'measurement_date', 'value'])
if 'birth_ga_weeks' not in st.session_state:
    st.session_state.birth_ga_weeks = 32
if 'birth_ga_days' not in st.session_state:
    st.session_state.birth_ga_days = 0
if 'birth_date' not in st.session_state:
    st.session_state.birth_date = datetime.now().date() - timedelta(days=30)


# --- UI: Sidebar ---
with st.sidebar:
    st.title("👶 Premature Growth Tracker")
    st.header("Patient Information")

    patient_name = st.text_input("Patient Name (Optional for PDF)", key="patient_name")
    
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.birth_ga_weeks = st.number_input("Birth GA (weeks)", min_value=22, max_value=42, value=st.session_state.birth_ga_weeks, step=1)
    with c2:
        st.session_state.birth_ga_days = st.number_input("(days)", min_value=0, max_value=6, value=st.session_state.birth_ga_days, step=1)
    
    st.session_state.birth_date = st.date_input("Birth Date", value=st.session_state.birth_date)

    sex = st.selectbox("Sex", ("Male", "Female"), key="sex")
    metric_map = {
        "Weight_kg": "weight",
        "Length_cm": "length",
        "Head_Circumference_cm": "hc"
    }
    reverse_metric_map = {v: k for k, v in metric_map.items()}
    
    metric = st.selectbox("Metric", ("weight", "length", "hc"), format_func=lambda x: {"weight": "Weight (kg)", "length": "Length (cm)", "hc": "Head Circumference (cm)"}[x], key="metric")
    display_mode = st.radio("Display Curves", ("Percentiles", "Z-Scores"), key="display_mode")

    st.header("Add Measurement")
    
    use_date_for_pma = st.toggle("Calculate PMA from Date", value=True)

    if use_date_for_pma:
        meas_date = st.date_input("Measurement Date", value=datetime.now().date())
        birth_ga_decimal = pma_to_decimal_weeks(st.session_state.birth_ga_weeks, st.session_state.birth_ga_days)
        pma_decimal_calc = calculate_pma_from_date(birth_ga_decimal, datetime.combine(st.session_state.birth_date, datetime.min.time()), datetime.combine(meas_date, datetime.min.time()))
        pma_w_calc, pma_d_calc = decimal_weeks_to_pma(pma_decimal_calc)
        st.info(f"Calculated PMA: {pma_w_calc} weeks + {pma_d_calc} days")
        meas_pma_weeks, meas_pma_days = pma_w_calc, pma_d_calc
    else:
        c3, c4 = st.columns(2)
        meas_pma_weeks = c3.number_input("PMA (weeks)", min_value=st.session_state.birth_ga_weeks, max_value=64, step=1)
        meas_pma_days = c4.number_input("PMA (days)", min_value=0, max_value=6, step=1)
        meas_date = None # Or estimate from PMA

    unit = {"weight": "kg", "length": "cm", "hc": "cm"}[metric]
    meas_value = st.number_input(f"Value ({unit})", min_value=0.0, step=0.1, format="%.3f" if metric == 'weight' else "%.1f")

    if st.button("➕ Add Measurement", use_container_width=True):
        new_data = {
            'pma_weeks': meas_pma_weeks,
            'pma_days': meas_pma_days,
            'measurement_date': meas_date,
            'value': meas_value
        }
        st.session_state.patient_data = pd.concat([st.session_state.patient_data, pd.DataFrame([new_data])], ignore_index=True)
        st.session_state.patient_data = st.session_state.patient_data.sort_values(by=['pma_weeks', 'pma_days']).reset_index(drop=True)

# --- Main Panel ---

# Use session state for filtering and labels
metric_label = {"weight": "Weight", "length": "Length", "hc": "Head Circumference"}[st.session_state.metric]
st.header(f"Growth Chart: {st.session_state.sex} - {metric_label}")

# Filter data for plotting (always use session state)
st.write("Debug - Estado da sessão:", {
    'sex': st.session_state.sex,
    'metric': st.session_state.metric
})

# Map interface values to dataframe values
sex_map = {"Male": "Boy", "Female": "Girl"}
selected_sex = sex_map[st.session_state.sex]
selected_metric = reverse_metric_map[st.session_state.metric]

st.write("Debug - Valores mapeados:", {
    'sex': selected_sex,
    'metric': selected_metric
})

chart_data = data[(data['sex'] == selected_sex) & (data['param'] == selected_metric) & (data['ga'] <= 64)]

st.write("Debug - Dados filtrados:", {
    'linhas': len(chart_data),
    'colunas': chart_data.columns.tolist(),
    'amostra': chart_data.head().to_dict()
})

# Debug curve columns
if st.session_state.display_mode == 'Percentiles':
    expected_curves = ['p3', 'p10', 'p50', 'p90', 'p97']
else:
    expected_curves = ['z-3', 'z-2', 'z-1', 'z0', 'z1', 'z2', 'z3']

st.write("Debug - Curve columns check:", {
    'expected_curves': expected_curves,
    'available_curves': [c for c in expected_curves if c in chart_data.columns],
    'sample_values': {c: chart_data[c].head().tolist() if c in chart_data.columns else 'Not found' 
                     for c in expected_curves}
})

# Create Figure with light theme
fig = go.Figure()
fig.update_layout(template="plotly_white")

# Add Percentile/Z-score curves
if not chart_data.empty:
    if st.session_state.display_mode == 'Percentiles':
        curves = ['p3', 'p10', 'p50', 'p90', 'p97']
        labels = ['3rd', '10th', '50th', '90th', '97th']
        colors = ['red', 'orange', 'green', 'orange', 'red']
    else: # Z-Scores
        curves = ['z-3', 'z-2', 'z-1', 'z0', 'z1', 'z2', 'z3']
        labels = ['-3', '-2', '-1', '0', '1', '2', '3']
        colors = ['darkred', 'red', 'orange', 'green', 'orange', 'red', 'darkred']

    for curve, label, color in zip(curves, labels, colors):
        if curve in chart_data.columns and chart_data[curve].notnull().any():
            fig.add_trace(go.Scatter(
                x=chart_data['ga'],
                y=chart_data[curve],
                mode='lines',
                name=label,
                line=dict(color=color, width=2, dash='dash' if curve not in ['p50', 'z0'] else 'solid'),
                hoverinfo='skip'
            ))

# Add Patient Data
if not st.session_state.patient_data.empty:
    patient_df = st.session_state.patient_data.copy()
    patient_df['pma_decimal'] = patient_df.apply(lambda row: pma_to_decimal_weeks(row['pma_weeks'], row['pma_days']), axis=1)
    
    fig.add_trace(go.Scatter(
        x=patient_df['pma_decimal'],
        y=patient_df['value'],
        mode='markers+lines',
        name='Patient',
        marker=dict(color='blue', size=10, symbol='circle'),
        line=dict(color='blue', width=1),
        hovertemplate='PMA: %{customdata[0]}w %{customdata[1]}d<br>Value: %{y:.2f}<extra></extra>',
        customdata=patient_df[['pma_weeks', 'pma_days']]
    ))

    # Add Trendline Projection
    if len(patient_df) >= 2:
        model = LinearRegression()
        X = patient_df[['pma_decimal']]
        y = patient_df['value']
        model.fit(X, y)
        
        last_pma = patient_df['pma_decimal'].max()
        projection_pma = np.array([[last_pma], [last_pma + 4]]) # Project 4 weeks ahead
        projection_values = model.predict(projection_pma)

        fig.add_trace(go.Scatter(
            x=projection_pma.flatten(),
            y=projection_values,
            mode='lines',
            name='Trend Projection',
            line=dict(color='purple', width=2, dash='dot'),
            hoverinfo='skip'
        ))

# Add Birth GA line
birth_ga_decimal = pma_to_decimal_weeks(st.session_state.birth_ga_weeks, st.session_state.birth_ga_days)
fig.add_vline(
    x=birth_ga_decimal,
    line_width=2,
    line_dash="dash",
    line_color="grey",
    annotation_text="Birth GA",
    annotation_position="top left"
)

# Update Layout
y_axis_title = {"weight": "Weight (kg)", "length": "Length (cm)", "hc": "Head Circumference (cm)"}[metric]
fig.update_layout(
    xaxis_title="Postmenstrual Age (weeks)",
    yaxis_title=y_axis_title,
    legend_title="Legend",
    hovermode="x unified",
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis=dict(range=[24, 64]), # Set default view range
    height=800  # Increased height (default is about 320px)
)

st.plotly_chart(fig, use_container_width=True)


# --- Data Table and Export ---
st.header("Patient Measurement History")

if not st.session_state.patient_data.empty:
    display_df = st.session_state.patient_data.copy()
    
    # Calculate derived values for the table
    birth_date_dt = datetime.combine(st.session_state.birth_date, datetime.min.time())
    birth_ga_decimal = pma_to_decimal_weeks(st.session_state.birth_ga_weeks, st.session_state.birth_ga_days)

    display_df['PMA (decimal)'] = display_df.apply(lambda r: pma_to_decimal_weeks(r['pma_weeks'], r['pma_days']), axis=1)
    display_df['Chronological Age (days)'] = display_df.apply(lambda r: calculate_chronological_age(birth_date_dt, datetime.combine(r['measurement_date'], datetime.min.time())) if r['measurement_date'] else 'N/A', axis=1)
    
    z_scores = []
    percentiles = []
    
    # Map the sex and metric values to match the dataframe values
    mapped_sex = sex_map[sex]  # Convert Male/Female to Boy/Girl
    mapped_metric = reverse_metric_map[metric]  # Convert weight/length/hc to Weight_kg/Length_cm/Head_Circumference_cm
    
    for index, row in display_df.iterrows():
        z, p = get_z_score_and_percentile(row['PMA (decimal)'], row['value'], mapped_sex, mapped_metric, data)
        z_scores.append(f"{z:.2f}" if not pd.isna(z) else "N/A")
        percentiles.append(f"{p:.1f}" if not pd.isna(p) else "N/A")

    display_df['Z-Score'] = z_scores
    display_df['Est. Percentile'] = percentiles

    # Format for display
    display_df['PMA'] = display_df.apply(lambda r: f"{r['pma_weeks']}w {r['pma_days']}d", axis=1)
    display_df['Value'] = display_df['value'].map(lambda x: f"{x:.3f}" if metric == 'weight' else f"{x:.1f}")
    display_df['Date'] = display_df['measurement_date'].map(lambda x: x.strftime('%Y-%m-%d') if pd.notnull(x) else 'N/A')

    st.dataframe(display_df[['PMA', 'Date', 'Chronological Age (days)', 'Value', 'Est. Percentile', 'Z-Score']], use_container_width=True)

    # Remove measurement
    if not display_df.empty:
        row_to_remove = st.selectbox("Select measurement to remove", options=display_df.index, format_func=lambda x: f"PMA: {display_df.loc[x, 'PMA']}, Value: {display_df.loc[x, 'Value']}")
        if st.button("➖ Remove Selected", use_container_width=True):
            st.session_state.patient_data = st.session_state.patient_data.drop(index=row_to_remove).reset_index(drop=True)
            st.rerun()

    # Export Buttons
    col1, col2 = st.columns(2)
    
    # CSV Export
    csv_export_df = display_df[['PMA', 'Date', 'Chronological Age (days)', 'Value', 'Est. Percentile', 'Z-Score']]
    csv = csv_export_df.to_csv(index=False).encode('utf-8')
    col1.download_button(
        label="📥 Download Data as CSV",
        data=csv,
        file_name=f"{patient_name or 'patient'}_growth_data.csv",
        mime='text/csv',
        use_container_width=True
    )

    # PDF Export
    if col2.button("📄 Generate PDF Report", use_container_width=True):
        with st.spinner("Generating PDF..."):
            pdf = PDF('P', 'mm', 'A4')
            st.session_state['pdf_title'] = f"Growth Report for {patient_name}" if patient_name else "Growth Report"
            pdf.add_page()
            
            # Add Chart Image
            img_bytes = fig.to_image(format="png", width=800, height=500, scale=2)
            pdf.image(img_bytes, x=10, y=25, w=190)
            pdf.ln(120) # Move down past the image

            # Add Summary Table
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, 'Measurement Summary', 0, 1, 'L')
            
            table_header = ['PMA', 'Chrono. Age (d)', 'Value', 'Percentile', 'Z-Score']
            table_data = [
                [row['PMA'], row['Chronological Age (days)'], row['Value'], row['Est. Percentile'], row['Z-Score']]
                for index, row in display_df.iterrows()
            ]
            pdf.summary_table(table_header, table_data)

            # Provide download link
            pdf_bytes = pdf.output(dest='S')
            if isinstance(pdf_bytes, str):
                pdf_output = pdf_bytes.encode('latin1')
            else:
                pdf_output = pdf_bytes
            st.markdown(create_download_link(pdf_output, f"{patient_name or 'patient'}_growth_report.pdf", "Click here to download your PDF report"), unsafe_allow_html=True)

else:
    st.info("No measurements added yet. Use the sidebar to add data points.")

# --- Instructions and Footer ---
st.divider()
with st.expander("Instructions"):
    st.markdown("""
    1.  **Enter Patient Info**: Fill in the patient's birth gestational age (GA), birth date, and sex in the sidebar.
    2.  **Select Metric**: Choose whether to track Weight, Length, or Head Circumference.
    3.  **Add Measurements**:
        - Use the "Calculate PMA from Date" toggle for automatic Postmenstrual Age (PMA) calculation.
        - Or, enter the PMA manually.
        - Input the measured value and click "Add Measurement".
    4.  **View Chart**: The chart will update automatically with each new data point.
    5.  **Manage Data**: View the history table below the chart. You can remove measurements if needed.
    6.  **Export**: Download the patient's data as a CSV or generate a comprehensive PDF report.
    """)

st.markdown("<div style='text-align: center; color: grey;'>Developed for clinical support based on INTERGROWTH-21st data.</div>", unsafe_allow_html=True)
