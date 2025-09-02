
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from fpdf import FPDF
import base64
import math
from growth_charts import create_full_chart, debug_print, pma_to_decimal_weeks

# --- Page Configuration ---
st.set_page_config(
    page_title="INTERGROWTH-21st Preterm Growth Tracker",
    page_icon="👶",
    initial_sidebar_state="expanded",
    layout="wide"
)

# Custom CSS to adjust sidebar width
st.markdown(
    """
    <style>
        [data-testid="stSidebar"][aria-expanded="true"]{
            min-width: 400px;
            max-width: 400px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Enable/Disable debug mode
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False

# --- Data Loading and Caching ---
@st.cache_data(show_spinner=True)
def load_data():
    """Loads, processes, and caches the INTERGROWTH-21st data."""
    try:
        df_long = pd.read_csv('intergrowth_combined_data.csv')
        debug_print("Colunas originais:", df_long.columns.tolist())
        debug_print("Primeiras linhas:", df_long.head())
        
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

        debug_print("Valores únicos:", {
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
        
        debug_print("Debug - Colunas após pivot:", df_wide.columns.tolist())
        debug_print("Debug - Primeiras linhas após pivot:", df_wide.head())
        
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

def calculate_chronological_age_days(birth_date, measurement_date):
    """Calculates chronological age in days."""
    if isinstance(birth_date, datetime) and isinstance(measurement_date, datetime):
        return (measurement_date - birth_date).days
    return 0

def format_age(days):
    """Formats age in months and days for display."""
    if isinstance(days, (int, float)) and days >= 0:
        months = int(days // 30)
        remaining_days = int(days % 30)
        return f"{months}m {remaining_days}d"
    return "N/A"

def calculate_corrected_age_days(birth_date, measurement_date, birth_ga_weeks, birth_ga_days):
    """Calculates corrected age in days."""
    if isinstance(birth_date, datetime) and isinstance(measurement_date, datetime):
        chronological_days = (measurement_date - birth_date).days
        # Calculate weeks to full term (40 weeks)
        weeks_to_term = 40 - (birth_ga_weeks + birth_ga_days/7)
        # Convert to days and subtract from chronological age
        corrected_days = chronological_days - (weeks_to_term * 7)
        return corrected_days if corrected_days >= 0 else 0
    return None

def calculate_pma_from_date(birth_ga_decimal, birth_date, measurement_date):
    """Calculates PMA in decimal weeks from dates."""
    chrono_age_days = calculate_chronological_age_days(birth_date, measurement_date)
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
    st.session_state.patient_data = pd.DataFrame(columns=[
        'pma_weeks', 'pma_days', 'measurement_date', 
        'weight', 'length', 'hc'
    ])
if 'birth_ga_weeks' not in st.session_state:
    st.session_state.birth_ga_weeks = 32
if 'birth_ga_days' not in st.session_state:
    st.session_state.birth_ga_days = 0
if 'birth_date' not in st.session_state:
    st.session_state.birth_date = datetime.now().date() - timedelta(days=30)
if 'display_mode' not in st.session_state:
    st.session_state.display_mode = "Percentiles"


# --- UI: Sidebar ---
with st.sidebar:
    st.title("👶 Preterm Growth Tracker")
    
    # Debug mode toggle
    st.session_state.debug_mode = st.toggle("Debug Mode", value=st.session_state.debug_mode)
    
    st.header("Patient Information")

    patient_name = st.text_input("Patient Name (Optional for PDF)", key="patient_name")
    
    # CSV File Upload - Moved higher
    uploaded_file = st.file_uploader(
        "Import measurements from CSV",
        type=['csv'],
        key="measurement_csv_upload",
        help="Upload a CSV file containing measurements. The file should have columns: PMA, Date, Weight, Length, HC"
    )
    
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
    display_mode = st.radio("Display Curves", ("Percentiles", "Z-Scores"), key="display_mode")

    # Check for patient info in imported file if available
    if uploaded_file is not None:
        try:
            # Try to extract patient info from filename
            filename = uploaded_file.name
            if '_GA' in filename and '_DOB' in filename:
                # Extract patient info from filename
                name_part = filename.split('_GA')[0]
                ga_part = filename.split('_GA')[1].split('_DOB')[0]
                dob_part = filename.split('_DOB')[1].split('.csv')[0]
                
                # Parse GA
                ga_weeks = int(ga_part.split('w')[0])
                ga_days = int(ga_part.split('w')[1].split('d')[0])
                
                # Parse birth date
                birth_date = datetime.strptime(dob_part, '%Y%m%d').date()
                
                # Update session state
                st.session_state.patient_name = name_part
                st.session_state.birth_ga_weeks = ga_weeks
                st.session_state.birth_ga_days = ga_days
                st.session_state.birth_date = birth_date
                
                st.success("Patient information loaded from filename!")
            
            imported_df = pd.read_csv(uploaded_file)
            required_columns = ['PMA', 'Date', 'Weight', 'Length', 'HC']
            if not all(col in imported_df.columns for col in required_columns):
                st.error("CSV must contain columns: PMA, Date, Weight, Length, HC")
            else:
                # Clear existing data before import
                st.session_state.patient_data = pd.DataFrame(columns=[
                    'pma_weeks', 'pma_days', 'measurement_date', 
                    'weight', 'length', 'hc'
                ])
                
                # Process the imported data
                for _, row in imported_df.iterrows():
                    pma_parts = row['PMA'].split('w')
                    weeks = int(pma_parts[0])
                    days = int(pma_parts[1].replace('d', ''))
                    new_data = {
                        'pma_weeks': weeks,
                        'pma_days': days,
                        'measurement_date': pd.to_datetime(row['Date']).date() if pd.notna(row['Date']) else None,
                        'weight': row['Weight'] if pd.notna(row['Weight']) else None,
                        'length': row['Length'] if pd.notna(row['Length']) else None,
                        'hc': row['HC'] if pd.notna(row['HC']) else None
                    }
                    st.session_state.patient_data = pd.concat([st.session_state.patient_data, pd.DataFrame([new_data])], ignore_index=True)
                st.success("Data imported successfully!")
                st.rerun()  # Rerun to update all UI elements
        except Exception as e:
            st.error(f"Error importing data: {str(e)}")

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
        meas_date = None

    # Measurement inputs
    st.subheader("Measurements")
    
    # CSV File Upload
    uploaded_file = st.file_uploader("Import measurements from CSV", type=['csv'], key="measurements_uploader")
    if uploaded_file is not None:
        try:
            imported_df = pd.read_csv(uploaded_file)
            required_columns = ['PMA', 'Date', 'Weight', 'Length', 'HC']
            
            if not all(col in imported_df.columns for col in required_columns):
                st.error("CSV must contain columns: PMA, Date, Weight, Length, HC")
            else:
                # Process the imported data
                for _, row in imported_df.iterrows():
                    pma_parts = row['PMA'].split('w')
                    weeks = int(pma_parts[0])
                    days = int(pma_parts[1].replace('d', ''))
                    new_data = {
                        'pma_weeks': weeks,
                        'pma_days': days,
                        'measurement_date': pd.to_datetime(row['Date']).date() if pd.notna(row['Date']) else None,
                        'weight': row['Weight'] if pd.notna(row['Weight']) else None,
                        'length': row['Length'] if pd.notna(row['Length']) else None,
                        'hc': row['HC'] if pd.notna(row['HC']) else None
                    }
                    st.session_state.patient_data = pd.concat([st.session_state.patient_data, pd.DataFrame([new_data])], ignore_index=True)
                # Sort and remove duplicates
                st.session_state.patient_data = (
                    st.session_state.patient_data
                    .sort_values(by=['pma_weeks', 'pma_days'])
                    .drop_duplicates(subset=['pma_weeks', 'pma_days', 'weight', 'length', 'hc'])
                    .reset_index(drop=True)
                )
                st.success("Data imported successfully!")
        except Exception as e:
            st.error(f"Error importing data: {str(e)}")
            
    weight = st.number_input("Weight (kg)", min_value=0.0, step=0.001, format="%.3f", key="weight")
    length = st.number_input("Length (cm)", min_value=0.0, step=0.1, format="%.1f", key="length")
    hc = st.number_input("Head Circumference (cm)", min_value=0.0, step=0.1, format="%.1f", key="hc")

    if st.button("➕ Add Measurement", use_container_width=True):
        if not any([weight > 0, length > 0, hc > 0]):
            st.error("At least one measurement must be provided")
        else:
            new_data = {
                'pma_weeks': meas_pma_weeks,
                'pma_days': meas_pma_days,
                'measurement_date': meas_date,
                'weight': weight if weight > 0 else None,
                'length': length if length > 0 else None,
                'hc': hc if hc > 0 else None
            }
            st.session_state.patient_data = pd.concat([st.session_state.patient_data, pd.DataFrame([new_data])], ignore_index=True)
            st.session_state.patient_data = st.session_state.patient_data.sort_values(by=['pma_weeks', 'pma_days']).reset_index(drop=True)

# --- Main Panel ---

# Instructions at the top
with st.expander("Instructions", expanded=False):
    st.markdown("""
    1.  **Enter Patient Info**: Fill in the patient's birth gestational age (GA), birth date, and sex in the sidebar.
    
    2.  **Adding Measurements**:
        - Use the "Calculate PMA from Date" toggle for automatic Postmenstrual Age (PMA) calculation.
        - Or, enter the PMA manually.
        - Input one or more measurements (Weight, Length, Head Circumference).
        - Click "Add Measurement".
    
    3.  **Import/Export Data**:
        - **Import**: Upload a previously exported CSV file to restore patient data, including:
          * Patient information (Name, Birth GA, Birth Date)
          * All recorded measurements
          * The CSV must have columns: PMA, Date, Weight, Length, HC
        - **Export**: 
          * CSV Export: Downloads all data in a format compatible with the import function
          * PDF Export: Generates a complete report with patient info, charts, and measurements
    
    4.  **View Charts**: 
        - Charts update automatically with each new data point
        - Switch between Percentiles and Z-Scores views
    
    5.  **Manage Data**: 
        - View all measurements in the table below the charts
        - Remove individual measurements if needed
    """)

# Create tabs for different measurements
weight_tab, length_tab, hc_tab = st.tabs(["Weight", "Length", "Head Circumference"])

# Map interface values to dataframe values
sex_map = {"Male": "Boy", "Female": "Girl"}
metric_map = {
    "Weight_kg": "weight",
    "Length_cm": "length",
    "Head_Circumference_cm": "hc"
}
reverse_metric_map = {v: k for k, v in metric_map.items()}

# Mapping for metrics
metric_configs = {
    "weight": {
        "display_name": "Weight",
        "unit": "kg",
        "db_name": "Weight_kg",
        "format": ":.3f",
        "y_axis_title": "Weight (kg)",
        "data_col": "weight"
    },
    "length": {
        "display_name": "Length",
        "unit": "cm",
        "db_name": "Length_cm",
        "format": ":.1f",
        "y_axis_title": "Length (cm)",
        "data_col": "length"
    },
    "hc": {
        "display_name": "Head Circumference",
        "unit": "cm",
        "db_name": "Head_Circumference_cm",
        "format": ":.1f",
        "y_axis_title": "Head Circumference (cm)",
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

# Map interface values to dataframe values
# Map interface values to dataframe values
sex_map = {"Male": "Boy", "Female": "Girl"}
metric_map = {
    "Weight_kg": "weight",
    "Length_cm": "length",
    "Head_Circumference_cm": "hc"
}
reverse_metric_map = {v: k for k, v in metric_map.items()}

# Create and populate each chart
# Function to create a chart tab
def create_full_chart_tab(tab, metric_key):
    """Creates a chart for a given metric in the specified tab."""
    config = metric_configs[metric_key]
    
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

# --- Data Table and Export ---


# --- Data Table and Export ---
st.header("Patient Measurement History")

if not st.session_state.patient_data.empty:
    display_df = st.session_state.patient_data.copy()
    
    # Calculate derived values for the table
    birth_date_dt = datetime.combine(st.session_state.birth_date, datetime.min.time())
    # Calculate ages
    display_df['Chronological Age'] = display_df.apply(
        lambda r: format_age(
            calculate_chronological_age_days(birth_date_dt, datetime.combine(r['measurement_date'], datetime.min.time()))
        ) if r['measurement_date'] else 'N/A', 
        axis=1)
    
    display_df['Corrected Age'] = display_df.apply(
        lambda r: format_age(
            calculate_corrected_age_days(
                birth_date_dt, 
                datetime.combine(r['measurement_date'], datetime.min.time()),
                st.session_state.birth_ga_weeks,
                st.session_state.birth_ga_days
            )
        ) if r['measurement_date'] else 'N/A',
        axis=1)
    
    # Map the sex and metric values to match the dataframe values
    mapped_sex = sex_map[st.session_state.sex]  # Convert Male/Female to Boy/Girl
    
    # Calculate z-scores for all metrics
    for metric_key, config in metric_configs.items():
        if display_df[config['data_col']].notnull().any():  # If we have any values for this metric
            metric_values = display_df[config['data_col']]
            metric_pma_decimal = display_df.apply(
                lambda r: pma_to_decimal_weeks(r['pma_weeks'], r['pma_days']), 
                axis=1
            )
            metric_z_scores = []
            metric_percentiles = []
            
            for i, value in enumerate(metric_values):
                if pd.notnull(value):
                    z, p = get_z_score_and_percentile(metric_pma_decimal.iloc[i], value, mapped_sex, config['db_name'], data)
                    metric_z_scores.append(f"{z:.2f}" if not pd.isna(z) else "N/A")
                    metric_percentiles.append(f"{p:.1f}" if not pd.isna(p) else "N/A")
                else:
                    metric_z_scores.append("N/A")
                    metric_percentiles.append("N/A")
            
            display_df[f"{metric_key}_Z-Score"] = metric_z_scores
            display_df[f"{metric_key}_Percentile"] = metric_percentiles

        # Format for display
        display_df['PMA'] = display_df.apply(lambda r: f"{r['pma_weeks']}w {r['pma_days']}d", axis=1)
        display_df['Date'] = display_df['measurement_date'].map(lambda x: x.strftime('%Y-%m-%d') if pd.notnull(x) else 'N/A')
        
        # Format all measurement values
        for metric_key, config in metric_configs.items():
            if display_df[config['data_col']].notnull().any():
                display_df[f"{metric_key}_Value"] = display_df[config['data_col']].map(
                    lambda x: f"{x:.3f}" if metric_key == 'weight' else f"{x:.1f}" if pd.notnull(x) else "N/A"
                )

    # Create list of columns to display
    display_columns = ['PMA', 'Date', 'Chronological Age', 'Corrected Age']
    for metric_key, config in metric_configs.items():
        if display_df[config['data_col']].notnull().any():  # Check if this metric has any non-null values
            display_df[config['display_name']] = display_df[config['data_col']].map(
                lambda x: f"{x:.3f}" if metric_key == 'weight' else f"{x:.1f}" if pd.notnull(x) else "-"
            )
            display_columns.append(config['display_name'])
            
    # Display the dataframe without index
    st.dataframe(
        display_df[display_columns].style.format({col: "{}".format for col in display_columns}),
        use_container_width=True,
        hide_index=True
    )

    # Remove measurement
    if not display_df.empty:
        # Create a descriptive string for each measurement
        def format_measurement(idx):
            row = display_df.loc[idx]
            values = []
            for metric_key in metric_configs:
                if f"{metric_key}_Value" in row and row[f"{metric_key}_Value"] != "N/A":
                    values.append(f"{metric_configs[metric_key]['display_name']}: {row[f'{metric_key}_Value']}")
            return f"PMA: {row['PMA']}, {', '.join(values)}"
            
        row_to_remove = st.selectbox(
            "Select measurement to remove", 
            options=display_df.index, 
            format_func=format_measurement
        )
        if st.button("➖ Remove Selected", use_container_width=True):
            st.session_state.patient_data = st.session_state.patient_data.drop(index=row_to_remove).reset_index(drop=True)
            st.rerun()

    # Export Buttons
    col1, col2 = st.columns(2)
    
    # CSV Export
    # Prepare export dataframe in the format compatible with import
    export_df = pd.DataFrame({
        'PMA': [f"{row['pma_weeks']}w {row['pma_days']}d" for _, row in st.session_state.patient_data.iterrows()],
        'Date': [row['measurement_date'].strftime('%Y-%m-%d') if pd.notnull(row['measurement_date']) else '' 
                for _, row in st.session_state.patient_data.iterrows()],
        'Weight': st.session_state.patient_data['weight'],
        'Length': st.session_state.patient_data['length'],
        'HC': st.session_state.patient_data['hc']
    })
    
    # Create filename with encoded patient information
    filename = (f"{patient_name or 'patient'}"
               f"_GA{st.session_state.birth_ga_weeks}w{st.session_state.birth_ga_days}d"
               f"_DOB{st.session_state.birth_date.strftime('%Y%m%d')}"
               f".csv")
    
    csv = export_df.to_csv(index=False).encode('utf-8')
    col1.download_button(
        label="📥 Download Data as CSV",
        data=csv,
        file_name=filename,
        mime='text/csv',
        use_container_width=True
    )

    # PDF Export
    if col2.button("📄 Generate PDF Report", use_container_width=True):
        with st.spinner("Generating PDF..."):
            pdf = PDF('P', 'mm', 'A4')
            st.session_state['pdf_title'] = f"Growth Report for {patient_name}" if patient_name else "Growth Report"
            
            # First page: Patient Information and Summary Table
            pdf.add_page()
            
            # Add Patient Information
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, 'Patient Information', 0, 1, 'L')
            pdf.set_font('Arial', '', 10)
            pdf.cell(0, 7, f"Name: {patient_name if patient_name else 'Not provided'}", 0, 1, 'L')
            pdf.cell(0, 7, f"Birth GA: {st.session_state.birth_ga_weeks}w {st.session_state.birth_ga_days}d", 0, 1, 'L')
            pdf.cell(0, 7, f"Birth Date: {st.session_state.birth_date.strftime('%Y-%m-%d')}", 0, 1, 'L')
            pdf.cell(0, 7, f"Sex: {st.session_state.sex}", 0, 1, 'L')
            pdf.ln(5)
            
            # Add Summary Table
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, 'Measurement Summary', 0, 1, 'L')
            
            # Create PDF table data
            table_header = ['PMA', 'Date', 'Chrono. Age', 'Corr. Age', 'Weight (kg)', 'Length (cm)', 'HC (cm)']
            table_data = []
            
            for idx, row in st.session_state.patient_data.iterrows():
                # Format weight with 3 decimal places if present
                weight_str = f"{row['weight']:.3f}" if pd.notnull(row['weight']) else '-'
                # Format length and HC with 1 decimal place if present
                length_str = f"{row['length']:.1f}" if pd.notnull(row['length']) else '-'
                hc_str = f"{row['hc']:.1f}" if pd.notnull(row['hc']) else '-'
                
                meas_date = datetime.combine(row['measurement_date'], datetime.min.time()) if pd.notnull(row['measurement_date']) else None
                
                row_data = [
                    f"{row['pma_weeks']}w {row['pma_days']}d",
                    row['measurement_date'].strftime('%Y-%m-%d') if pd.notnull(row['measurement_date']) else 'N/A',
                    format_age(
                        calculate_chronological_age_days(birth_date_dt, meas_date)
                    ) if pd.notnull(row['measurement_date']) else 'N/A',
                    format_age(
                        calculate_corrected_age_days(
                            birth_date_dt, 
                            meas_date,
                            st.session_state.birth_ga_weeks,
                            st.session_state.birth_ga_days
                        )
                    ) if pd.notnull(row['measurement_date']) else 'N/A',
                    weight_str,
                    length_str,
                    hc_str
                ]
                table_data.append(row_data)
            
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

st.markdown("<div style='text-align: center; color: grey;'>Developed for clinical support based on INTERGROWTH-21st data.</div>", unsafe_allow_html=True)
