
"""Main application file for the INTERGROWTH-21st Preterm Growth Tracker."""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import base64

# Import custom modules
from data_processing import load_intergrowth_data, get_z_score_and_percentile, prepare_export_dataframe, import_patient_data
from charts import create_full_chart
from pdf_export import generate_pdf_report, create_download_link
from patient import Patient
from util import pma_to_decimal_weeks, decimal_weeks_to_pma, calculate_chronological_age_days, format_age, calculate_corrected_age_days, calculate_pma_from_date, debug_print

# --- Page Configuration ---
st.set_page_config(
    page_title="INTERGROWTH-21st Preterm Growth Tracker",
    page_icon="👶",
    initial_sidebar_state="expanded",
    layout="wide"
)

# Set default sidebar width to 400px
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] { width: 400px; min-width: 400px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Enable/Disable debug mode
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False

# Check if PDF export is available (Plotly static image export)
def _is_pdf_export_available():
    try:
        import plotly.graph_objects as go
        # Minimal test figure to probe image export capability
        fig = go.Figure()
        fig.add_scatter(x=[0, 1], y=[0, 1], mode='lines')
        _ = fig.to_image(format="png", width=2, height=2)
        return True
    except Exception:
        return False

if 'pdf_export_available' not in st.session_state:
    st.session_state.pdf_export_available = _is_pdf_export_available()

# Load the data
data, error_msg = load_intergrowth_data()
if error_msg:
    st.error(error_msg)

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
if 'patient_name' not in st.session_state:
    st.session_state.patient_name = ""
if 'sex' not in st.session_state:
    st.session_state.sex = "Male"

# Handle temp patient info if available
if 'temp_patient_info' in st.session_state:
    if 'name' in st.session_state.temp_patient_info:
        st.session_state.patient_name = st.session_state.temp_patient_info['name']
    if 'ga_weeks' in st.session_state.temp_patient_info:
        st.session_state.birth_ga_weeks = st.session_state.temp_patient_info['ga_weeks']
    if 'ga_days' in st.session_state.temp_patient_info:
        st.session_state.birth_ga_days = st.session_state.temp_patient_info['ga_days']
    if 'birth_date' in st.session_state.temp_patient_info:
        st.session_state.birth_date = st.session_state.temp_patient_info['birth_date']
    if 'sex' in st.session_state.temp_patient_info:
        st.session_state.sex = st.session_state.temp_patient_info['sex']

# --- UI: Sidebar ---
with st.sidebar:
    st.title("👶 Preterm Growth Tracker")
       
    st.header("Patient Information")

    # CSV File Upload - Moved higher
    uploaded_file = st.file_uploader(
        "Import data from CSV",
        type=['csv'],
        key="measurement_csv_upload",
        help="Upload a CSV file containing measurements. The file should have columns: PMA, Date, Weight, Length, HC"
    )

    # Use key only, don't set value from session_state when key is the same
    patient_name = st.text_input("Patient Name (Optional for PDF)", key="patient_name")
    
    c1, c2 = st.columns(2)
    with c1:
        # Use key only, don't set value from session_state when key is the same
        birth_ga_weeks = st.number_input("Birth GA (weeks)", min_value=22, max_value=42, step=1, key="birth_ga_weeks")
    with c2:
        birth_ga_days = st.number_input("(days)", min_value=0, max_value=6, step=1, key="birth_ga_days")
    
    # Use key only, don't set value from session_state when key is the same
    birth_date = st.date_input("Birth Date", key="birth_date")
    # Use key only, don't set index from session_state when key is the same
    sex = st.selectbox("Sex", ("Male", "Female"), key="sex")
    
    metric_map = {
        "Weight_kg": "weight",
        "Length_cm": "length",
        "Head_Circumference_cm": "hc"
    }
    reverse_metric_map = {v: k for k, v in metric_map.items()}
    # Use key only, don't set index from session_state when key is the same
    display_mode = st.radio("Display Curves", ("Percentiles", "Z-Scores"), key="display_mode")

    # Check for patient info in imported file if available
    if uploaded_file is not None:
        # Save the uploaded file to a temporary location with original filename
        import tempfile
        import os
        
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Preserve the original filename
        original_filename = uploaded_file.name
        temp_file_path = os.path.join(temp_dir, original_filename)
        
        # Check if we've already processed this file
        skip_processing = False
        if 'last_processed_file' in st.session_state and st.session_state.last_processed_file == original_filename:
            # Skip processing if we've already processed this file
            # st.info(f"File '{original_filename}' has already been processed.")
            skip_processing = True
            
        if not skip_processing:
            # Save the file with its original name
            with open(temp_file_path, 'wb') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
            
            # Process the saved file
            patient_data, patient_info, error_msg = import_patient_data(temp_file_path)
            
            # Clean up the temporary file and directory
            os.unlink(temp_file_path)
            os.rmdir(temp_dir)
            
            # Store the filename we just processed
            st.session_state.last_processed_file = original_filename

            if error_msg:
                st.error(error_msg)
            elif not patient_data.empty:
                st.session_state.patient_data = patient_data
                st.success("Data imported successfully!")
                
                if patient_info:
                    # Check if we need to update patient info
                    need_update = False
                    
                    # Create a new temp_patient_info dictionary
                    temp_patient_info = {}
                    
                    if patient_info.get('name'):
                        temp_patient_info['name'] = patient_info['name']
                    if patient_info.get('ga_weeks'):
                        temp_patient_info['ga_weeks'] = patient_info['ga_weeks']
                    if patient_info.get('ga_days'):
                        temp_patient_info['ga_days'] = patient_info['ga_days']
                    if patient_info.get('birth_date'):
                        temp_patient_info['birth_date'] = patient_info['birth_date']
                    if patient_info.get('sex'):
                        temp_patient_info['sex'] = patient_info['sex']
                    
                    st.session_state.temp_patient_info = temp_patient_info
                    st.success("Patient information loaded from filename!")
                    st.rerun()

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
            
    weight = st.number_input("Weight (kg)", min_value=0.0, step=0.001, value=None, placeholder="2.500", format="%.3f", key="weight")
    length = st.number_input("Length (cm)", min_value=0.0, step=0.1, value=None, placeholder="35.0", format="%.1f", key="length")
    hc = st.number_input("Head Circumference (cm)", min_value=0.0, step=0.1, value=None, placeholder="30.0", format="%.1f", key="hc")

    if st.button("➕ Add Measurement", width='stretch'):
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

    # Debug mode toggle
    st.session_state.debug_mode = st.toggle("Debug Mode", value=st.session_state.debug_mode)
# --- Main Panel ---

# Instructions at the top
with st.expander("Instructions", expanded=False):
    st.markdown("""
    ### Using the INTERGROWTH-21st Preterm Growth Tracker
    
    1.  **Enter Patient Information**: 
        - Fill in the patient's name (optional)
        - Enter birth gestational age (GA) in weeks and days
        - Select birth date using the date picker
        - Choose the patient's sex (Male/Female)
    
    2.  **Adding Growth Measurements**:
        - Toggle "Calculate PMA from Date" ON for automatic Postmenstrual Age (PMA) calculation based on birth GA and measurement date
        - Toggle OFF to manually enter PMA in weeks and days
        - Input one or more measurements (Weight in kg, Length in cm, Head Circumference in cm)
        - Click "Add Measurement" to record the data point
    
    3.  **Importing and Exporting Data**:
        - **Import**: Upload a previously exported CSV file to restore patient data, including:
          * Patient information (Name, Birth GA, Birth Date, Sex)
          * All recorded measurements
          * The CSV must have columns: PMA_weeks, PMA_days, measurement_date, weight, length, hc
        - **Export Options**: 
          * CSV Export: Downloads all data in a format compatible with the import function
          * PDF Export: Generates a complete clinical report with patient info, growth charts, and measurement table
    
    4.  **Viewing Growth Charts**: 
        - Charts update automatically with each new measurement
        - Switch between Percentiles and Z-Scores views using the radio button
        - Each tab shows a different growth parameter (Weight, Length, Head Circumference)
        - Patient measurements appear as points on the chart
    
    5.  **Managing Measurement Data**: 
        - All measurements appear in the table below the charts
        - Sort by clicking column headers
        - Remove individual measurements using the "Delete" button if needed
        - Clear all data using the "Clear All Data" button in the sidebar
    """)

# Create tabs for different measurements
weight_tab, length_tab, hc_tab = st.tabs(["Weight", "Length", "Head Circumference"])

# Map interface values to dataframe values
sex_map = {"Male": "Boy", "Female": "Girl"}

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

st.session_state.chart_figures = {}

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
            # Filter patient data for this metric to only include rows with non-null values for this metric
            patient_metric_data = st.session_state.patient_data[
                st.session_state.patient_data[config['data_col']].notnull()
            ].copy() if not st.session_state.patient_data.empty else None
            
            fig = create_full_chart(chart_data, config, metric=metric_key, patient_data=patient_metric_data)

            st.session_state.chart_figures[metric_key] = fig


# --- Data Table and Export ---
st.header("Patient Measurement History")

if not st.session_state.patient_data.empty:
    # Delegate dataframe preparation to Patient object
    patient_obj = Patient(
        name=st.session_state.patient_name,
        birth_ga_weeks=st.session_state.birth_ga_weeks,
        birth_ga_days=st.session_state.birth_ga_days,
        birth_date=st.session_state.birth_date,
        sex=st.session_state.sex,
    )
    patient_obj.measurements = st.session_state.patient_data.copy()

    display_df, display_columns = patient_obj.get_measurements_dataframe_for_table(metric_configs, data)
            
    # Display the dataframe with row selection
    if not display_df.empty:
        # Initialize selection state if not exists
        if 'selected_row' not in st.session_state:
            st.session_state.selected_row = None
            
        # Display dataframe with selection enabled
        selected_df = st.dataframe(
            display_df[display_columns].style.format({col: "{}".format for col in display_columns}),
            width='stretch',
            hide_index=True,
            selection_mode="single-row",
            on_select="rerun",
            key="measurements_table"
        )
        
        # Get selected row from the dataframe selection
        selected_row = None
        if selected_df.selection.rows:
            selected_row = selected_df.selection.rows[0]
            
        remove_btn = st.button("➖ Remove Selected", width='stretch', disabled=selected_row is None)

        # Handle remove action
        if remove_btn and selected_row is not None:
            st.session_state.patient_data = st.session_state.patient_data.drop(index=selected_row).reset_index(drop=True)
            st.rerun()
            
    else:
        # Display empty dataframe when no data
        st.dataframe(
            display_df[display_columns] if not display_columns else pd.DataFrame(columns=display_columns),
            width='stretch',
            hide_index=True
        )

  
    # CSV Export
    # Prepare export dataframe in the format compatible with import
    export_df = prepare_export_dataframe(st.session_state.patient_data)
    
    # Create filename with encoded patient information
    filename = (
        f"P[{patient_name or 'patient'}]"
        f"_GA[{st.session_state.birth_ga_weeks}w{st.session_state.birth_ga_days}d]"
        f"_DOB[{st.session_state.birth_date.strftime('%Y%m%d')}]"
        f"_G[{'M' if st.session_state.sex == 'Male' else 'F'}]"
        f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    
    csv = export_df.to_csv(index=False).encode('utf-8')

    # Export Buttons
    col1, col2 = st.columns(2)
    
    col1.download_button(
        label="📥 Download Data as CSV",
        data=csv,
        file_name=filename,
        mime='text/csv',
        width='stretch'
    )

    # PDF Export (conditionally available)
    if st.session_state.pdf_export_available:
        if col2.button("📄 Generate PDF Report", width='stretch'):
            with st.spinner("Generating PDF..."):
                pdf_output = generate_pdf_report(
                    st.session_state.patient_data,
                    patient_name,
                    st.session_state.birth_ga_weeks,
                    st.session_state.birth_ga_days,
                    st.session_state.birth_date,
                    st.session_state.sex,
                    st.session_state.chart_figures
                )
                
                st.markdown(
                    create_download_link(
                        pdf_output, 
                        f"{patient_name or 'patient'}_growth_report.pdf", 
                        "Click here to download your PDF report"
                    ), 
                    unsafe_allow_html=True
                )
    else:
        # Hide the button on environments without image export support (e.g., Streamlit Cloud without Chrome)
        st.info("PDF export is unavailable in this environment.")

else:
    st.info("No measurements added yet. Use the sidebar to add data points.")

# Footer with attribution and links
st.markdown("""
<div style='text-align: center; color: grey; padding: 20px;'>
    <p>Based on data from <a href='https://intergrowth21.ndog.ox.ac.uk/preterm/' target='_blank'>INTERGROWTH-21st Preterm Growth Standards</a></p>
    <p>Developed by Maycon Queiros | <a href='https://github.com/mayconrpgm/preterm_baby_growth_tracker' target='_blank'>Source Code on GitHub</a></p>
</div>
""", unsafe_allow_html=True)
