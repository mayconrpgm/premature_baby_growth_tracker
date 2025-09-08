
"""Main application file for the INTERGROWTH-21st Preterm Growth Tracker."""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import base64

# Import custom modules
from data_processing import load_intergrowth_data, get_z_score_and_percentile, prepare_export_dataframe, import_patient_data
from charts import create_full_chart, debug_print
from pdf_export import generate_pdf_report, create_download_link
from patient import Patient
from util import pma_to_decimal_weeks, decimal_weeks_to_pma, calculate_chronological_age_days, format_age, calculate_corrected_age_days, calculate_pma_from_date

# --- Page Configuration ---
st.set_page_config(
    page_title="INTERGROWTH-21st Preterm Growth Tracker",
    page_icon="👶",
    initial_sidebar_state="expanded",
    layout="wide"
)

# Enable/Disable debug mode
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False

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
    del st.session_state['temp_patient_info']

# --- UI: Sidebar ---
with st.sidebar:
    st.title("👶 Preterm Growth Tracker")
    
    # Debug mode toggle
    st.session_state.debug_mode = st.toggle("Debug Mode", value=st.session_state.debug_mode)
    
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
            st.info(f"File '{original_filename}' has already been processed.")
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
                    new_temp_patient_info = {}
                    
                    if patient_info.get('name'):
                        new_temp_patient_info['name'] = patient_info['name']
                    if patient_info.get('birth_ga_weeks'):
                        new_temp_patient_info['ga_weeks'] = patient_info['birth_ga_weeks']
                    if patient_info.get('birth_ga_days'):
                        new_temp_patient_info['ga_days'] = patient_info['birth_ga_days']
                    if patient_info.get('birth_date'):
                        new_temp_patient_info['birth_date'] = patient_info['birth_date']
                    if patient_info.get('sex'):
                        new_temp_patient_info['sex'] = patient_info['sex']
                    
                    # Check if the new info is different from what we already have
                    if 'temp_patient_info' not in st.session_state or st.session_state.temp_patient_info != new_temp_patient_info:
                        st.session_state.temp_patient_info = new_temp_patient_info
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
            
    weight = st.number_input("Weight (kg)", min_value=0.0, step=0.001, format="%.3f", key="weight")
    length = st.number_input("Length (cm)", min_value=0.0, step=0.1, format="%.1f", key="length")
    hc = st.number_input("Head Circumference (cm)", min_value=0.0, step=0.1, format="%.1f", key="hc")

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
            # Filter patient data for this metric to only include rows with non-null values for this metric
            patient_metric_data = st.session_state.patient_data[
                st.session_state.patient_data[config['data_col']].notnull()
            ].copy() if not st.session_state.patient_data.empty else None
            
            create_full_chart(chart_data, config, metric=metric_key, patient_data=patient_metric_data)

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
        width='stretch',
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
        if st.button("➖ Remove Selected", width='stretch'):
            st.session_state.patient_data = st.session_state.patient_data.drop(index=row_to_remove).reset_index(drop=True)
            st.rerun()

    # Export Buttons
    col1, col2 = st.columns(2)
    
    # CSV Export
    # Prepare export dataframe in the format compatible with import
    export_df = prepare_export_dataframe(st.session_state.patient_data)
    
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
        width='stretch'
    )

    # PDF Export
    if col2.button("📄 Generate PDF Report", width='stretch'):
        with st.spinner("Generating PDF..."):
            pdf_output = generate_pdf_report(
                st.session_state.patient_data,
                patient_name,
                st.session_state.birth_ga_weeks,
                st.session_state.birth_ga_days,
                st.session_state.birth_date,
                st.session_state.sex
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
    st.info("No measurements added yet. Use the sidebar to add data points.")

st.markdown("<div style='text-align: center; color: grey;'>Developed for clinical support based on INTERGROWTH-21st data.</div>", unsafe_allow_html=True)
