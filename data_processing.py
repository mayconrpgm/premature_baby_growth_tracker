"""Data processing module for loading, transforming, and exporting data."""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import math
import re
from typing import Dict, List, Tuple, Optional, Union, Any

# --- Data Loading and Processing ---
def load_intergrowth_data() -> Tuple[pd.DataFrame, Optional[str]]:
    """Loads and processes the INTERGROWTH-21st data.
    
    Returns:
        Tuple containing:
            - DataFrame: Processed data or empty DataFrame if error occurs
            - Optional[str]: Error message if any, None otherwise
    """
    try:
        # Updated path to the data folder
        df_long = pd.read_csv('data/intergrowth_combined_data.csv')
        
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

        # Pivot the table from long to wide format
        df_wide = df_long.pivot_table(
            index=['sex', 'param', 'ga'],
            columns='curve',
            values='measurement'
        ).reset_index()

        # Clean up column names after pivot
        df_wide.columns.name = None
        
        # Convert gestational age weeks to numeric
        df_wide['ga'] = pd.to_numeric(df_wide['ga'], errors='coerce')
        df_wide.dropna(subset=['ga'], inplace=True)
        
        return df_wide, None
    except FileNotFoundError:
        error_msg = "Error: `data/intergrowth_combined_data.csv` not found. Please make sure the data file is in the data directory."
        return pd.DataFrame(), error_msg
    except Exception as e:
        error_msg = f"An error occurred while processing the data file: {e}\nPlease ensure the CSV has columns like 'sex', 'metric', 'measurement_type', 'age_weeks', and 'value'."
        return pd.DataFrame(), error_msg

def get_z_score_and_percentile(pma_decimal: float, value: float, sex: str, metric: str, df: pd.DataFrame) -> Tuple[float, float]:
    """
    Calculates the Z-score and estimated percentile for a given measurement.
    This is a simplified estimation by interpolating between the nearest curves.
    
    Args:
        pma_decimal: Decimal weeks of postmenstrual age
        value: Measurement value
        sex: 'Boy' or 'Girl'
        metric: Measurement type (e.g., 'Weight_kg', 'Length_cm', 'Head_Circumference_cm')
        df: DataFrame containing reference data
        
    Returns:
        Tuple containing:
            - float: Estimated Z-score
            - float: Estimated percentile
        Tuple of (z_score, percentile)
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
    import math
    percentile = 0.5 * (1 + math.erf(z_score / np.sqrt(2))) * 100
    
    return z_score, percentile

def prepare_export_dataframe(patient_data):
    """
    Prepares patient data for CSV export.
    
    Args:
        patient_data: DataFrame containing patient measurements
        
    Returns:
        DataFrame formatted for export
    """
    export_df = pd.DataFrame({
        'PMA': [f"{row['pma_weeks']}w {row['pma_days']}d" for _, row in patient_data.iterrows()],
        'Date': [row['measurement_date'].strftime('%Y-%m-%d') if pd.notnull(row['measurement_date']) else '' 
                for _, row in patient_data.iterrows()],
        'Weight': patient_data['weight'],
        'Length': patient_data['length'],
        'HC': patient_data['hc']
    })
    
    return export_df

def import_patient_data(file_path):
    """
    Imports patient data from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Tuple of (patient_data_df, patient_info_dict or None, error_message or None)
    """
    try:
        # Try to extract patient info from filename using robust bracketed tags
        patient_info = None
        filename = os.path.basename(file_path)

        # New format: P[<name>] _GA[<weeks>w<days>d] _DOB[YYYYMMDD] _G[M|F] _<timestamp>.csv
        name_match = re.search(r"P\[(.*?)\]", filename)
        ga_match = re.search(r"GA\[(.*?)\]", filename)
        dob_match = re.search(r"DOB\[(\d{8})\]", filename)
        sex_match = re.search(r"G\[(M|F)\]", filename)

        parsed = False
        if name_match or ga_match or dob_match or sex_match:
            try:
                name_part = name_match.group(1) if name_match else None
                ga_str = ga_match.group(1) if ga_match else None

                ga_weeks, ga_days = None, None
                if ga_str:
                    # Accept formats like '31w4d', '31w', '31w 4d'
                    m = re.match(r"^(\d+)\s*w(?:\s*(\d+)\s*d)?$", ga_str)
                    if m:
                        ga_weeks = int(m.group(1))
                        ga_days = int(m.group(2)) if m.group(2) is not None else 0

                birth_date = None
                if dob_match:
                    birth_date = datetime.strptime(dob_match.group(1), '%Y%m%d').date()

                sex = None
                if sex_match:
                    sex = 'Male' if sex_match.group(1) == 'M' else 'Female'

                patient_info = {}
                if name_part:
                    patient_info['name'] = name_part
                if ga_weeks is not None:
                    patient_info['ga_weeks'] = ga_weeks
                if ga_days is not None:
                    patient_info['ga_days'] = ga_days
                if birth_date is not None:
                    patient_info['birth_date'] = birth_date
                if sex is not None:
                    patient_info['sex'] = sex

                # Only mark parsed if we actually extracted something meaningful
                parsed = len(patient_info) > 0
            except Exception:
                parsed = False

        # Fallback to legacy parsing if bracketed tags weren't found or failed
        if not parsed and ('_GA' in filename and '_DOB' in filename):
            try:
                name_part = filename.split('_GA')[0]
                ga_part = filename.split('_GA')[1].split('_DOB')[0]
                dob_part = filename.split('_DOB')[1]

                # Check if sex is included in the filename
                sex = None
                if '_M.csv' in dob_part:
                    sex = "Male"
                    dob_part = dob_part.split('_M.csv')[0]
                elif '_F.csv' in dob_part:
                    sex = "Female"
                    dob_part = dob_part.split('_F.csv')[0]
                else:
                    dob_part = dob_part.split('.csv')[0]

                # Parse GA
                ga_weeks = int(ga_part.split('w')[0])
                ga_days = int(ga_part.split('w')[1].split('d')[0])

                # Parse birth date
                birth_date = datetime.strptime(dob_part, '%Y%m%d').date()

                # Store patient info
                patient_info = {
                    'name': name_part,
                    'ga_weeks': ga_weeks,
                    'ga_days': ga_days,
                    'birth_date': birth_date
                }

                # Add sex if available
                if sex:
                    patient_info['sex'] = sex
            except Exception:
                patient_info = None
        
        imported_df = pd.read_csv(file_path)
        required_columns = ['PMA', 'Date', 'Weight', 'Length', 'HC']
        
        if not all(col in imported_df.columns for col in required_columns):
            error_msg = "CSV must contain columns: PMA, Date, Weight, Length, HC"
            return pd.DataFrame(), patient_info, error_msg
        
        # Process the imported data
        patient_data = pd.DataFrame(columns=[
            'pma_weeks', 'pma_days', 'measurement_date', 
            'weight', 'length', 'hc'
        ])
        
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
            # Create a new DataFrame with the row data and ensure it has the same structure
            new_row_df = pd.DataFrame([new_data])
            # Only concatenate if the new row has valid data
            if not new_row_df.empty:
                patient_data = pd.concat([patient_data, new_row_df], ignore_index=True)
        
        # Sort and remove duplicates
        patient_data = (
            patient_data
            .sort_values(by=['pma_weeks', 'pma_days'])
            .drop_duplicates(subset=['pma_weeks', 'pma_days', 'weight', 'length', 'hc'])
            .reset_index(drop=True)
        )
        
        return patient_data, patient_info, None
    except Exception as e:
        error_msg = f"Error importing data: {str(e)}"
        return pd.DataFrame(), None, error_msg