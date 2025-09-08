"""PDF export functionality for growth charts."""

from fpdf import FPDF
import base64
from datetime import datetime
import pandas as pd
from util import format_age, calculate_chronological_age_days, calculate_corrected_age_days

class PDF(FPDF):
    """Custom PDF class for growth chart reports."""
    
    def header(self):
        """Define the header for each page."""
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, self.title, 0, 1, 'C')
        self.ln(5)

    def footer(self):
        """Define the footer for each page."""
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
        self.set_x(10)
        self.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 0, 'L')

    def summary_table(self, header, data):
        """Create a summary table in the PDF.
        
        Args:
            header: List of column headers
            data: List of rows, each containing a list of cell values
        """
        self.set_font('Arial', 'B', 10)
        col_widths = [25, 30, 25, 25, 25, 25, 25]  # Added one more column for HC
        for i, item in enumerate(header):
            self.cell(col_widths[i], 7, item, 1, 0, 'C')
        self.ln()
        self.set_font('Arial', '', 9)
        for row in data:
            for i, item in enumerate(row):
                self.cell(col_widths[i], 6, str(item), 1, 0, 'C')
            self.ln()

def create_download_link(val, filename, link_text):
    """Generates a download link for a file.
    
    Args:
        val: Binary content of the file
        filename: Name of the file to download
        link_text: Text to display for the download link
        
    Returns:
        HTML string containing the download link
    """
    b64 = base64.b64encode(val)
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}">{link_text}</a>'

def generate_pdf_report(patient_data, patient_name, birth_ga_weeks, birth_ga_days, birth_date, sex):
    """Generate a PDF report for the patient's growth data.
    
    Args:
        patient_data: DataFrame containing patient measurements
        patient_name: String name of the patient
        birth_ga_weeks: Integer weeks of gestational age at birth
        birth_ga_days: Integer days of gestational age at birth
        birth_date: Date object of birth date
        sex: String sex of the patient ('Male' or 'Female')
        
    Returns:
        Binary PDF content
    """
    pdf = PDF('P', 'mm', 'A4')
    pdf.title = f"Growth Report for {patient_name}" if patient_name else "Growth Report"
    
    # First page: Patient Information and Summary Table
    pdf.add_page()
    
    # Add Patient Information
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Patient Information', 0, 1, 'L')
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 7, f"Name: {patient_name if patient_name else 'Not provided'}", 0, 1, 'L')
    pdf.cell(0, 7, f"Birth GA: {birth_ga_weeks}w {birth_ga_days}d", 0, 1, 'L')
    pdf.cell(0, 7, f"Birth Date: {birth_date.strftime('%Y-%m-%d')}", 0, 1, 'L')
    pdf.cell(0, 7, f"Sex: {sex}", 0, 1, 'L')
    pdf.ln(5)
    
    # Add Summary Table
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Measurement Summary', 0, 1, 'L')
    
    # Create PDF table data
    table_header = ['PMA', 'Date', 'Chrono. Age', 'Corr. Age', 'Weight (kg)', 'Length (cm)', 'HC (cm)']
    table_data = []
    
    birth_date_dt = datetime.combine(birth_date, datetime.min.time())
    
    for _, row in patient_data.iterrows():
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
                    birth_ga_weeks,
                    birth_ga_days
                )
            ) if pd.notnull(row['measurement_date']) else 'N/A',
            weight_str,
            length_str,
            hc_str
        ]
        table_data.append(row_data)
    
    pdf.summary_table(table_header, table_data)
    
    # Get PDF bytes
    pdf_bytes = pdf.output(dest='S')
    if isinstance(pdf_bytes, str):
        pdf_output = pdf_bytes.encode('latin1')
    else:
        pdf_output = pdf_bytes
        
    return pdf_output