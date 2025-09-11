# INTERGROWTH-21st Preterm Growth Tracker

## About the Application

The INTERGROWTH-21st Preterm Growth Tracker is a clinical support tool designed to help healthcare professionals monitor the growth of preterm infants using the internationally recognized INTERGROWTH-21st standards. This application provides visual growth charts and data management tools to track a preterm baby's development over time.

## What is a Preterm Baby?

A preterm baby (or premature baby) is one born before 37 completed weeks of gestation. Preterm births are categorized by gestational age:

- **Extremely preterm**: Less than 28 weeks
- **Very preterm**: 28 to less than 32 weeks
- **Moderate to late preterm**: 32 to less than 37 weeks

Preterm birth occurs in about 10% of all pregnancies worldwide and is the leading cause of death in children under 5 years of age.

## Why Monitor Preterm Baby Growth?

Monitoring the growth of preterm infants is crucial for several reasons:

1. **Early Detection of Growth Problems**: Regular monitoring helps identify growth faltering or excessive weight gain early, allowing for timely interventions.

2. **Nutritional Management**: Growth data guides nutritional interventions, helping healthcare providers adjust feeding strategies to optimize growth.

3. **Neurodevelopmental Outcomes**: Growth patterns in early life, particularly head circumference growth, are associated with neurodevelopmental outcomes.

4. **Individualized Care**: Growth tracking allows for personalized care plans tailored to each infant's specific needs.

5. **Long-term Health**: Appropriate growth in early life is associated with better health outcomes in childhood and adulthood.

The INTERGROWTH-21st standards used in this application are based on a large international study and provide a standardized way to assess preterm infant growth across different populations.

## Features

- Track weight, length, and head circumference measurements
- Visualize growth on percentile or z-score charts
- Calculate postmenstrual age (PMA) automatically
- Import and export patient data
- Generate PDF reports for clinical documentation
- Compare measurements against international standards

## Installation and Setup

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Installation Steps

1. Clone the repository:
   ```
   git clone https://github.com/mayconrpgm/preterm_baby_growth_tracker.git
   cd preterm_baby_growth_tracker
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Run the application:
   ```
   streamlit run app.py
   ```

5. Open your web browser and navigate to the URL displayed in the terminal (typically http://localhost:8501)

## How to Use

1. **Enter Patient Information**:
   - Fill in the patient's birth gestational age (GA), birth date, and sex in the sidebar.

2. **Add Measurements**:
   - Toggle "Calculate PMA from Date" for automatic Postmenstrual Age calculation, or enter PMA manually.
   - Input one or more measurements (Weight, Length, Head Circumference).
   - Click "Add Measurement".

3. **Import/Export Data**:
   - Import: Upload a previously exported CSV file to restore patient data.
   - Export: Download data as CSV or generate a PDF report.

4. **View Charts**:
   - Charts update automatically with each new data point.
   - Switch between Percentiles and Z-Scores views.

5. **Manage Data**:
   - View all measurements in the table below the charts.
   - Remove individual measurements if needed.

## Data Sources

This application uses growth standards from the INTERGROWTH-21st Project, an international, multicenter, population-based project conducted by the University of Oxford. The standards are based on data collected from healthy pregnant women and their babies across eight countries.

For more information, visit: [INTERGROWTH-21st Preterm Growth Standards](https://intergrowth21.ndog.ox.ac.uk/preterm/)

## License

This project is open source and available under the MIT License.

## Developer

Developed by Maycon Queiros

GitHub: [https://github.com/mayconrpgm/preterm_baby_growth_tracker](https://github.com/mayconrpgm/preterm_baby_growth_tracker)