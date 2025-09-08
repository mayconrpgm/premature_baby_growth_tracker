"""Test script for Patient __str__ method."""

from patient import Patient
from datetime import datetime, date
import pandas as pd

# Create a patient
patient = Patient(
    name="Test Patient",
    birth_ga_weeks=30,
    birth_ga_days=5,
    birth_date=date(2023, 1, 15),
    sex="Female"
)

# Print the patient (should use __str__ method)
print("Patient with no measurements:")
print(patient)
print()

# Add some measurements
patient.add_measurement(
    pma_weeks=32,
    pma_days=2,
    measurement_date=date(2023, 2, 1),
    weight=1.5,
    length=40.0,
    hc=28.5
)

patient.add_measurement(
    pma_weeks=34,
    pma_days=3,
    measurement_date=date(2023, 2, 15),
    weight=1.8,
    length=42.5,
    hc=30.0
)

# Add one more measurement to better demonstrate the table
patient.add_measurement(
    pma_weeks=36,
    pma_days=1,
    measurement_date=date(2023, 3, 1),
    weight=2.1,
    length=44.5,
    hc=31.2
)

# Print the patient again with measurements (now with ASCII table)
print("Patient with measurements (ASCII table):")
print(patient)