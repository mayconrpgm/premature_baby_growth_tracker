"""Patient class for managing patient information and measurements."""

import pandas as pd
from datetime import datetime
from util import pma_to_decimal_weeks, decimal_weeks_to_pma, calculate_pma_from_date

class Patient:
    """Class representing a patient with growth measurements."""
    
    def __init__(self, name="", birth_ga_weeks=32, birth_ga_days=0, birth_date=None, sex="Male"):
        """Initialize a patient with basic information.
        
        Args:
            name: String name of the patient
            birth_ga_weeks: Integer weeks of gestational age at birth
            birth_ga_days: Integer days of gestational age at birth
            birth_date: Date object of birth date
            sex: String sex of the patient ('Male' or 'Female')
        """
        self.name = name
        self.birth_ga_weeks = birth_ga_weeks
        self.birth_ga_days = birth_ga_days
        self.birth_date = birth_date or datetime.now().date()
        self.sex = sex
        self.measurements = pd.DataFrame(columns=[
            'pma_weeks', 'pma_days', 'measurement_date', 
            'weight', 'length', 'hc'
        ])
    
    def __str__(self) -> str:
        """Return a string representation of the patient.
        
        Returns:
            String representation with patient info and measurement summary as ASCII table
        """
        measurement_count = len(self.measurements)
        measurements_summary = ""
        
        if measurement_count > 0:
            latest = self.measurements.iloc[-1]
            measurements_summary = f"\nMeasurements: {measurement_count} total, latest at {latest['pma_weeks']}w{latest['pma_days']}d PMA"
            
            # Create ASCII table for measurements
            table = ["\n┌─────────────┬──────────┬──────────┬──────────┬────────────────┐"]
            table.append("│ PMA         │ Weight   │ Length   │ HC       │ Date           │")
            table.append("├─────────────┼──────────┼──────────┼──────────┼────────────────┤")
            
            for _, row in self.measurements.iterrows():
                pma = f"{row['pma_weeks']}w{row['pma_days']}d"
                weight = f"{row['weight']:.2f}" if pd.notna(row['weight']) else "--"
                length = f"{row['length']:.1f}" if pd.notna(row['length']) else "--"
                hc = f"{row['hc']:.1f}" if pd.notna(row['hc']) else "--"
                date = row['measurement_date'].strftime("%Y-%m-%d") if pd.notna(row['measurement_date']) else "--"
                
                table.append(f"│ {pma:<11} │ {weight:<8} │ {length:<8} │ {hc:<8} │ {date:<14} │")
            
            table.append("└─────────────┴──────────┴──────────┴──────────┴────────────────┘")
            measurements_summary += "\n" + "\n".join(table)

        return f"Patient: {self.name or 'Unnamed'}, {self.sex}\nBirth GA: {self.birth_ga_weeks}w{self.birth_ga_days}d, DOB: {self.birth_date}{measurements_summary}"
    
    @property
    def birth_ga_decimal(self):
        """Get birth gestational age in decimal weeks."""
        return pma_to_decimal_weeks(self.birth_ga_weeks, self.birth_ga_days)
    
    def add_measurement(self, pma_weeks=None, pma_days=None, measurement_date=None, 
                       weight=None, length=None, hc=None, use_date_for_pma=False):
        """Add a new measurement for the patient.
        
        Args:
            pma_weeks: Integer weeks of postmenstrual age
            pma_days: Integer days of postmenstrual age
            measurement_date: Date object of measurement date
            weight: Float weight in kg
            length: Float length in cm
            hc: Float head circumference in cm
            use_date_for_pma: Boolean to calculate PMA from date
            
        Returns:
            Boolean indicating success
        """
        if not any([weight, length, hc]):
            return False
        
        # Calculate PMA from date if requested
        if use_date_for_pma and measurement_date:
            birth_date_dt = datetime.combine(self.birth_date, datetime.min.time())
            meas_date_dt = datetime.combine(measurement_date, datetime.min.time())
            pma_decimal = calculate_pma_from_date(self.birth_ga_decimal, birth_date_dt, meas_date_dt)
            pma_weeks, pma_days = decimal_weeks_to_pma(pma_decimal)
        
        new_data = {
            'pma_weeks': pma_weeks,
            'pma_days': pma_days,
            'measurement_date': measurement_date,
            'weight': weight if weight else None,
            'length': length if length else None,
            'hc': hc if hc else None
        }
        
        self.measurements = pd.concat([self.measurements, pd.DataFrame([new_data])], ignore_index=True)
        self.measurements = self.measurements.sort_values(by=['pma_weeks', 'pma_days']).reset_index(drop=True)
        return True
    
    def remove_measurement(self, index):
        """Remove a measurement by index.
        
        Args:
            index: Integer index of the measurement to remove
            
        Returns:
            Boolean indicating success
        """
        if index in self.measurements.index:
            self.measurements = self.measurements.drop(index=index).reset_index(drop=True)
            return True
        return False
    
    def get_export_filename(self):
        """Generate a filename for exporting patient data.
        
        Returns:
            String filename
        """
        return (f"{self.name or 'patient'}"
               f"_GA{self.birth_ga_weeks}w{self.birth_ga_days}d"
               f"_DOB{self.birth_date.strftime('%Y%m%d')}"
               f".csv")
    
    def to_dict(self):
        """Convert patient information to a dictionary.
        
        Returns:
            Dictionary of patient information
        """
        return {
            'name': self.name,
            'birth_ga_weeks': self.birth_ga_weeks,
            'birth_ga_days': self.birth_ga_days,
            'birth_date': self.birth_date,
            'sex': self.sex
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create a patient from a dictionary.
        
        Args:
            data: Dictionary containing patient information
            
        Returns:
            Patient object
        """
        return cls(
            name=data.get('name', ''),
            birth_ga_weeks=data.get('birth_ga_weeks', 32),
            birth_ga_days=data.get('birth_ga_days', 0),
            birth_date=data.get('birth_date'),
            sex=data.get('sex', 'Male')
        )
    
    def get_measurements_for_chart(self, metric=None):
        """Get patient measurements formatted for chart integration.
        
        Args:
            metric: String indicating which measurement to return ('weight', 'length', 'hc').
                   If None, returns all measurements.
                   
        Returns:
            DataFrame with measurements ready for chart integration.
            If metric is specified, returns only that metric's data.
        """
        # Create a copy to avoid modifying the original
        df = self.measurements.copy()
        
        # Add decimal PMA for easier plotting
        df['pma_decimal'] = df.apply(
            lambda row: pma_to_decimal_weeks(row['pma_weeks'], row['pma_days']), 
            axis=1
        )
        
        # If a specific metric is requested, return only that data
        if metric and metric in ['weight', 'length', 'hc']:
            # Return only rows where the metric is not null
            return df[df[metric].notna()][["pma_decimal", "pma_weeks", "pma_days", "measurement_date", metric]]
        
        # Otherwise return all measurements
        return df
    
    def get_measurement_series(self, metric):
        """Get a specific measurement as a pandas Series indexed by PMA decimal.
        
        Args:
            metric: String indicating which measurement to return ('weight', 'length', 'hc')
            
        Returns:
            Series with the specified measurement indexed by PMA decimal
        """
        if metric not in ['weight', 'length', 'hc']:
            raise ValueError(f"Invalid metric: {metric}. Must be 'weight', 'length', or 'hc'.")
            
        # Get measurements and filter out null values for the specified metric
        df = self.get_measurements_for_chart(metric)
        
        # Return as a Series indexed by PMA decimal
        return pd.Series(df[metric].values, index=df['pma_decimal'])