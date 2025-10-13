"""Utility functions for the growth chart application."""

from datetime import datetime, timedelta
import streamlit as st
from typing import Any, Dict, List, Tuple, Optional, Union

def debug_print(*args: Any, **kwargs: Any) -> None:
    """Prints debug information if debug mode is enabled.
    
    Args:
        *args: Variable length argument list to print
        **kwargs: Arbitrary keyword arguments to pass to st.write
    """
    if st.session_state.debug_mode:
        st.write(*args, **kwargs)

def pma_to_decimal_weeks(weeks: int, days: int) -> float:
    """Converts PMA from (weeks, days) to decimal weeks.
    
    Args:
        weeks: Integer number of weeks
        days: Integer number of days
        
    Returns:
        Float decimal weeks
    """
    return weeks + days / 7

def decimal_weeks_to_pma(decimal_weeks: float) -> Tuple[int, int]:
    """Converts decimal weeks to PMA (weeks, days).
    
    Args:
        decimal_weeks: Float decimal weeks
        
    Returns:
        Tuple of (weeks, days)
    """
    weeks = int(decimal_weeks)
    days = int(round((decimal_weeks - weeks) * 7))
    return weeks, days

def calculate_chronological_age_days(birth_date: Union[datetime, datetime.date], 
                                    measurement_date: Union[datetime, datetime.date]) -> int:
    """Calculates chronological age in days.
    
    Args:
        birth_date: Datetime object of birth date
        measurement_date: Datetime object of measurement date
        
    Returns:
        Integer days of age
    """
    if isinstance(birth_date, (datetime, datetime.date)) and isinstance(measurement_date, (datetime, datetime.date)):
        return (measurement_date - birth_date).days
    return 0

def format_age(days: Union[int, float]) -> str:
    """Formats age in months and days for display.
    
    Args:
        days: Integer days of age
        
    Returns:
        String formatted age
    """
    if isinstance(days, (int, float)) and days >= 0:
        months = int(days // 30)
        remaining_days = int(days % 30)
        return f"{months}m {remaining_days}d"
    return "N/A"

def calculate_corrected_age_days(birth_date: Union[datetime, datetime.date],
                               measurement_date: Union[datetime, datetime.date],
                               birth_ga_weeks: int,
                               birth_ga_days: int) -> Optional[float]:
    """Calculates corrected age in days.
    
    Args:
        birth_date: Datetime object of birth date
        measurement_date: Datetime object of measurement date
        birth_ga_weeks: Integer weeks of gestational age at birth
        birth_ga_days: Integer days of gestational age at birth
        
    Returns:
        Float days of corrected age or None if calculation not possible
    """
    if isinstance(birth_date, (datetime, datetime.date)) and isinstance(measurement_date, (datetime, datetime.date)):
        chronological_days = (measurement_date - birth_date).days
        # Calculate weeks to full term (40 weeks)
        weeks_to_term = 40 - (birth_ga_weeks + birth_ga_days/7)
        # Convert to days and subtract from chronological age
        corrected_days = chronological_days - (weeks_to_term * 7)
        return corrected_days if corrected_days >= 0 else 0
    return None

def calculate_pma_from_date(birth_ga_decimal: float, 
                          birth_date: Union[datetime, datetime.date], 
                          measurement_date: Union[datetime, datetime.date]) -> float:
    """Calculates PMA in decimal weeks from dates.
    
    Args:
        birth_ga_decimal: Float decimal weeks of gestational age at birth
        birth_date: Datetime object of birth date
        measurement_date: Datetime object of measurement date
        
    Returns:
        Float decimal weeks of postmenstrual age
    """
    chrono_age_days = calculate_chronological_age_days(birth_date, measurement_date)
    return birth_ga_decimal + (chrono_age_days / 7)


# Check if PDF export is available (Plotly static image export)
def is_pdf_export_available():
    try:
        import plotly.graph_objects as go
        # Minimal test figure to probe image export capability
        fig = go.Figure()
        fig.add_scatter(x=[0, 1], y=[0, 1], mode='lines')
        _ = fig.to_image(format="png", width=2, height=2)
        return True
    except Exception:
        return False