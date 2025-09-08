"""Utility functions for the growth chart application."""

from datetime import datetime, timedelta

def pma_to_decimal_weeks(weeks, days):
    """Converts PMA from (weeks, days) to decimal weeks.
    
    Args:
        weeks: Integer number of weeks
        days: Integer number of days
        
    Returns:
        Float decimal weeks
    """
    return weeks + days / 7

def decimal_weeks_to_pma(decimal_weeks):
    """Converts decimal weeks to PMA (weeks, days).
    
    Args:
        decimal_weeks: Float decimal weeks
        
    Returns:
        Tuple of (weeks, days)
    """
    weeks = int(decimal_weeks)
    days = int(round((decimal_weeks - weeks) * 7))
    return weeks, days

def calculate_chronological_age_days(birth_date, measurement_date):
    """Calculates chronological age in days.
    
    Args:
        birth_date: Datetime object of birth date
        measurement_date: Datetime object of measurement date
        
    Returns:
        Integer days of age
    """
    if isinstance(birth_date, datetime) and isinstance(measurement_date, datetime):
        return (measurement_date - birth_date).days
    return 0

def format_age(days):
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

def calculate_corrected_age_days(birth_date, measurement_date, birth_ga_weeks, birth_ga_days):
    """Calculates corrected age in days.
    
    Args:
        birth_date: Datetime object of birth date
        measurement_date: Datetime object of measurement date
        birth_ga_weeks: Integer weeks of gestational age at birth
        birth_ga_days: Integer days of gestational age at birth
        
    Returns:
        Integer days of corrected age
    """
    if isinstance(birth_date, datetime) and isinstance(measurement_date, datetime):
        chronological_days = (measurement_date - birth_date).days
        # Calculate weeks to full term (40 weeks)
        weeks_to_term = 40 - (birth_ga_weeks + birth_ga_days/7)
        # Convert to days and subtract from chronological age
        corrected_days = chronological_days - (weeks_to_term * 7)
        return corrected_days if corrected_days >= 0 else 0
    return None

def calculate_pma_from_date(birth_ga_decimal, birth_date, measurement_date):
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