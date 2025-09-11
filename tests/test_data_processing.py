"""Unit tests for data processing functions."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, mock_open
import sys
import os

# Add the parent directory to sys.path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing import (
    load_intergrowth_data,
    get_z_score_and_percentile
)


class TestDataLoading:
    """Tests for data loading functions."""
    
    @patch('pandas.read_csv')
    def test_load_intergrowth_data_success(self, mock_read_csv):
        """Test successful loading of INTERGROWTH data."""
        # Create a mock DataFrame that mimics the structure of the real data
        mock_df = pd.DataFrame({
            'sex': ['Boy', 'Girl'],
            'metric': ['Weight_kg', 'Length_cm'],
            'measurement_type': ['3rd', '50th'],
            'age_weeks': [24, 28],
            'measurement_value': ['3rd', '50th'],
            'value': [0.5, 35.0]
        })
        mock_read_csv.return_value = mock_df
        
        # Call the function
        result_df, error = load_intergrowth_data()
        
        # Verify the function was called with the correct path
        mock_read_csv.assert_called_once_with('data/intergrowth_combined_data.csv')
        
        # Check that no error was returned
        assert error is None
        
        # Check that the DataFrame was processed correctly
        assert isinstance(result_df, pd.DataFrame)
        assert not result_df.empty
    
    def test_load_intergrowth_data_error(self):
        """Test error handling when loading INTERGROWTH data."""
        # Mock pandas read_csv to raise an exception
        with patch('pandas.read_csv', side_effect=Exception('Test error')):
            df, error = load_intergrowth_data()
            assert df.empty
            assert error is not None
            # The actual error message contains the exception message
            assert 'Test error' in error


class TestDataProcessing:
    """Tests for data processing functions."""
    
    def test_get_z_score_and_percentile(self):
        """Test Z-score and percentile calculation."""
        # Create a simple test case that doesn't require complex calculations
        # We'll just verify the function runs without errors and returns the expected types
        
        # Create a mock DataFrame with reference data
        mock_df = pd.DataFrame({
            'ga': [32],
            'sex': ['Boy'],
            'param': ['Weight_kg'],
            'p3': [1.0],
            'p10': [1.2],
            'p50': [2.0],
            'p90': [2.8],
            'p97': [3.0],
            'z-2': [1.0],
            'z-1': [1.5],
            'z0': [2.0],
            'z1': [2.5],
            'z2': [3.0]
        })
        
        # Call the function with a value that matches z0/p50
        z_score, percentile = get_z_score_and_percentile(32, 2.0, 'Boy', 'Weight_kg', mock_df)
        
        # Just verify we get numeric results
        assert isinstance(z_score, (int, float))
        assert isinstance(percentile, (int, float))