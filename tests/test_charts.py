"""Unit tests for chart creation functions."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add the parent directory to sys.path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock streamlit before importing charts
import streamlit as st
st.session_state = MagicMock()
st.session_state.debug_mode = False
st.session_state.display_mode = 'Percentiles'
st.session_state.birth_ga_weeks = 32
st.session_state.birth_ga_days = 0

# Now import the charts module
from charts import create_full_chart


class TestChartCreation:
    """Tests for chart creation functions."""
    
    def setup_method(self):
        """Set up test data before each test."""
        # Create a mock chart data DataFrame
        self.chart_data = pd.DataFrame({
            'ga': [24, 25, 26, 27, 28, 29, 30],
            'p3': [500, 550, 600, 650, 700, 750, 800],
            'p10': [550, 600, 650, 700, 750, 800, 850],
            'p50': [650, 700, 750, 800, 850, 900, 950],
            'p90': [750, 800, 850, 900, 950, 1000, 1050],
            'p97': [800, 850, 900, 950, 1000, 1050, 1100],
            'z-3': [450, 500, 550, 600, 650, 700, 750],
            'z-2': [500, 550, 600, 650, 700, 750, 800],
            'z-1': [550, 600, 650, 700, 750, 800, 850],
            'z0': [650, 700, 750, 800, 850, 900, 950],
            'z1': [750, 800, 850, 900, 950, 1000, 1050],
            'z2': [800, 850, 900, 950, 1000, 1050, 1100],
            'z3': [850, 900, 950, 1000, 1050, 1100, 1150]
        })
        
        # Create a mock patient data DataFrame
        self.patient_data = pd.DataFrame({
            'pma_weeks': [32, 33, 34],
            'pma_days': [0, 0, 0],
            'weight': [1500, 1600, 1700],
            'length': [40, 41, 42],
            'hc': [28, 29, 30],
            'measurement_date': [pd.Timestamp('2023-01-01'), pd.Timestamp('2023-01-08'), pd.Timestamp('2023-01-15')]
        })
        
        # Create a mock config
        self.config = {
            'title': 'Weight Chart',
            'y_axis_title': 'Weight (g)',
            'data_col': 'weight'
        }
    
    def test_create_full_chart_percentiles(self, mocker):
        """Test chart creation with percentile mode."""
        # Mock Plotly Figure and add_trace
        mock_figure = mocker.MagicMock()
        mocker.patch('plotly.graph_objects.Figure', return_value=mock_figure)
        
        # Set up session state
        st.session_state.display_mode = 'Percentiles'
        
        # Call the function
        result = create_full_chart(self.chart_data, self.config, 'weight', self.patient_data)
        
        # Check that the function was called and returned the mock figure
        assert result == mock_figure
        # Verify update_layout was called
        assert mock_figure.update_layout.called
    
    def test_create_full_chart_zscores(self, mocker):
        """Test chart creation with z-scores mode."""
        # Mock Plotly Figure and add_trace
        mock_figure = mocker.MagicMock()
        mocker.patch('plotly.graph_objects.Figure', return_value=mock_figure)
        
        # Set session state to z-scores mode
        st.session_state.display_mode = 'Z-Scores'
        
        # Call the function
        result = create_full_chart(self.chart_data, self.config, 'weight', self.patient_data)
        
        # Check that the function was called and returned the mock figure
        assert result == mock_figure
        # Verify update_layout was called
        assert mock_figure.update_layout.called
    
    def test_create_full_chart_no_patient_data(self, mocker):
        """Test chart creation without patient data."""
        # Mock Plotly Figure and add_trace
        mock_figure = mocker.MagicMock()
        mocker.patch('plotly.graph_objects.Figure', return_value=mock_figure)
        
        # Call the function without patient data
        result = create_full_chart(self.chart_data, self.config, 'weight', None)
        
        # Check that the function was called and returned the mock figure
        assert result == mock_figure
        # Verify update_layout was called
        assert mock_figure.update_layout.called