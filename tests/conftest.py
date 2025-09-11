"""Pytest configuration file for the baby growth curves application."""

import pytest
import sys
import os
from unittest.mock import MagicMock, patch

# Add the parent directory to sys.path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock streamlit before importing any modules that use it
import streamlit as st
st.write = MagicMock()
st.line_chart = MagicMock()
st.plotly_chart = MagicMock()
st.session_state = MagicMock()

# Mock plotly
import plotly.graph_objects as go
go.Figure = MagicMock()
go.Scatter = MagicMock()

@pytest.fixture(autouse=True)
def setup_streamlit_mocks():
    """Set up streamlit mocks for all tests."""
    st.session_state.debug_mode = False
    st.session_state.display_mode = 'Percentiles'
    st.session_state.birth_ga_weeks = 32
    st.session_state.birth_ga_days = 0
    st.session_state.birth_date = None
    st.session_state.sex = 'M'
    yield