import streamlit as st
from datetime import datetime
import pandas as pd

def load_data(data_file):
    """Load player data from CSV file with caching."""
    if data_file.exists():
        try:
            df = pd.read_csv(data_file)
            last_updated = datetime.fromtimestamp(data_file.stat().st_mtime)
            return df, last_updated
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return pd.DataFrame(), None
    return pd.DataFrame(), None