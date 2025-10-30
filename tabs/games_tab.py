# players_tab.py
import streamlit as st
from pathlib import Path
from datetime import datetime
import pandas as pd
from utils.scrape_games import scrape_all_games_team_stats

data_file = Path("latest_data/games/team_stats.csv")

def load_data():
    if data_file.exists():
        df = pd.read_csv(data_file)
        last_updated = datetime.fromtimestamp(data_file.stat().st_mtime)
        return df, last_updated
    return pd.DataFrame(), None

def save_data(df):
    data_file.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(data_file, index=False)

def render_games_tab():
    df, last_updated = load_data()

    # Header row with buttons
    header_col, fetch_col, download_col, spacer_col, last_updated_col = st.columns([4, 2, 2, 0.2, 2.4])
    
    with header_col:
        st.markdown("### 🏀 NBA Game Logs")
    
    with fetch_col:
        st.write(" ")
        fetch_clicked = st.button("🔄 Fetch Data")
        if fetch_clicked:
            with st.spinner("Scraping game logs..."):
                try:
                    df = scrape_all_games_team_stats()
                    save_data(df)
                    st.success("Player data fetched and saved successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to fetch player data: {e}")

    
    with last_updated_col:
        st.write(" ")
        st.markdown(
            f"<div style='text-align: right; font-size: 14px; color: gray;'>"
            f"Last retrieved: {last_updated.strftime('%Y-%m-%d %H:%M:%S') if last_updated else 'Never'}"
            f"</div>", 
            unsafe_allow_html=True
        )

    # Display table full width
    if not df.empty:
        st.dataframe(df.reset_index(drop=True), use_container_width=True)
    else:
        st.info("No player data available. Click 'Fetch Player Data' to load.")
