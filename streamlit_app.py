import streamlit as st
from tabs.injuries_tab import render_injuries_tab
from tabs.todays_matchups_tab import render_todays_matchups_tab
from tabs.players_tab import render_players_tab
from tabs.simulator_tab import render_simulator_tab
from tabs.teams_tab import render_teams_tab
from tabs.single_player_tab import render_single_player_tab
from tabs.single_team_tab import render_single_team_tab
from tabs.chat_tab import render_chat_tab
import pandas as pd

from utils.scrape_todays_matchups import scrape_todays_matchups
from utils.scrape_daily_injuries import scrape_daily_injuries
from utils.scrape_player_game_stats import scrape_all_games_player_stats
from utils.scrape_games import scrape_all_games_team_stats
from utils.scrape_team_rosters import scrape_all_team_rosters

from utils.data_helpers.load_data import load_data

from pathlib import Path

MATCHUPS_FILE = Path("latest_data/matchups.csv")
INJURY_FILE = Path("latest_data/injuries.csv")
PLAYERS_FILE = Path("latest_data/players.csv")
TEAMS_FILE = Path("latest_data/games/team_stats.csv")

def save_csv(df: pd.DataFrame, file_path: Path):
    file_path.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(file_path, index=False)

# Set wide layout
st.set_page_config(page_title="NBA Analytics Dashboard", layout="wide")

if "active_tab" not in st.session_state:
    st.session_state.active_tab = 0

cols = st.columns([6.7,1])
with cols[0]:
    st.title("🏀 NBA Analytics Dashboard")
    df, last_updated = load_data(TEAMS_FILE)

    if last_updated:
        st.caption(f"📅 Data last updated: {last_updated.strftime('%Y-%m-%d %H:%M:%S')}")


with cols[1]:
    st.write(" ")
    st.write(" ")
    fetch_clicked = st.button("🔄 Fetch Data")
    if fetch_clicked:
        scrape_tasks = [
            ("Today's matchups", scrape_todays_matchups, MATCHUPS_FILE),
            ("Injury data", scrape_daily_injuries, INJURY_FILE),
            ("Player logs", scrape_all_games_player_stats, PLAYERS_FILE),
            ("Team logs", scrape_all_games_team_stats, TEAMS_FILE),
            ("Team rosters", scrape_all_team_rosters, None),
        ]

        for label, func, path in scrape_tasks:
            with st.spinner(f"Scraping {label}..."):
                try:
                    result = func() if func.__name__ != "scrape_all_team_rosters" else func("2025-26")

                    if isinstance(result, tuple):  # rosters returns (players, coaches)
                        players, coaches = result
                        players.to_csv("latest_data/team_rosters_players.csv", index=False)
                        coaches.to_csv("latest_data/team_rosters_coaches.csv", index=False)
                        st.success(f"✅ {label} saved ({len(players)} players, {len(coaches)} coaches)")
                    elif result is not None and not result.empty:
                        if path:
                            save_csv(result, path)
                        st.success(f"✅ {label} updated")
                    else:
                        st.error(f"No data returned for {label}")
                except Exception as e:
                    st.error(f"❌ Failed to fetch {label}: {e}")

        st.rerun()

        

tabs = st.tabs(["Matchups", "Injuries", "Players", "Single Player", "Teams", "Single Team", "Simulator", "Chat"])


for i, tab in enumerate(tabs):
    with tab:
        st.session_state.active_tab = i
        if st.session_state.active_tab == i:
            if i == 0:
                render_todays_matchups_tab()
            elif i == 1:
                render_injuries_tab()
            elif i == 2:
                render_players_tab()
            elif i == 3:
                render_single_player_tab()
            elif i == 4:
                render_teams_tab()
            elif i == 5:
                render_single_team_tab()
            elif i == 6:
                render_simulator_tab()
            elif i == 7:
                render_chat_tab()
