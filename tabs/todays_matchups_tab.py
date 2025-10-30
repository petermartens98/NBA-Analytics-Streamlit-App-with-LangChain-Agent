import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
from datetime import datetime
from utils.scrape_todays_matchups import scrape_todays_matchups
from tabs.simulator_tab import (
    calculate_team_stats,
    run_monte_carlo,
    DEFAULT_SIMULATIONS
)

MATCHUPS_FILE = Path("latest_data/matchups.csv")
TEAM_STATS_FILE = Path("latest_data/games/team_stats.csv")
INJURIES_FILE = Path("latest_data/injuries.csv")

TEAM_NAME_MAP = {
    "Rockets": "Houston Rockets",
    "Cavaliers": "Cleveland Cavaliers",
    "Magic": "Orlando Magic",
    "Hawks": "Atlanta Hawks",
    "Kings": "Sacramento Kings",
    "Pacers": "Indiana Pacers",
    "Pelicans": "New Orleans Pelicans",
    "Trail Blazers": "Portland Trail Blazers",
    "Lakers": "Los Angeles Lakers",
    "Grizzlies": "Memphis Grizzlies",
    "Raptors": "Toronto Raptors",
    "Celtics": "Boston Celtics",
    "Pistons": "Detroit Pistons",
    "Nets": "Brooklyn Nets",
    "Bulls": "Chicago Bulls",
    "Mavericks": "Dallas Mavericks",
    "Nuggets": "Denver Nuggets",
    "Jazz": "Utah Jazz",
    "Timberwolves": "Minnesota Timberwolves",
    "Heat": "Miami Heat",
    "Hornets": "Charlotte Hornets",
    "Wizards": "Washington Wizards",
    "Clippers": "Los Angeles Clippers",
    "Spurs": "San Antonio Spurs",
    "Suns": "Phoenix Suns",
    "Thunder": "Oklahoma City Thunder",
    "Warriors": "Golden State Warriors",
    "Bucks": "Milwaukee Bucks",
    "Knicks": "New York Knicks",
    "Sixers": "Philadelphia 76ers"
}

# Mapping from injuries.csv team names to matchup abbreviations
INJURIES_TEAM_MAP = {
    "Atlanta": "Hawks",
    "Boston": "Celtics",
    "Brooklyn": "Nets",
    "Charlotte": "Hornets",
    "Chicago": "Bulls",
    "Cleveland": "Cavaliers",
    "Dallas": "Mavericks",
    "Denver": "Nuggets",
    "Detroit": "Pistons",
    "Golden State": "Warriors",
    "Houston": "Rockets",
    "Indiana": "Pacers",
    "LA Clippers": "Clippers",
    "LA Lakers": "Lakers",
    "Memphis": "Grizzlies",
    "Miami": "Heat",
    "Milwaukee": "Bucks",
    "Minnesota": "Timberwolves",
    "New Orleans": "Pelicans",
    "New York": "Knicks",
    "Oklahoma City": "Thunder",
    "Orlando": "Magic",
    "Philadelphia": "Sixers",
    "Phoenix": "Suns",
    "Portland": "Trail Blazers",
    "Sacramento": "Kings",
    "San Antonio": "Spurs",
    "Toronto": "Raptors",
    "Utah": "Jazz",
    "Washington": "Wizards"
}

def load_csv(file_path: Path):
    if file_path.exists():
        df = pd.read_csv(file_path)
        last_updated = datetime.fromtimestamp(file_path.stat().st_mtime)
        return df, last_updated
    return pd.DataFrame(), None

def save_csv(df: pd.DataFrame, file_path: Path):
    file_path.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(file_path, index=False)

def get_team_injuries(df_injuries, team_abbr):
    """Get injuries for a team using the injuries.csv format"""
    if df_injuries.empty:
        return pd.DataFrame()
    
    # Find the full team name from injuries CSV that maps to our abbreviation
    injury_team_name = None
    for inj_name, abbr in INJURIES_TEAM_MAP.items():
        if abbr == team_abbr:
            injury_team_name = inj_name
            break
    
    if injury_team_name:
        return df_injuries[df_injuries['Team'] == injury_team_name]
    
    return pd.DataFrame()

def render_todays_matchups_tab():
    df_matchups, last_updated = load_csv(MATCHUPS_FILE)
    df_teams_stats, _ = load_csv(TEAM_STATS_FILE)
    df_injuries, _ = load_csv(INJURIES_FILE)

    if df_matchups.empty:
        st.info("No games available. Click 'Fetch Matchups' to load.")
        return
    if df_teams_stats.empty:
        st.warning("Team stats data missing. Please fetch team stats first.")
        return

    # Show full dataframe at top
    st.markdown("### 🗓️ Today's Matchups")
    if "Date" in df_matchups.columns:
        df_display = df_matchups.drop(columns=["Date"], errors="ignore")
    else:
        df_display = df_matchups.copy()

    if "Time" in df_display.columns:
        df_display = df_display.set_index("Time")

    st.dataframe(df_display, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🎯 Game Predictions & Team Info")

    # One row per game
    for idx in range(len(df_matchups)):
        row = df_matchups.iloc[idx]
        away_team, home_team, game_time = row['Away'], row['Home'], row['Time']
        
        st.markdown(f"#### {away_team} @ {home_team} - *{game_time}*")
        
        away_full = TEAM_NAME_MAP.get(away_team, away_team)
        home_full = TEAM_NAME_MAP.get(home_team, home_team)
        
        # Get recent stats
        away_recent = df_teams_stats[df_teams_stats['TEAM_NAME'] == away_full].tail(5)
        home_recent = df_teams_stats[df_teams_stats['TEAM_NAME'] == home_full].tail(5)
        
        # Get injuries
        away_injuries = get_team_injuries(df_injuries, away_team)
        home_injuries = get_team_injuries(df_injuries, home_team)
        
        col1, col2 = st.columns(2)
        
        # Away Team
        with col1:
            st.markdown(f"**{away_team}**")
            if not away_recent.empty:
                st.markdown(f"📊 **Avg PTS:** {away_recent['PTS'].mean():.1f} | **FG%:** {away_recent['FG_PCT'].mean()*100:.1f}% | **Record:** {(away_recent['WL']=='W').sum()}-{(away_recent['WL']=='L').sum()}")
            
            if not away_injuries.empty:
                st.markdown("🏥 **Injuries:**")
                for _, inj in away_injuries.iterrows():
                    status_emoji = "🟡" if "Game Time" in inj['Status'] else "🔴"
                    st.caption(f"{status_emoji} {inj['Name']} ({inj['POS']}) - {inj['Injury']} - {inj['Status']}")
        
        # Home Team
        with col2:
            st.markdown(f"**{home_team}**")
            if not home_recent.empty:
                st.markdown(f"📊 **Avg PTS:** {home_recent['PTS'].mean():.1f} | **FG%:** {home_recent['FG_PCT'].mean()*100:.1f}% | **Record:** {(home_recent['WL']=='W').sum()}-{(home_recent['WL']=='L').sum()}")
            
            if not home_injuries.empty:
                st.markdown("🏥 **Injuries:**")
                for _, inj in home_injuries.iterrows():
                    status_emoji = "🟡" if "Game Time" in inj['Status'] else "🔴"
                    st.caption(f"{status_emoji} {inj['Name']} ({inj['POS']}) - {inj['Injury']} - {inj['Status']}")
        
        # Monte Carlo Simulation
        away_stats = calculate_team_stats(df_teams_stats, away_full)
        home_stats = calculate_team_stats(df_teams_stats, home_full)
        
        if away_stats and home_stats:
            with st.spinner("Running simulation..."):
                result = run_monte_carlo(home_stats, away_stats)
            
            col_a, col_b, col_c = st.columns([1, 1, 2])
            with col_a:
                st.metric(f"{home_team} Win %", f"{result['team_a_win_pct']:.1f}%")
                st.caption(f"Pred: {result['avg_team_a_score']:.1f} pts")
            with col_b:
                st.metric(f"{away_team} Win %", f"{result['team_b_win_pct']:.1f}%")
                st.caption(f"Pred: {result['avg_team_b_score']:.1f} pts")
            with col_c:
                # Score Distribution
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=result['team_a_scores'],
                    name=home_team,
                    opacity=0.7,
                    marker_color='#4ECDC4',
                    nbinsx=20
                ))
                fig.add_trace(go.Histogram(
                    x=result['team_b_scores'],
                    name=away_team,
                    opacity=0.7,
                    marker_color='#FF6B6B',
                    nbinsx=20
                ))
                fig.update_layout(
                    barmode='overlay',
                    height=240,
                    margin=dict(l=10, r=10, t=10, b=10),
                    xaxis_title='Points',
                    yaxis_title='Frequency',
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="top", y=1.1, xanchor="center", x=0.5)
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Cannot run simulation - stats missing")
        
        st.divider()