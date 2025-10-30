import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime

DATA_FILE = Path("latest_data/injuries.csv")
PLAYER_FILE = Path("latest_data/players.csv")
TEAM_FILE = Path("latest_data/games/team_stats.csv")

STATUS_COLORS = {
    "Game Time Decision": "background-color: #ffae42; color: black",
    "Expected to be out until at least": "background-color: #fff14c; color: black",
    "Out for the season": "background-color: #ff4c4c; color: white",
    "Active / Probable": "background-color: #4caf50; color: white"
}

def highlight_status(val):
    """Apply color styling based on injury status."""
    for key in STATUS_COLORS:
        if key.lower() in str(val).lower():
            return STATUS_COLORS[key]
    return ""

def load_data():
    """Load injury data from CSV file."""
    if DATA_FILE.exists():
        try:
            df = pd.read_csv(DATA_FILE)
            last_updated = datetime.fromtimestamp(DATA_FILE.stat().st_mtime)
            return df, last_updated
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return pd.DataFrame(), None
    return pd.DataFrame(), None

def save_data(df):
    """Save injury data to CSV file."""
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_FILE, index=False)

def clean_injury_data(injury_df):
    """Clean and standardize injury data."""
    df = injury_df.copy()
    df['Name'] = df['Name'].str.strip().str.lower()
    df['Team'] = df['Team'].str.strip()
    df['Severity'] = df['Status'].apply(lambda s:
        'High' if 'out for the season' in str(s).lower() else
        'Medium' if 'expected to be out' in str(s).lower() else
        'Low'
    )
    
    def parse_date(d):
        try:
            return datetime.strptime(d.split(',')[1].strip(), '%b %d')
        except:
            return None
    
    df['Updated_dt'] = df['Updated'].apply(parse_date)
    return df

def augment_injury_data():
    """Merge injury data with player and team stats."""
    if not (DATA_FILE.exists() and PLAYER_FILE.exists() and TEAM_FILE.exists()):
        return pd.DataFrame(), pd.DataFrame()

    try:
        injuries = pd.read_csv(DATA_FILE)
        players = pd.read_csv(PLAYER_FILE)
        teams = pd.read_csv(TEAM_FILE)
    except Exception as e:
        st.error(f"Error reading data files: {e}")
        return pd.DataFrame(), pd.DataFrame()

    # Detect player name columns
    player_col = next((c for c in players.columns if 'player' in c.lower()), None)
    injury_col = next((c for c in injuries.columns if 'name' in c.lower()), None)
    
    if not player_col or not injury_col:
        st.warning("Could not find player name columns in one of the CSVs.")
        return pd.DataFrame(), pd.DataFrame()

    # Normalize player names for matching
    injuries['Player'] = injuries[injury_col].astype(str).str.strip().str.lower()
    players['Player'] = players[player_col].astype(str).str.strip().str.lower()

    merged = pd.merge(injuries, players, on='Player', how='left')

    # Calculate severity scores for team impact
    merged['SeverityScore'] = merged['Status'].apply(lambda s:
        3 if 'out for the season' in str(s).lower() else
        2 if 'expected to be out' in str(s).lower() else
        1
    )

    # Calculate team impact
    team_impact = (
        merged.groupby('Team', as_index=False)
        .agg(injured_players=('Player', 'count'), TotalImpact=('SeverityScore', 'sum'))
        .sort_values('TotalImpact', ascending=False)
    )

    return merged, team_impact

def render_injuries_tab():
    """Main rendering function for the injuries tab."""
    df, last_updated = load_data()

    # Header section with action buttons
    header_col, fetch_col, download_col, spacer_col, last_updated_col = st.columns([4, 2, 2, 0.2, 2.4])
    
    with header_col:
        st.markdown("### 📋 Daily Injuries Tracker")
    
    


    # Team, Position, and Injury Type filters
    filtered_df = pd.DataFrame()
    if not df.empty:
        col_filter, col_pos, col_injury = st.columns([1, 1, 1])
        
        with col_filter:
            teams = ["All Teams"] + sorted(df['Team'].unique().tolist())
            selected_team = st.selectbox("Filter by Team", teams, index=0)
        
        with col_pos:
            # Get unique positions if the column exists
            if 'POS' in df.columns:
                positions = ["All Positions"] + sorted(df['POS'].dropna().unique().tolist())
                selected_position = st.selectbox("Filter by Position", positions, index=0)
            else:
                selected_position = "All Positions"
        
        with col_injury:
            # Get unique injury types if the column exists
            if 'Injury' in df.columns:
                injury_types = ["All Injury Types"] + sorted(df['Injury'].dropna().unique().tolist())
                selected_injury = st.selectbox("Filter by Injury Type", injury_types, index=0)
            else:
                selected_injury = "All Injury Types"
        
        # Apply team filter
        filtered_df = df if selected_team == "All Teams" else df[df['Team'] == selected_team]
        
        # Apply position filter
        if selected_position != "All Positions" and 'POS' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['POS'] == selected_position]
        
        # Apply injury type filter
        if selected_injury != "All Injury Types" and 'Injury' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Injury'] == selected_injury]
        
        filtered_df = filtered_df.reset_index(drop=True)

    # Display injury data
    if not filtered_df.empty:
        # Use map instead of deprecated applymap
        if hasattr(filtered_df.style, 'map'):
            styled_df = filtered_df.style.map(highlight_status, subset=['Status'])
        else:
            styled_df = filtered_df.style.applymap(highlight_status, subset=['Status'])
        st.dataframe(styled_df, use_container_width=True)
    else:
        st.info("No data available. Click 'Fetch Latest Injuries' to load.")

    # Injury Impact Analysis section
    st.divider()
    st.markdown("### 📊 Injury Impact Analysis")
    
    # Add explanation of Total Impact scoring
    with st.expander("ℹ️ How is Total Impact calculated?"):
        st.markdown("""
        **Total Impact Score** measures the cumulative severity of injuries for each team:
        
        - 🟢 **Game Time Decision / Low Severity** = 1 point
        - 🟡 **Expected to be out** = 2 points  
        - 🔴 **Out for the season** = 3 points
        
        A team's Total Impact is the sum of all their injured players' severity scores.
        
        *Example: A team with 2 players out for the season (6 pts) and 1 game-time decision (1 pt) = 7 Total Impact*
        """)

    merged, team_impact = augment_injury_data()
    
    if not team_impact.empty:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Rename columns for clarity
            display_impact = team_impact.rename(columns={
                'injured_players': 'Injured Players',
                'TotalImpact': 'Total Impact Score'
            })
            st.dataframe(display_impact, use_container_width=True, hide_index=True)
        
        with col2:
            # Use plotly or altair for proper sorting, or reformat data
            # Streamlit bar_chart doesn't preserve order, so we'll use st.bar_chart with y-axis
            import plotly.express as px
            fig = px.bar(team_impact, x='Team', y='TotalImpact', 
                        labels={'TotalImpact': 'Total Impact Score', 'Team': 'Team'},
                        title='')
            fig.update_layout(showlegend=False, xaxis={'categoryorder':'total descending'})
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Teams with highest impact scores face the most severe injury challenges")
    else:
        st.info("Player or team data not available for impact analysis.")

# If running as standalone script
if __name__ == "__main__":
    render_injuries_tab()