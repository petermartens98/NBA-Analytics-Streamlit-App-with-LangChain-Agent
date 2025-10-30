import streamlit as st
import pandas as pd
import numpy as np
from utils.data_helpers.load_data import load_data
from utils.scrape_player_career_stats import get_player_career_stats
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

DATA_FILE = Path("latest_data/players.csv")

# Constants
NUMERIC_COLS = [
    'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG2M', 'FG2A', 'FG2_PTS', 
    'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 
    'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 
    'PTS', 'PLUS_MINUS', 'FANTASY_PTS', 'FG3_PTS'
]

STAT_OPTIONS = [
    'PTS', 'REB', 'AST', 'PLUS_MINUS', 
    'FG_PCT', 'FG3_PCT', 'FT_PCT', 'FANTASY_PTS'
]


@st.cache_data(ttl=3600)
def load_career_stats(player_id):
    """Load and cache career statistics for a player."""
    try:
        per_game_stats = get_player_career_stats(player_id, per_mode36="PerGame")
        totals_stats = get_player_career_stats(player_id, per_mode36="Totals")
        
        # Check if data was successfully fetched
        if not per_game_stats or not totals_stats:
            return None, None
        
        # Handle both list and dict returns
        if isinstance(per_game_stats, list):
            # Convert list to dict with indices
            per_game_dict = {f"dataset_{i}": df for i, df in enumerate(per_game_stats)}
            # Try to find the SeasonTotalsRegularSeason by looking at dataframes
            per_game_stats = {"SeasonTotalsRegularSeason": per_game_stats[0] if len(per_game_stats) > 0 else pd.DataFrame()}
        
        if isinstance(totals_stats, list):
            totals_stats = {"SeasonTotalsRegularSeason": totals_stats[0] if len(totals_stats) > 0 else pd.DataFrame()}
            
        return per_game_stats, totals_stats
    except Exception as e:
        st.error(f"Error loading career stats: {e}")
        return None, None


def format_dataframe(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    """Apply formatting and styling to the dataframe."""
    format_dict = {
        col: "{:.1f}" if 'PCT' not in col else "{:.2%}" 
        for col in numeric_cols
    }
    return df.style.format(format_dict).background_gradient(
        subset=numeric_cols, 
        cmap='YlGnBu'
    )


def create_career_progression_chart(career_df: pd.DataFrame, player_name: str, stat: str = 'PTS') -> go.Figure:
    """Create a chart showing career progression over seasons."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=career_df['SEASON_ID'],
        y=career_df[stat],
        mode='lines+markers',
        name=stat,
        marker=dict(size=10, color='#4ECDC4'),
        line=dict(width=3, color='#4ECDC4'),
        hovertemplate=f'Season: %{{x}}<br>{stat}: %{{y:.1f}}<extra></extra>'
    ))
    
    # Add trend line
    if len(career_df) > 1:
        z = np.polyfit(range(len(career_df)), career_df[stat], 1)
        p = np.poly1d(z)
        fig.add_trace(go.Scatter(
            x=career_df['SEASON_ID'],
            y=p(range(len(career_df))),
            mode='lines',
            name='Trend',
            line=dict(width=2, color='orange', dash='dash'),
            hovertemplate='Trend<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(text=f"{player_name} - Career {stat} Progression", x=0.5, xanchor='center'),
        xaxis_title="Season",
        yaxis_title=stat,
        template="plotly_white",
        height=400,
        hovermode="x unified",
        margin=dict(t=60, b=60, l=60, r=60)
    )
    
    return fig


def create_career_comparison_chart(career_df: pd.DataFrame, player_name: str) -> go.Figure:
    """Create a chart comparing key stats across career."""
    stats_to_compare = ['PTS', 'REB', 'AST', 'STL', 'BLK']
    
    fig = go.Figure()
    
    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#F38181', '#AA96DA']
    
    for stat, color in zip(stats_to_compare, colors):
        if stat in career_df.columns:
            fig.add_trace(go.Scatter(
                x=career_df['SEASON_ID'],
                y=career_df[stat],
                mode='lines+markers',
                name=stat,
                marker=dict(size=8, color=color),
                line=dict(width=2, color=color),
                hovertemplate=f'{stat}: %{{y:.1f}}<extra></extra>'
            ))
    
    fig.update_layout(
        title=dict(text=f"{player_name} - Career Stats Comparison", x=0.5, xanchor='center'),
        xaxis_title="Season",
        yaxis_title="Value",
        template="plotly_white",
        height=450,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=80, b=60, l=60, r=60)
    )
    
    return fig


def create_trend_chart(player_df: pd.DataFrame, player_name: str, 
                       selected_stat: str, rolling_window: int) -> go.Figure:
    """Create an interactive trend chart with rolling average."""
    player_df = player_df.copy()
    player_df[f"{selected_stat}_MA"] = (
        player_df[selected_stat].rolling(rolling_window, min_periods=1).mean()
    )
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=player_df['GAME_DATE'],
        y=player_df[selected_stat],
        mode='markers+lines',
        name='Game Value',
        marker=dict(size=8, color='dodgerblue', opacity=0.7),
        line=dict(width=1.5, color='lightblue', dash='dot'),
        hovertemplate=f"{selected_stat}: %{{y:.2f}}<br>Date: %{{x|%Y-%m-%d}}<extra></extra>"
    ))
    
    fig.add_trace(go.Scatter(
        x=player_df['GAME_DATE'],
        y=player_df[f"{selected_stat}_MA"],
        mode='lines',
        name=f'{rolling_window}-Game Avg',
        line=dict(width=3, color='orange'),
        hovertemplate=f"{rolling_window}-Game Avg: %{{y:.2f}}<br>Date: %{{x|%Y-%m-%d}}<extra></extra>"
    ))
    
    fig.update_layout(
        title=dict(text=f"{player_name} - {selected_stat} Performance", x=0.5, xanchor='center'),
        xaxis_title="Game Date",
        yaxis_title=selected_stat,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500,
        margin=dict(t=80, b=60, l=60, r=60)
    )
    
    return fig


def create_shooting_breakdown_chart(player_df: pd.DataFrame, player_name: str) -> go.Figure:
    """Create a stacked bar chart showing shooting breakdown by game."""
    player_df = player_df.copy()
    
    player_df['2PT_PTS'] = player_df['FG2M'] * 2
    player_df['3PT_PTS'] = player_df['FG3M'] * 3
    player_df['FT_PTS'] = player_df['FTM']
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Free Throws',
        x=player_df['GAME_DATE'],
        y=player_df['FT_PTS'],
        marker_color='#FFA07A',
        hovertemplate='FT: %{y} pts<br>%{text}<extra></extra>',
        text=[f"{ftm}/{fta}" for ftm, fta in zip(player_df['FTM'], player_df['FTA'])]
    ))
    
    fig.add_trace(go.Bar(
        name='2-Pointers',
        x=player_df['GAME_DATE'],
        y=player_df['2PT_PTS'],
        marker_color='#1E88E5',
        hovertemplate='2PT: %{y} pts<br>%{text}<extra></extra>',
        text=[f"{fg2m}/{fg2a}" for fg2m, fg2a in zip(player_df['FG2M'], player_df['FG2A'])]
    ))
    
    fig.add_trace(go.Bar(
        name='3-Pointers',
        x=player_df['GAME_DATE'],
        y=player_df['3PT_PTS'],
        marker_color='#43A047',
        hovertemplate='3PT: %{y} pts<br>%{text}<extra></extra>',
        text=[f"{fg3m}/{fg3a}" for fg3m, fg3a in zip(player_df['FG3M'], player_df['FG3A'])]
    ))
    
    fig.update_layout(
        barmode='stack',
        title=dict(text=f"{player_name} - Points Breakdown by Source", x=0.5, xanchor='center'),
        xaxis_title="Game Date",
        yaxis_title="Points",
        template="plotly_white",
        height=450,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=80, b=60, l=60, r=60)
    )
    
    return fig


def create_shot_distribution_chart(player_df: pd.DataFrame, player_name: str) -> go.Figure:
    """Create a chart showing shot selection and efficiency."""
    avg_stats = {
        '2PT Attempts': player_df['FG2A'].mean(),
        '3PT Attempts': player_df['FG3A'].mean(),
        'FT Attempts': player_df['FTA'].mean()
    }
    
    efficiency_stats = {
        '2PT': player_df['FG2M'].sum() / player_df['FG2A'].sum() if player_df['FG2A'].sum() > 0 else 0,
        '3PT': player_df['FG3M'].sum() / player_df['FG3A'].sum() if player_df['FG3A'].sum() > 0 else 0,
        'FT': player_df['FTM'].sum() / player_df['FTA'].sum() if player_df['FTA'].sum() > 0 else 0
    }
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Avg Attempts',
        x=list(avg_stats.keys()),
        y=list(avg_stats.values()),
        marker_color=['#4ECDC4', '#95E1D3', '#FF6B6B'],
        text=[f"{v:.1f}" for v in avg_stats.values()],
        textposition='inside',
        textfont=dict(color='white', size=12),
        yaxis='y',
        hovertemplate='%{x}<br>Avg: %{y:.1f} attempts<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        name='Efficiency %',
        x=list(efficiency_stats.keys()),
        y=[v * 100 for v in efficiency_stats.values()],
        mode='lines+markers+text',
        marker=dict(size=12, color='orange', symbol='diamond'),
        line=dict(width=3, color='orange'),
        text=[f"{v*100:.1f}%" for v in efficiency_stats.values()],
        textposition='top center',
        textfont=dict(size=11),
        yaxis='y2',
        hovertemplate='%{x}<br>Efficiency: %{y:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text=f"{player_name} - Shot Selection & Efficiency", x=0.5, xanchor='center'),
        xaxis_title="Shot Type",
        yaxis=dict(title="Avg Attempts per Game", side='left'),
        yaxis2=dict(title="Shooting %", side='right', overlaying='y', range=[0, 110]),
        template="plotly_white",
        height=450,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=80, b=60, l=60, r=60)
    )
    
    return fig


def create_impact_metrics_chart(player_df: pd.DataFrame, player_name: str) -> go.Figure:
    """Create a radar chart showing overall player impact."""
    metrics = {
        'Scoring': (player_df['PTS'].mean() / 40) * 100,
        'Playmaking': (player_df['AST'].mean() / 12) * 100,
        'Rebounding': (player_df['REB'].mean() / 15) * 100,
        'Defense': ((player_df['STL'].mean() + player_df['BLK'].mean()) / 5) * 100,
        'Efficiency': (player_df['FG_PCT'].mean()) * 100,
        'Impact': ((player_df['PLUS_MINUS'].mean() + 20) / 40) * 100
    }
    
    categories = list(metrics.keys())
    values = list(metrics.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(78, 205, 196, 0.3)',
        line=dict(color='rgb(78, 205, 196)', width=2),
        name=player_name,
        hovertemplate='%{theta}<br>Score: %{r:.1f}/100<extra></extra>'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], ticksuffix=''),
            bgcolor='rgba(240, 240, 240, 0.5)'
        ),
        title=dict(text=f"{player_name} - Overall Impact Profile", x=0.5, xanchor='center'),
        template="plotly_white",
        height=500,
        showlegend=False,
        margin=dict(t=80, b=60, l=80, r=80)
    )
    
    return fig


def create_consistency_chart(player_df: pd.DataFrame, player_name: str) -> go.Figure:
    """Create a box plot showing consistency across key stats."""
    stats_to_plot = ['PTS', 'REB', 'AST', 'FG_PCT', 'PLUS_MINUS']
    
    fig = go.Figure()
    
    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#F38181', '#AA96DA']
    
    for stat, color in zip(stats_to_plot, colors):
        if stat in player_df.columns:
            values = player_df[stat].dropna()
            fig.add_trace(go.Box(
                y=values * (100 if stat == 'FG_PCT' else 1),
                name=stat if stat != 'FG_PCT' else 'FG%',
                marker_color=color,
                boxmean='sd',
                hovertemplate='%{y:.1f}<extra></extra>'
            ))
    
    fig.update_layout(
        title=dict(text=f"{player_name} - Performance Consistency", x=0.5, xanchor='center'),
        yaxis_title="Value",
        template="plotly_white",
        height=450,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=80, b=60, l=60, r=60)
    )
    
    return fig


def create_win_loss_comparison(player_df: pd.DataFrame, player_name: str):
    """Compare player performance in wins vs losses."""
    wins = player_df[player_df['WL'] == 'W']
    losses = player_df[player_df['WL'] == 'L']
    
    if len(wins) == 0 or len(losses) == 0:
        return None
    
    stats_compare = ['PTS', 'REB', 'AST', 'FG_PCT', 'PLUS_MINUS']
    
    win_avgs = [wins[stat].mean() * (100 if stat == 'FG_PCT' else 1) for stat in stats_compare]
    loss_avgs = [losses[stat].mean() * (100 if stat == 'FG_PCT' else 1) for stat in stats_compare]
    
    max_val = max(max(win_avgs), max(loss_avgs))
    y_range = [0, max_val * 1.2]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Wins',
        x=[s if s != 'FG_PCT' else 'FG%' for s in stats_compare],
        y=win_avgs,
        marker_color='#52B788',
        text=[f"{v:.1f}" for v in win_avgs],
        textposition='inside',
        textfont=dict(color='white', size=11)
    ))
    
    fig.add_trace(go.Bar(
        name='Losses',
        x=[s if s != 'FG_PCT' else 'FG%' for s in stats_compare],
        y=loss_avgs,
        marker_color='#D62828',
        text=[f"{v:.1f}" for v in loss_avgs],
        textposition='inside',
        textfont=dict(color='white', size=11)
    ))
    
    fig.update_layout(
        title=dict(text=f"{player_name} - Performance in Wins vs Losses", x=0.5, xanchor='center'),
        yaxis_title="Average Value",
        yaxis=dict(range=y_range),
        template="plotly_white",
        height=400,
        barmode='group',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=80, b=60, l=60, r=60)
    )
    
    return fig


def render_player_stats_table(player_df: pd.DataFrame):
    """Render the player statistics table with formatting."""
    display_cols = ['GAME_DATE', 'MATCHUP', 'WL'] + NUMERIC_COLS
    available_cols = [col for col in display_cols if col in player_df.columns]
    available_numeric = [col for col in NUMERIC_COLS if col in player_df.columns]
    
    st.dataframe(
        format_dataframe(player_df[available_cols], available_numeric),
        width='stretch'
    )


def render_summary_stats(player_df: pd.DataFrame):
    """Display summary statistics in metric cards."""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_fg3m = player_df['FG3M'].sum()
    total_fg3a = player_df['FG3A'].sum()
    fg3_pct = (total_fg3m / total_fg3a) if total_fg3a > 0 else 0
    
    total_ftm = player_df['FTM'].sum()
    total_fta = player_df['FTA'].sum()
    ft_pct = (total_ftm / total_fta) if total_fta > 0 else 0
    
    total_fgm = player_df['FGM'].sum()
    total_fga = player_df['FGA'].sum()
    fg_pct = (total_fgm / total_fga) if total_fga > 0 else 0
    
    with col1:
        st.metric("Games Played", len(player_df))
        st.metric("Avg Points", f"{player_df['PTS'].mean():.1f}")
    
    with col2:
        st.metric("Avg Rebounds", f"{player_df['REB'].mean():.1f}")
        st.metric("Avg Assists", f"{player_df['AST'].mean():.1f}")
    
    with col3:
        st.metric("FG%", f"{fg_pct:.1%}")
        st.metric("3PT%", f"{fg3_pct:.1%}")
    
    with col4:
        st.metric("FT%", f"{ft_pct:.1%}")
        st.metric("Avg +/-", f"{player_df['PLUS_MINUS'].mean():.1f}")
    
    with col5:
        st.metric("Avg Steals", f"{player_df['STL'].mean():.1f}")
        st.metric("Avg Blocks", f"{player_df['BLK'].mean():.1f}")


def render_career_stats_tab(player_id: int, player_name: str):
    """Render the Career Statistics tab."""
    with st.spinner("Loading career statistics..."):
        per_game_stats, totals_stats = load_career_stats(player_id)
    
    if per_game_stats is None or totals_stats is None:
        st.error("Unable to load career statistics. Please try again later.")
        return
    
    # Get regular season data
    per_game_df = per_game_stats.get("SeasonTotalsRegularSeason")
    totals_df = totals_stats.get("SeasonTotalsRegularSeason")
    
    if per_game_df is None or per_game_df.empty:
        st.info("No career statistics available for this player")
        return
    
    # Ensure numeric columns are properly typed
    numeric_cols = ['GP', 'GS', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 
                    'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 
                    'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']
    
    for col in numeric_cols:
        if col in per_game_df.columns:
            per_game_df[col] = pd.to_numeric(per_game_df[col], errors='coerce')
        if col in totals_df.columns:
            totals_df[col] = pd.to_numeric(totals_df[col], errors='coerce')
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Career Overview",
        "📈 Progression Charts", 
        "🏆 Season Highlights",
        "📋 Full Stats Table"
    ])
    
    # TAB 1: Career Overview
    with tab1:
        st.markdown("#### 🎯 Career at a Glance")
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        total_games = totals_df['GP'].sum() if 'GP' in totals_df.columns else 0
        total_seasons = len(per_game_df)
        career_ppg = per_game_df['PTS'].mean() if 'PTS' in per_game_df.columns else 0
        career_rpg = per_game_df['REB'].mean() if 'REB' in per_game_df.columns else 0
        career_apg = per_game_df['AST'].mean() if 'AST' in per_game_df.columns else 0
        total_points = totals_df['PTS'].sum() if 'PTS' in totals_df.columns else 0
        
        with col1:
            st.metric("🏀 Seasons", total_seasons)
        with col2:
            st.metric("🎮 Games", f"{total_games:.0f}")
        with col3:
            st.metric("⭐ PPG", f"{career_ppg:.1f}")
        with col4:
            st.metric("📦 RPG", f"{career_rpg:.1f}")
        with col5:
            st.metric("🎯 APG", f"{career_apg:.1f}")
        with col6:
            st.metric("🔥 Total Pts", f"{total_points:.0f}")
        
        st.divider()
        
        # Shooting percentages
        st.markdown("#### 🎲 Career Shooting Efficiency")
        col1, col2, col3, col4 = st.columns(4)
        
        career_fg_pct = per_game_df['FG_PCT'].mean() if 'FG_PCT' in per_game_df.columns else 0
        career_3p_pct = per_game_df['FG3_PCT'].mean() if 'FG3_PCT' in per_game_df.columns else 0
        career_ft_pct = per_game_df['FT_PCT'].mean() if 'FT_PCT' in per_game_df.columns else 0
        
        with col1:
            st.metric("FG%", f"{career_fg_pct:.1%}")
        with col2:
            st.metric("3P%", f"{career_3p_pct:.1%}")
        with col3:
            st.metric("FT%", f"{career_ft_pct:.1%}")
        with col4:
            # True shooting percentage
            total_pts = totals_df['PTS'].sum()
            total_fga = totals_df['FGA'].sum()
            total_fta = totals_df['FTA'].sum()
            ts_pct = (total_pts / (2 * (total_fga + 0.44 * total_fta))) if (total_fga + total_fta) > 0 else 0
            st.metric("TS%", f"{ts_pct:.1%}")
        
        st.divider()
        
        # Career milestones
        st.markdown("#### 🌟 Career Milestones")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            best_season = per_game_df.loc[per_game_df['PTS'].idxmax()] if 'PTS' in per_game_df.columns else None
            if best_season is not None:
                st.markdown("**🔥 Best Scoring Season**")
                st.write(f"**{best_season.get('SEASON_ID', 'N/A')}** with {best_season.get('TEAM_ABBREVIATION', 'N/A')}")
                st.write(f"• {best_season['PTS']:.1f} PPG in {best_season.get('GP', 0):.0f} games")
        
        with col2:
            if 'GP' in totals_df.columns:
                most_games = totals_df.loc[totals_df['GP'].idxmax()]
                st.markdown("**💪 Most Durable Season**")
                st.write(f"**{most_games.get('SEASON_ID', 'N/A')}** with {most_games.get('TEAM_ABBREVIATION', 'N/A')}")
                st.write(f"• {most_games['GP']:.0f} games, {most_games.get('MIN', 0):.0f} total minutes")
        
        with col3:
            if 'FG_PCT' in per_game_df.columns:
                # Filter to seasons with significant minutes
                qualified_seasons = per_game_df[per_game_df['MIN'] > 20] if 'MIN' in per_game_df.columns else per_game_df
                if not qualified_seasons.empty:
                    best_efficiency = qualified_seasons.loc[qualified_seasons['FG_PCT'].idxmax()]
                    st.markdown("**🎯 Most Efficient Season**")
                    st.write(f"**{best_efficiency.get('SEASON_ID', 'N/A')}** with {best_efficiency.get('TEAM_ABBREVIATION', 'N/A')}")
                    st.write(f"• {best_efficiency['FG_PCT']:.1%} FG%, {best_efficiency.get('PTS', 0):.1f} PPG")
    
    # TAB 2: Progression Charts
    with tab2:
        col1, col2 = st.columns([1, 3])
        
        with col1:
            selected_career_stat = st.selectbox(
                "Select Statistic:",
                ['PTS', 'REB', 'AST', 'FG_PCT', 'FG3_PCT', 'MIN', 'STL', 'BLK'],
                key="career_stat_select"
            )
        
        with col2:
            st.empty()
        
        # Single stat progression
        if selected_career_stat in per_game_df.columns:
            import numpy as np
            fig_progression = create_career_progression_chart(per_game_df, player_name, selected_career_stat)
            st.plotly_chart(fig_progression, use_container_width=True)
        
        st.divider()
        
        # Multi-stat comparison
        st.markdown("#### 📊 Multi-Season Comparison")
        fig_comparison = create_career_comparison_chart(per_game_df, player_name)
        st.plotly_chart(fig_comparison, use_container_width=True)
    
    # TAB 3: Season Highlights
    with tab3:
                # Statistical breakdown with radar charts
        st.markdown("#### 🌟 Peak vs Current Season Comparison")
        
        if len(per_game_df) > 0:
            col1, col2, col3 = st.columns([1, 1, 1])
            
            # Get peak season and current season
            peak_season = per_game_df.loc[per_game_df['PTS'].idxmax()] if 'PTS' in per_game_df.columns else None
            current_season = per_game_df.iloc[-1]
            career_avg = per_game_df[['PTS', 'REB', 'AST', 'STL', 'BLK']].mean()
            
        # ---------- Peak Season ----------
        with col1:
            st.markdown("**🏆 Peak Season**")
            if peak_season is not None:
                st.write(f"**{peak_season.get('SEASON_ID', 'N/A')}** with {peak_season.get('TEAM_ABBREVIATION', 'N/A')}")
                ppg_col, rpg_col, apg_col, fg_col = st.columns(4)
                ppg_col.metric("PPG", f"{peak_season.get('PTS', 0):.1f}")
                rpg_col.metric("RPG", f"{peak_season.get('REB', 0):.1f}")
                apg_col.metric("APG", f"{peak_season.get('AST', 0):.1f}")
                fg_col.metric("FG%", f"{peak_season.get('FG_PCT', 0):.1%}")

        # ---------- Career Average ----------
        with col2:
            st.markdown("**📊 Career Average**")
            st.write("**All Seasons**")
            ppg_col, rpg_col, apg_col, fg_col = st.columns(4)
            ppg_col.metric("PPG", f"{career_avg.get('PTS', 0):.1f}")
            rpg_col.metric("RPG", f"{career_avg.get('REB', 0):.1f}")
            apg_col.metric("APG", f"{career_avg.get('AST', 0):.1f}")
            avg_fg = per_game_df['FG_PCT'].mean() if 'FG_PCT' in per_game_df.columns else 0
            fg_col.metric("FG%", f"{avg_fg:.1%}")

        # ---------- Current Season ----------
        with col3:
            st.markdown("**🔥 Current Season**")
            st.write(f"**{current_season.get('SEASON_ID', 'N/A')}** with {current_season.get('TEAM_ABBREVIATION', 'N/A')}")
            ppg_col, rpg_col, apg_col, fg_col = st.columns(4)

            current_pts = current_season.get('PTS', 0)
            delta_pts = current_pts - career_avg.get('PTS', 0)
            ppg_col.metric("PPG", f"{current_pts:.1f}", delta=f"{delta_pts:+.1f}")

            current_reb = current_season.get('REB', 0)
            delta_reb = current_reb - career_avg.get('REB', 0)
            rpg_col.metric("RPG", f"{current_reb:.1f}", delta=f"{delta_reb:+.1f}")

            current_ast = current_season.get('AST', 0)
            delta_ast = current_ast - career_avg.get('AST', 0)
            apg_col.metric("APG", f"{current_ast:.1f}", delta=f"{delta_ast:+.1f}")

            current_fg = current_season.get('FG_PCT', 0)
            delta_fg = (current_fg - avg_fg) * 100
            fg_col.metric("FG%", f"{current_fg:.1%}", delta=f"{delta_fg:+.1f}%")
        
        st.divider()
        
        st.markdown("#### 🏆 Best Performances by Category")
        
        # Create visualizations for top seasons
        col1, col2 = st.columns(2)
        
        with col1:
            # Top 5 scoring seasons bar chart
            if 'PTS' in per_game_df.columns:
                top_scoring = per_game_df.nlargest(5, 'PTS')[['SEASON_ID', 'TEAM_ABBREVIATION', 'PTS', 'GP']].copy()
                top_scoring['Label'] = top_scoring['SEASON_ID'].astype(str) + ' (' + top_scoring['TEAM_ABBREVIATION'].astype(str) + ')'
                
                fig_scoring = go.Figure()
                fig_scoring.add_trace(go.Bar(
                    y=top_scoring['Label'][::-1],  # Reverse for top-to-bottom
                    x=top_scoring['PTS'][::-1],
                    orientation='h',
                    marker=dict(
                        color=top_scoring['PTS'][::-1],
                        colorscale='Reds',
                        showscale=False
                    ),
                    text=[f"{pts:.1f} PPG" for pts in top_scoring['PTS'][::-1]],
                    textposition='outside',
                    hovertemplate='<b>%{y}</b><br>%{x:.1f} PPG<extra></extra>'
                ))
                
                fig_scoring.update_layout(
                    title="🔥 Top 5 Scoring Seasons",
                    xaxis_title="Points Per Game",
                    height=300,
                    margin=dict(l=150, r=50, t=60, b=40),
                    template="plotly_white"
                )
                st.plotly_chart(fig_scoring, use_container_width=True)
        
        with col2:
            # Top 5 efficiency seasons (with minimum games filter)
            if 'FG_PCT' in per_game_df.columns:
                qualified = per_game_df[per_game_df['GP'] > 20] if 'GP' in per_game_df.columns else per_game_df
                top_efficiency = qualified.nlargest(5, 'FG_PCT')[['SEASON_ID', 'TEAM_ABBREVIATION', 'FG_PCT', 'PTS']].copy()
                top_efficiency['Label'] = top_efficiency['SEASON_ID'].astype(str) + ' (' + top_efficiency['TEAM_ABBREVIATION'].astype(str) + ')'
                
                fig_efficiency = go.Figure()
                fig_efficiency.add_trace(go.Bar(
                    y=top_efficiency['Label'][::-1],
                    x=(top_efficiency['FG_PCT'][::-1] * 100),
                    orientation='h',
                    marker=dict(
                        color=top_efficiency['FG_PCT'][::-1] * 100,
                        colorscale='Greens',
                        showscale=False
                    ),
                    text=[f"{pct:.1f}%" for pct in top_efficiency['FG_PCT'][::-1] * 100],
                    textposition='outside',
                    hovertemplate='<b>%{y}</b><br>%{x:.1f}% FG<extra></extra>'
                ))
                
                fig_efficiency.update_layout(
                    title="🎯 Top 5 Efficiency Seasons (FG%)",
                    xaxis_title="Field Goal Percentage",
                    height=300,
                    margin=dict(l=150, r=50, t=60, b=40),
                    template="plotly_white"
                )
                st.plotly_chart(fig_efficiency, use_container_width=True)
        
        st.divider()
        
        # Heatmap of key stats across all seasons
        st.markdown("#### 🔥 Performance Heatmap Across Career")
        
        # Prepare data for heatmap
        heatmap_stats = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'FG_PCT', 'FG3_PCT']
        available_stats = [stat for stat in heatmap_stats if stat in per_game_df.columns]
        
        if available_stats:
            heatmap_data = per_game_df[['SEASON_ID'] + available_stats].copy()
            
            # Normalize each stat to 0-100 scale for better visualization
            for stat in available_stats:
                if stat in ['FG_PCT', 'FG3_PCT', 'FT_PCT']:
                    heatmap_data[stat] = heatmap_data[stat] * 100
                else:
                    # Normalize to percentile
                    max_val = heatmap_data[stat].max()
                    if max_val > 0:
                        heatmap_data[stat] = (heatmap_data[stat] / max_val) * 100
            
            # Create heatmap
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=heatmap_data[available_stats].values.T,
                x=heatmap_data['SEASON_ID'],
                y=[stat.replace('_', ' ').replace('PCT', '%') for stat in available_stats],
                colorscale='RdYlGn',
                text=heatmap_data[available_stats].values.T,
                texttemplate='%{text:.1f}',
                textfont={"size": 10},
                colorbar=dict(title="Performance<br>Percentile"),
                hovertemplate='<b>%{y}</b><br>Season: %{x}<br>Value: %{text:.1f}<extra></extra>'
            ))
            
            fig_heatmap.update_layout(
                title="Career Performance Heatmap (Normalized)",
                xaxis_title="Season",
                height=350,
                template="plotly_white",
                margin=dict(l=100, r=50, t=60, b=60)
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        st.divider()
        

        # Career trajectory with annotations
        st.markdown("#### 📈 Career Scoring Trajectory")
        
        if 'PTS' in per_game_df.columns and 'SEASON_ID' in per_game_df.columns:
            fig_trajectory = go.Figure()
            
            # Main scoring line
            fig_trajectory.add_trace(go.Scatter(
                x=per_game_df['SEASON_ID'],
                y=per_game_df['PTS'],
                mode='lines+markers',
                name='PPG',
                line=dict(color='#4ECDC4', width=3),
                marker=dict(size=10, color='#4ECDC4'),
                fill='tozeroy',
                fillcolor='rgba(78, 205, 196, 0.2)',
                hovertemplate='<b>%{x}</b><br>%{y:.1f} PPG<extra></extra>'
            ))
            
            # Add career average line
            fig_trajectory.add_hline(
                y=career_avg.get('PTS', 0),
                line_dash="dash",
                line_color="orange",
                annotation_text=f"Career Avg: {career_avg.get('PTS', 0):.1f}",
                annotation_position="right"
            )
            
            # Highlight peak season
            peak_idx = per_game_df['PTS'].idxmax()
            peak_season_data = per_game_df.loc[peak_idx]
            fig_trajectory.add_trace(go.Scatter(
                x=[peak_season_data['SEASON_ID']],
                y=[peak_season_data['PTS']],
                mode='markers+text',
                marker=dict(size=20, color='red', symbol='star'),
                text=['Peak'],
                textposition='top center',
                name='Peak Season',
                hovertemplate=f"<b>Peak: {peak_season_data['SEASON_ID']}</b><br>{peak_season_data['PTS']:.1f} PPG<extra></extra>"
            ))
            
            fig_trajectory.update_layout(
                title=f"{player_name} - Career Scoring Journey",
                xaxis_title="Season",
                yaxis_title="Points Per Game",
                height=400,
                template="plotly_white",
                hovermode='x unified',
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig_trajectory, use_container_width=True)
        
        st.divider()
        
        # Additional categories in compact format
        st.markdown("#### 📊 Category Leaders")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("**📦 Rebounding**")
            if 'REB' in per_game_df.columns:
                top_reb = per_game_df.nlargest(3, 'REB')[['SEASON_ID', 'REB']]
                for idx, row in top_reb.iterrows():
                    st.write(f"**{row['SEASON_ID']}**")
                    st.write(f"└ {row['REB']:.1f} RPG")
        
        with col2:
            st.markdown("**🎭 Playmaking**")
            if 'AST' in per_game_df.columns:
                top_ast = per_game_df.nlargest(3, 'AST')[['SEASON_ID', 'AST']]
                for idx, row in top_ast.iterrows():
                    st.write(f"**{row['SEASON_ID']}**")
                    st.write(f"└ {row['AST']:.1f} APG")
        
        with col3:
            st.markdown("**🛡️ Steals**")
            if 'STL' in per_game_df.columns:
                top_stl = per_game_df.nlargest(3, 'STL')[['SEASON_ID', 'STL']]
                for idx, row in top_stl.iterrows():
                    st.write(f"**{row['SEASON_ID']}**")
                    st.write(f"└ {row['STL']:.1f} SPG")
        
        with col4:
            st.markdown("**🚫 Blocks**")
            if 'BLK' in per_game_df.columns:
                top_blk = per_game_df.nlargest(3, 'BLK')[['SEASON_ID', 'BLK']]
                for idx, row in top_blk.iterrows():
                    st.write(f"**{row['SEASON_ID']}**")
                    st.write(f"└ {row['BLK']:.1f} BPG")
    
    # TAB 4: Full Stats Table
    with tab4:
        st.markdown("#### 📋 Complete Season-by-Season Statistics")
        
        subtab1, subtab2 = st.tabs(["Per Game", "Season Totals"])
        
        with subtab1:
            st.caption("Average statistics per game for each season")
            # Select key columns for cleaner display
            display_cols = ['SEASON_ID', 'TEAM_ABBREVIATION', 'GP', 'MIN', 'PTS', 'REB', 
                          'AST', 'STL', 'BLK', 'FG_PCT', 'FG3_PCT', 'FT_PCT']
            available_cols = [col for col in display_cols if col in per_game_df.columns]
            st.dataframe(per_game_df[available_cols], width='stretch', height=400)
            
            with st.expander("📊 View All Columns"):
                st.dataframe(per_game_df, width='stretch', height=400)
        
        with subtab2:
            st.caption("Cumulative totals for each season")
            display_cols = ['SEASON_ID', 'TEAM_ABBREVIATION', 'GP', 'MIN', 'PTS', 'REB', 
                          'AST', 'STL', 'BLK', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA']
            available_cols = [col for col in display_cols if col in totals_df.columns]
            st.dataframe(totals_df[available_cols], width='stretch', height=400)
            
            with st.expander("📊 View All Columns"):
                st.dataframe(totals_df, width='stretch', height=400)


def render_performance_trends_tab(player_df: pd.DataFrame, player_name: str):
    """Render the Performance Trends tab."""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_stat = st.selectbox(
            "Select Statistic:",
            STAT_OPTIONS,
            help="Choose a statistic to visualize over time"
        )
    
    with col2:
        rolling_window = st.slider(
            "Rolling Average:",
            min_value=1,
            max_value=10,
            value=3,
            help="Number of games for rolling average calculation"
        )

    fig_trend = create_trend_chart(player_df, player_name, selected_stat, rolling_window)
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # Recent Form Analysis
    st.markdown("#### 📊 Recent Form Analysis")
    col1, col2, col3, col4 = st.columns(4)
    
    avg_pts = player_df['PTS'].mean()
    pts_trend = player_df['PTS'].tail(5).mean() - player_df['PTS'].head(5).mean()
    
    with col1:
        st.markdown("**📈 Recent Form**")
        last_5_avg = player_df['PTS'].tail(5).mean()
        last_10_avg = player_df['PTS'].tail(10).mean()
        consistency_score = 100 - (player_df['PTS'].std() / player_df['PTS'].mean() * 100) if player_df['PTS'].mean() > 0 else 0
        
        st.write(f"• Season Avg: {avg_pts:.1f} PPG")
        st.write(f"• Last 5 Games: {last_5_avg:.1f} PPG")
        st.write(f"• Last 10 Games: {last_10_avg:.1f} PPG")
        st.write(f"• Consistency: {consistency_score:.0f}/100")
        
        if pts_trend > 3:
            st.success(f"📈 Heating up (+{pts_trend:.1f} PPG)")
        elif pts_trend < -3:
            st.error(f"📉 Cooling off ({pts_trend:.1f} PPG)")
        else:
            st.info("➡️ Consistent performance")
    
    with col2:
        st.markdown("**🔥 Hot Streaks**")
        if len(player_df) >= 3:
            best_3game_pts = 0
            for i in range(len(player_df) - 2):
                three_game_pts = player_df.iloc[i:i+3]['PTS'].sum()
                if three_game_pts > best_3game_pts:
                    best_3game_pts = three_game_pts
            
            st.write(f"• Best 3-game: {best_3game_pts:.0f} pts")
            st.write(f"• Avg: {best_3game_pts/3:.1f} PPG")
            
            games_20plus = len(player_df[player_df['PTS'] >= 20])
            games_30plus = len(player_df[player_df['PTS'] >= 30])
            st.write(f"• 20+ pt games: {games_20plus}")
            st.write(f"• 30+ pt games: {games_30plus}")
            
            if games_30plus > 0:
                st.success("💥 Big game player")
    
    with col3:
        st.markdown("**⚖️ Consistency Metrics**")
        pts_std = player_df['PTS'].std()
        pts_cv = (pts_std / player_df['PTS'].mean() * 100) if player_df['PTS'].mean() > 0 else 0
        
        st.write(f"• Std Dev: {pts_std:.1f}")
        st.write(f"• Variability: {pts_cv:.1f}%")
        st.write(f"• High Game: {player_df['PTS'].max():.0f}")
        st.write(f"• Low Game: {player_df['PTS'].min():.0f}")
        
        if pts_cv < 30:
            st.success("✅ Very consistent")
        elif pts_cv < 50:
            st.info("📊 Moderately consistent")
        else:
            st.warning("⚡ High variance")
    
    with col4:
        st.markdown("**🌟 Best Performance**")
        best_pts_game = player_df.loc[player_df['PTS'].idxmax()]
        game_date = pd.to_datetime(best_pts_game['GAME_DATE']).strftime('%b %d')
        
        st.write(f"**{best_pts_game['PTS']:.0f} PTS** vs {best_pts_game['MATCHUP'].split()[-1]}")
        st.write(f"• {best_pts_game['REB']:.0f} REB / {best_pts_game['AST']:.0f} AST")
        st.write(f"• {best_pts_game['FG_PCT']:.1%} FG")
        st.write(f"• Date: {game_date}")
        
        if best_pts_game['WL'] == 'W':
            st.success(f"✅ Won ({best_pts_game['PLUS_MINUS']:+.0f})")
        else:
            st.error(f"❌ Lost ({best_pts_game['PLUS_MINUS']:+.0f})")


def render_shooting_analysis_tab(player_df: pd.DataFrame, player_name: str):
    """Render the Shooting Analysis tab."""
    st.caption("See how the player scores points - breakdown by 2PT, 3PT, and Free Throws")
    fig_breakdown = create_shooting_breakdown_chart(player_df, player_name)
    st.plotly_chart(fig_breakdown, use_container_width=True)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        fig_distribution = create_shot_distribution_chart(player_df, player_name)
        st.plotly_chart(fig_distribution, use_container_width=True)
    
    with col2:
        fig_consistency = create_consistency_chart(player_df, player_name)
        st.plotly_chart(fig_consistency, use_container_width=True)
    
    st.divider()
    
    # Shooting Insights
    st.markdown("#### 🎯 Shooting Insights")
    col1, col2, col3, col4 = st.columns(4)
    
    total_2pt = player_df['FG2M'].sum() * 2
    total_3pt = player_df['FG3M'].sum() * 3
    total_ft = player_df['FTM'].sum()
    total_pts = total_2pt + total_3pt + total_ft
    
    pct_2pt = (total_2pt / total_pts * 100) if total_pts > 0 else 0
    pct_3pt = (total_3pt / total_pts * 100) if total_pts > 0 else 0
    pct_ft = (total_ft / total_pts * 100) if total_pts > 0 else 0
    
    with col1:
        st.markdown("**🎯 Scoring Profile**")
        st.write(f"• 2-Pointers: {pct_2pt:.1f}%")
        st.write(f"• 3-Pointers: {pct_3pt:.1f}%")
        st.write(f"• Free Throws: {pct_ft:.1f}%")
        
        if pct_3pt > 40:
            st.success("🏹 Three-point specialist")
        elif pct_2pt > 60:
            st.info("💪 Paint dominator")
        else:
            st.warning("⚖️ Balanced scorer")
    
    with col2:
        st.markdown("**🎲 Efficiency Metrics**")
        true_shooting = (player_df['PTS'].sum() / (2 * (player_df['FGA'].sum() + 0.44 * player_df['FTA'].sum()))) * 100 if (player_df['FGA'].sum() + player_df['FTA'].sum()) > 0 else 0
        efg = ((player_df['FGM'].sum() + 0.5 * player_df['FG3M'].sum()) / player_df['FGA'].sum() * 100) if player_df['FGA'].sum() > 0 else 0
        
        st.write(f"• True Shooting: {true_shooting:.1f}%")
        st.write(f"• Effective FG%: {efg:.1f}%")
        st.write(f"• AST/TO: {(player_df['AST'].sum() / player_df['TOV'].sum()):.2f}" if player_df['TOV'].sum() > 0 else "• AST/TO: N/A")
        
        if true_shooting > 60:
            st.success("🎯 Elite efficiency")
        elif true_shooting > 55:
            st.info("✅ Above average")
        else:
            st.warning("📊 Room to improve")
    
    with col3:
        st.markdown("**📊 Shot Volume**")
        st.write(f"• Avg FGA: {player_df['FGA'].mean():.1f}")
        st.write(f"• Avg 3PA: {player_df['FG3A'].mean():.1f}")
        st.write(f"• Avg FTA: {player_df['FTA'].mean():.1f}")
        st.write(f"• Total Shots: {(player_df['FGA'].sum() + player_df['FTA'].sum()):.0f}")
    
    with col4:
        st.markdown("**📅 Notable Games**")
        player_df_efficient = player_df[player_df['PTS'] >= player_df['PTS'].quantile(0.6)].copy()
        if len(player_df_efficient) > 0:
            most_efficient = player_df_efficient.loc[player_df_efficient['FG_PCT'].idxmax()]
            st.write(f"**Most Efficient:**")
            st.write(f"• {most_efficient['PTS']:.0f} pts on {most_efficient['FG_PCT']:.1%} FG")
            st.write(f"• vs {most_efficient['MATCHUP'].split()[-1]}")


def render_player_impact_tab(player_df: pd.DataFrame, player_name: str):
    """Render the Player Impact tab."""
    col1, col2 = st.columns([1, 1])
    
    with col1:
        fig_impact = create_impact_metrics_chart(player_df, player_name)
        st.plotly_chart(fig_impact, use_container_width=True)
    
    with col2:
        fig_wl = create_win_loss_comparison(player_df, player_name)
        if fig_wl:
            st.plotly_chart(fig_wl, use_container_width=True)
        else:
            st.info("Not enough win/loss data for comparison")
    
    st.divider()
    
    # Impact Insights
    st.markdown("#### 💡 Impact Insights")
    col1, col2, col3, col4 = st.columns(4)
    
    wins = len(player_df[player_df['WL'] == 'W'])
    losses = len(player_df[player_df['WL'] == 'L'])
    win_pct = (wins / len(player_df) * 100) if len(player_df) > 0 else 0
    
    with col1:
        st.markdown("**🏆 Impact on Winning**")
        st.write(f"• Record: {wins}-{losses} ({win_pct:.1f}%)")
        
        if wins > 0 and losses > 0:
            avg_plus_minus_wins = player_df[player_df['WL'] == 'W']['PLUS_MINUS'].mean()
            avg_plus_minus_losses = player_df[player_df['WL'] == 'L']['PLUS_MINUS'].mean()
            plus_minus_diff = avg_plus_minus_wins - avg_plus_minus_losses
            
            pts_in_wins = player_df[player_df['WL'] == 'W']['PTS'].mean()
            pts_in_losses = player_df[player_df['WL'] == 'L']['PTS'].mean()
            
            st.write(f"• Avg in Wins: {pts_in_wins:.1f} PPG")
            st.write(f"• Avg in Losses: {pts_in_losses:.1f} PPG")
            st.write(f"• +/- Diff: {plus_minus_diff:+.1f}")
            
            if plus_minus_diff > 10:
                st.success("⭐ Elite impact in wins")
            elif pts_in_wins > pts_in_losses + 5:
                st.info("🔥 Scores more in wins")
            else:
                st.warning("⚡ Consistent regardless")
        else:
            st.write("• Not enough data")
    
    with col2:
        st.markdown("**🛡️ Defensive Impact**")
        avg_stl = player_df['STL'].mean()
        avg_blk = player_df['BLK'].mean()
        avg_def_reb = player_df['DREB'].mean()
        defensive_rating = (avg_stl + avg_blk + avg_def_reb) / 3
        
        st.write(f"• Avg Steals: {avg_stl:.1f}")
        st.write(f"• Avg Blocks: {avg_blk:.1f}")
        st.write(f"• Avg Def Reb: {avg_def_reb:.1f}")
        st.write(f"• Combined: {defensive_rating:.1f}")
        
        if defensive_rating > 3:
            st.success("🛡️ Strong defender")
        elif defensive_rating > 2:
            st.info("✅ Solid defense")
        else:
            st.warning("⚡ Offense-focused")
    
    with col3:
        st.markdown("**🎭 Versatility**")
        double_doubles = len(player_df[
            ((player_df['PTS'] >= 10) & (player_df['REB'] >= 10)) |
            ((player_df['PTS'] >= 10) & (player_df['AST'] >= 10)) |
            ((player_df['REB'] >= 10) & (player_df['AST'] >= 10))
        ])
        
        triple_doubles = len(player_df[
            (player_df['PTS'] >= 10) & 
            (player_df['REB'] >= 10) & 
            (player_df['AST'] >= 10)
        ])
        
        st.write(f"• Double-doubles: {double_doubles}")
        st.write(f"• Triple-doubles: {triple_doubles}")
        st.write(f"• DD Rate: {(double_doubles/len(player_df)*100):.1f}%")
        
        if double_doubles / len(player_df) > 0.5:
            st.success("⭐ Consistent all-around")
        elif double_doubles > 0:
            st.info("✅ Versatile player")
    
    with col4:
        st.markdown("**⚡ Overall Rating**")
        # Calculate composite rating
        pts_rating = min(player_df['PTS'].mean() / 30 * 100, 100)
        eff_rating = player_df['FG_PCT'].mean() * 100
        impact_rating = ((player_df['PLUS_MINUS'].mean() + 15) / 30 * 100)
        overall_rating = (pts_rating + eff_rating + impact_rating) / 3
        
        st.write(f"• Scoring: {pts_rating:.0f}/100")
        st.write(f"• Efficiency: {eff_rating:.0f}/100")
        st.write(f"• Impact: {max(0, min(100, impact_rating)):.0f}/100")
        st.write(f"• Overall: {overall_rating:.0f}/100")
        
        if overall_rating > 75:
            st.success("🌟 Elite player")
        elif overall_rating > 60:
            st.info("⭐ Strong contributor")
        else:
            st.warning("📊 Role player")


def render_single_player_tab():
    """Main function to render the single player analytics tab."""
    df, last_updated = load_data(DATA_FILE)

    if df.empty:
        st.info("📊 No player data available. Please fetch the latest data first.")
        return

    # Header
    st.markdown("### 🏀 Single Player Analytics")

    # Player selection
    player_name = st.selectbox(
        "Select Player:",
        options=sorted(df['PLAYER_NAME'].unique()),
        help="Choose a player to view their detailed statistics"
    )
    
    player_df = (
        df[df['PLAYER_NAME'] == player_name]
        .sort_values('GAME_DATE')
        .reset_index(drop=True)
    )

    if player_df.empty:
        st.warning("⚠️ No game data available for this player.")
        return
    
    # Get player ID for career stats
    player_id = player_df['PLAYER_ID'].iloc[0] if 'PLAYER_ID' in player_df.columns else None

    # Summary statistics
    st.markdown("#### 📈 Season Summary")
    render_summary_stats(player_df)
    
    st.divider()

    # Game log table
    with st.expander("📋 View Full Game Log", expanded=False):
        render_player_stats_table(player_df)
    
    st.divider()
    st.markdown("### 📈 Advanced Analytics")


    # Create tabs for different analysis sections
    if player_id:
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Performance Trends",
            "🎯 Shooting Analysis", 
            "🌟 Player Impact",
            "🏆 Career Stats"
        ])
    else:
        tab1, tab2, tab3 = st.tabs([
            "📊 Performance Trends",
            "🎯 Shooting Analysis", 
            "🌟 Player Impact"
        ])
    
    with tab1:
        render_performance_trends_tab(player_df, player_name)
    
    with tab2:
        render_shooting_analysis_tab(player_df, player_name)
    
    with tab3:
        render_player_impact_tab(player_df, player_name)
    
    if player_id:
        with tab4:
            render_career_stats_tab(player_id, player_name)