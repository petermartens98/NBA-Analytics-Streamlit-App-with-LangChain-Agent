import streamlit as st
import pandas as pd
from utils.data_helpers.load_data import load_data
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

DATA_FILE = Path("latest_data/players.csv")
COACH_FILE = Path("latest_data/team_rosters_coaches.csv")
PLAYER_FILE = Path("latest_data/team_rosters_players.csv")

def load_roster_data():
    coaches = pd.read_csv(COACH_FILE) if COACH_FILE.exists() else pd.DataFrame()
    players = pd.read_csv(PLAYER_FILE) if PLAYER_FILE.exists() else pd.DataFrame()
    return coaches, players

def prepare_team_data(df, team_name):
    """Filter and prepare team data with calculated statistics."""
    team_df = df[df['TEAM_NAME'] == team_name].copy()
    
    if not pd.api.types.is_datetime64_any_dtype(team_df['GAME_DATE']):
        team_df['GAME_DATE'] = pd.to_datetime(team_df['GAME_DATE'])
    
    team_df = team_df.sort_values('GAME_DATE').reset_index(drop=True)
    
    # Aggregate team stats per game
    team_game_stats = team_df.groupby(['GAME_ID', 'GAME_DATE', 'MATCHUP', 'WL']).agg({
        'PTS': 'sum', 'REB': 'sum', 'AST': 'sum', 'STL': 'sum', 'BLK': 'sum', 'TOV': 'sum',
        'FGA': 'sum', 'FGM': 'sum', 'FG3M': 'sum', 'FG3A': 'sum', 'FTM': 'sum', 'FTA': 'sum'
    }).reset_index()

    # Create DATE_MATCHUP column
    team_game_stats['DATE_MATCHUP'] = team_game_stats['GAME_DATE'].dt.strftime('%Y-%m-%d') + ' - ' + team_game_stats['MATCHUP']

    # Calculate shooting percentages
    team_game_stats['FG_PCT'] = (team_game_stats['FGM'] / team_game_stats['FGA'] * 100).fillna(0)
    team_game_stats['FG3_PCT'] = (team_game_stats['FG3M'] / team_game_stats['FG3A'] * 100).fillna(0)
    team_game_stats['FT_PCT'] = (team_game_stats['FTM'] / team_game_stats['FTA'] * 100).fillna(0)
    
    return team_df, team_game_stats

def render_season_overview(team_game_stats):
    """Display season overview metrics."""
    st.markdown("---")
    st.markdown("#### Season Overview")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    wins = len(team_game_stats[team_game_stats['WL'] == 'W'])
    losses = len(team_game_stats[team_game_stats['WL'] == 'L'])
    win_pct = wins/(wins+losses)*100 if (wins+losses) > 0 else 0
    
    with col1:
        st.metric("Record", f"{wins}-{losses}")
    with col2:
        st.metric("Win Rate", f"{win_pct:.1f}%")
    with col3:
        st.metric("PPG", f"{team_game_stats['PTS'].mean():.1f}")
    with col4:
        st.metric("APG", f"{team_game_stats['AST'].mean():.1f}")
    with col5:
        st.metric("RPG", f"{team_game_stats['REB'].mean():.1f}")
    with col6:
        st.metric("FG%", f"{team_game_stats['FG_PCT'].mean():.1f}%")

    # Recent form
    recent_games = team_game_stats.tail(5)
    recent_form = ''.join(['🟢' if w == 'W' else '🔴' for w in recent_games['WL']])
    recent_wins = recent_games['WL'].value_counts().get('W', 0)
    recent_losses = recent_games['WL'].value_counts().get('L', 0)
    st.markdown(f"**Last 5 Games:** {recent_form} ({recent_wins}-{recent_losses})")

def apply_table_filters(team_game_stats):
    """Apply filters and sorting to the games table."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        result_filter = st.selectbox(
            "Filter by Result:",
            options=['All', 'Wins', 'Losses'],
            key='result_filter'
        )
    with col2:
        sort_by = st.selectbox(
            "Sort by:",
            options=['Date (Newest)', 'Date (Oldest)', 'Points (High)', 'Points (Low)', 'FG% (High)', 'FG% (Low)'],
            key='sort_by'
        )
    with col3:
        min_rows = min(5, len(team_game_stats))
        default_rows = min(10, len(team_game_stats))
        show_rows = st.number_input(
            "Show rows:",
            min_value=min_rows,
            max_value=len(team_game_stats),
            value=default_rows,
            step=min_rows,
            key='show_rows'
        )
    
    # Apply filters
    filtered_stats = team_game_stats.copy()
    if result_filter == 'Wins':
        filtered_stats = filtered_stats[filtered_stats['WL'] == 'W']
    elif result_filter == 'Losses':
        filtered_stats = filtered_stats[filtered_stats['WL'] == 'L']
    
    # Apply sorting
    sort_mapping = {
        'Date (Newest)': ('GAME_DATE', False),
        'Date (Oldest)': ('GAME_DATE', True),
        'Points (High)': ('PTS', False),
        'Points (Low)': ('PTS', True),
        'FG% (High)': ('FG_PCT', False),
        'FG% (Low)': ('FG_PCT', True)
    }
    sort_col, ascending = sort_mapping[sort_by]
    filtered_stats = filtered_stats.sort_values(sort_col, ascending=ascending)
    
    return filtered_stats.head(int(show_rows))

def render_games_table(filtered_stats, team_name):
    """Display the games table with styling."""
    display_cols = ['DATE_MATCHUP', 'WL', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 
                    'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT']
    
    numeric_cols = [col for col in display_cols if col not in ['DATE_MATCHUP', 'WL']]
    pct_cols = [col for col in numeric_cols if 'PCT' in col]
    regular_cols = [col for col in numeric_cols if 'PCT' not in col]
    
    format_dict = {col: '{:.0f}' for col in regular_cols}
    format_dict.update({col: '{:.1f}%' for col in pct_cols})
    
    def highlight_result(val):
        if val == 'W':
            return 'background-color: #90EE90; color: #006400; font-weight: bold'
        elif val == 'L':
            return 'background-color: #FFB6C6; color: #8B0000; font-weight: bold'
        return ''
    
    st.dataframe(
        filtered_stats[display_cols].style
        .format(format_dict)
        .applymap(highlight_result, subset=['WL']),
        use_container_width=True,
        height=min(400, (len(filtered_stats) + 1) * 35 + 3)
    )

def create_points_trend_chart(team_game_stats, team_name):
    """Create points per game trend chart."""
    # Sort by date first
    sorted_stats = team_game_stats.sort_values('GAME_DATE').reset_index(drop=True)
    
    fig_points = go.Figure()
    
    wins_df = sorted_stats[sorted_stats['WL'] == 'W']
    losses_df = sorted_stats[sorted_stats['WL'] == 'L']
    avg_pts = sorted_stats['PTS'].mean()
    
    fig_points.add_trace(go.Scatter(
        x=wins_df['GAME_DATE'], y=wins_df['PTS'],
        mode='markers+lines', name='Wins',
        marker=dict(color='green', size=12, symbol='circle'),
        line=dict(color='lightgreen', width=2)
    ))
    
    fig_points.add_trace(go.Scatter(
        x=losses_df['GAME_DATE'], y=losses_df['PTS'],
        mode='markers+lines', name='Losses',
        marker=dict(color='red', size=12, symbol='x'),
        line=dict(color='lightcoral', width=2)
    ))
    
    fig_points.add_hline(
        y=avg_pts, line_dash="dash", line_color="gray",
        annotation_text=f"Season Avg: {avg_pts:.1f}",
        annotation_position="right"
    )
    
    fig_points.update_layout(
        title=f"{team_name} Points per Game",
        xaxis_title="Game Date", yaxis_title="Points",
        hovermode='x unified', height=500
    )
    return fig_points

def create_win_loss_comparison(team_game_stats, wins, losses):
    """Create win/loss comparison charts."""
    win_loss_stats = pd.DataFrame({
        'Result': ['Wins', 'Losses'],
        'Avg Points': [
            team_game_stats[team_game_stats['WL'] == 'W']['PTS'].mean() if wins > 0 else 0,
            team_game_stats[team_game_stats['WL'] == 'L']['PTS'].mean() if losses > 0 else 0
        ],
        'Avg Assists': [
            team_game_stats[team_game_stats['WL'] == 'W']['AST'].mean() if wins > 0 else 0,
            team_game_stats[team_game_stats['WL'] == 'L']['AST'].mean() if losses > 0 else 0
        ],
        'Avg Rebounds': [
            team_game_stats[team_game_stats['WL'] == 'W']['REB'].mean() if wins > 0 else 0,
            team_game_stats[team_game_stats['WL'] == 'L']['REB'].mean() if losses > 0 else 0
        ]
    })
    
    # Points comparison
    fig_pts = px.bar(
        win_loss_stats, x='Result', y='Avg Points',
        title="Average Points: Wins vs Losses",
        color='Result', color_discrete_map={'Wins': 'green', 'Losses': 'red'},
        text='Avg Points'
    )
    fig_pts.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    fig_pts.update_layout(
        showlegend=False,
        yaxis=dict(range=[0, win_loss_stats['Avg Points'].max() * 1.15])
    )
    
    # Multi-stat comparison
    fig_stats = go.Figure()
    fig_stats.add_trace(go.Bar(
        name='Wins', x=['Points', 'Assists', 'Rebounds'],
        y=[win_loss_stats.loc[0, 'Avg Points'], 
           win_loss_stats.loc[0, 'Avg Assists'], 
           win_loss_stats.loc[0, 'Avg Rebounds']],
        marker_color='green',
        text=[f"{win_loss_stats.loc[0, 'Avg Points']:.1f}",
              f"{win_loss_stats.loc[0, 'Avg Assists']:.1f}",
              f"{win_loss_stats.loc[0, 'Avg Rebounds']:.1f}"],
        textposition='outside'
    ))
    fig_stats.add_trace(go.Bar(
        name='Losses', x=['Points', 'Assists', 'Rebounds'],
        y=[win_loss_stats.loc[1, 'Avg Points'], 
           win_loss_stats.loc[1, 'Avg Assists'], 
           win_loss_stats.loc[1, 'Avg Rebounds']],
        marker_color='red',
        text=[f"{win_loss_stats.loc[1, 'Avg Points']:.1f}",
              f"{win_loss_stats.loc[1, 'Avg Assists']:.1f}",
              f"{win_loss_stats.loc[1, 'Avg Rebounds']:.1f}"],
        textposition='outside'
    ))
    max_val = max(win_loss_stats.loc[0, 'Avg Points'], win_loss_stats.loc[1, 'Avg Points'])
    fig_stats.update_layout(
        title='Key Stats Comparison', barmode='group',
        yaxis_title='Average', yaxis=dict(range=[0, max_val * 1.15])
    )
    
    return fig_pts, fig_stats

def create_shooting_charts(team_game_stats):
    """Create shooting efficiency charts."""
    # Sort by date first
    sorted_stats = team_game_stats.sort_values('GAME_DATE').reset_index(drop=True)
    
    # FG% trend
    fig_fg = go.Figure()
    fig_fg.add_trace(go.Scatter(
        x=sorted_stats['GAME_DATE'], y=sorted_stats['FG_PCT'],
        mode='lines+markers', name='FG%',
        line=dict(color='royalblue', width=3), marker=dict(size=8)
    ))
    avg_fg = sorted_stats['FG_PCT'].mean()
    fig_fg.add_hline(y=avg_fg, line_dash="dash", line_color="gray",
                     annotation_text=f"Avg: {avg_fg:.1f}%")
    fig_fg.update_layout(
        title="Field Goal Percentage Trend",
        xaxis_title="Game Date", yaxis_title="FG%", height=400
    )
    
    # 3P% trend
    fig_3pt = go.Figure()
    fig_3pt.add_trace(go.Scatter(
        x=sorted_stats['GAME_DATE'], y=sorted_stats['FG3_PCT'],
        mode='lines+markers', name='3P%',
        line=dict(color='orange', width=3), marker=dict(size=8)
    ))
    avg_3pt = sorted_stats['FG3_PCT'].mean()
    fig_3pt.add_hline(y=avg_3pt, line_dash="dash", line_color="gray",
                      annotation_text=f"Avg: {avg_3pt:.1f}%")
    fig_3pt.update_layout(
        title="Three-Point Percentage Trend",
        xaxis_title="Game Date", yaxis_title="3P%", height=400
    )
    
    return fig_fg, fig_3pt

def create_shooting_comparison(team_game_stats, wins, losses):
    """Create win/loss shooting comparison chart."""
    # Sort by date first
    sorted_stats = team_game_stats.sort_values('GAME_DATE').reset_index(drop=True)
    
    shooting_comparison = pd.DataFrame({
        'Result': ['Wins', 'Losses'],
        'FG%': [
            sorted_stats[sorted_stats['WL'] == 'W']['FG_PCT'].mean() if wins > 0 else 0,
            sorted_stats[sorted_stats['WL'] == 'L']['FG_PCT'].mean() if losses > 0 else 0
        ],
        '3P%': [
            sorted_stats[sorted_stats['WL'] == 'W']['FG3_PCT'].mean() if wins > 0 else 0,
            sorted_stats[sorted_stats['WL'] == 'L']['FG3_PCT'].mean() if losses > 0 else 0
        ],
        'FT%': [
            sorted_stats[sorted_stats['WL'] == 'W']['FT_PCT'].mean() if wins > 0 else 0,
            sorted_stats[sorted_stats['WL'] == 'L']['FT_PCT'].mean() if losses > 0 else 0
        ]
    })
    
    fig_shooting = go.Figure()
    fig_shooting.add_trace(go.Bar(
        name='Wins', x=['FG%', '3P%', 'FT%'],
        y=[shooting_comparison.loc[0, 'FG%'], 
           shooting_comparison.loc[0, '3P%'], 
           shooting_comparison.loc[0, 'FT%']],
        marker_color='green',
        text=[f"{shooting_comparison.loc[0, 'FG%']:.1f}%",
              f"{shooting_comparison.loc[0, '3P%']:.1f}%",
              f"{shooting_comparison.loc[0, 'FT%']:.1f}%"],
        textposition='outside'
    ))
    fig_shooting.add_trace(go.Bar(
        name='Losses', x=['FG%', '3P%', 'FT%'],
        y=[shooting_comparison.loc[1, 'FG%'], 
           shooting_comparison.loc[1, '3P%'], 
           shooting_comparison.loc[1, 'FT%']],
        marker_color='red',
        text=[f"{shooting_comparison.loc[1, 'FG%']:.1f}%",
              f"{shooting_comparison.loc[1, '3P%']:.1f}%",
              f"{shooting_comparison.loc[1, 'FT%']:.1f}%"],
        textposition='outside'
    ))
    max_pct = max(shooting_comparison[['FG%', '3P%', 'FT%']].max())
    fig_shooting.update_layout(
        title='Shooting Percentages: Wins vs Losses',
        barmode='group', yaxis_title='Percentage',
        height=400, yaxis=dict(range=[0, max_pct * 1.15])
    )
    return fig_shooting

def create_team_stats_chart(team_game_stats):
    """Create team statistics trends chart."""
    sorted_stats = team_game_stats.sort_values('GAME_DATE').reset_index(drop=True)
    
    fig_stats = go.Figure()
    stats_to_plot = [
        ('REB', 'Rebounds', dict(width=2)),
        ('AST', 'Assists', dict(width=2)),
        ('STL', 'Steals', dict(width=2)),
        ('BLK', 'Blocks', dict(width=2)),
        ('TOV', 'Turnovers', dict(width=2, dash='dash'))
    ]
    
    for stat, name, line_style in stats_to_plot:
        fig_stats.add_trace(go.Scatter(
            x=sorted_stats['GAME_DATE'], y=sorted_stats[stat],
            mode='lines+markers', name=name, line=line_style
        ))
    
    fig_stats.update_layout(
        title="Team Statistics Trends",
        xaxis_title="Game Date", yaxis_title="Count",
        hovermode='x unified', height=500
    )
    return fig_stats

def render_top_performers(team_df):
    """Display top performers by game."""
    st.markdown("##### Top Performers by Game")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🏆 Leading Scorers**")
        top_scorers = team_df.loc[team_df.groupby('GAME_ID')['PTS'].idxmax()]
        top_scorers['DATE_MATCHUP'] = top_scorers['GAME_DATE'].dt.strftime('%Y-%m-%d') + ' - ' + top_scorers['MATCHUP']
        display = top_scorers[['DATE_MATCHUP', 'PLAYER_NAME', 'PTS']].sort_values('DATE_MATCHUP', ascending=False).head(10)
        st.dataframe(display, use_container_width=True, height=350)
    
    with col2:
        st.markdown("**🎯 Best Playmakers**")
        top_assisters = team_df.loc[team_df.groupby('GAME_ID')['AST'].idxmax()]
        top_assisters['DATE_MATCHUP'] = top_assisters['GAME_DATE'].dt.strftime('%Y-%m-%d') + ' - ' + top_assisters['MATCHUP']
        display = top_assisters[['DATE_MATCHUP', 'PLAYER_NAME', 'AST']].sort_values('DATE_MATCHUP', ascending=False).head(10)
        st.dataframe(display, use_container_width=True, height=350)
    
    with col3:
        st.markdown("**🔥 Top Rebounders**")
        top_rebounders = team_df.loc[team_df.groupby('GAME_ID')['REB'].idxmax()]
        top_rebounders['DATE_MATCHUP'] = top_rebounders['GAME_DATE'].dt.strftime('%Y-%m-%d') + ' - ' + top_rebounders['MATCHUP']
        display = top_rebounders[['DATE_MATCHUP', 'PLAYER_NAME', 'REB']].sort_values('DATE_MATCHUP', ascending=False).head(10)
        st.dataframe(display, use_container_width=True, height=350)

def render_individual_game_analysis(team_df, team_game_stats):
    """Display individual game analysis."""
    st.markdown("##### Individual Game Analysis")
    
    # Create game selector
    game_options = team_game_stats['DATE_MATCHUP'] + ' (' + team_game_stats['WL'] + ')'
    game_id_map = dict(zip(game_options, team_game_stats['GAME_ID']))
    
    selected_game_label = st.selectbox(
        "Select Game:", 
        options=game_options[::-1],
        key='game_selector'
    )
    selected_game_id = game_id_map[selected_game_label]
    
    game_players = team_df[team_df['GAME_ID'] == selected_game_id].sort_values('PTS', ascending=False)
    
    if not game_players.empty:
        game_info = team_game_stats[team_game_stats['GAME_ID'] == selected_game_id].iloc[0]
        
        st.markdown(f"##### {game_info['MATCHUP']}")
        
        # Game summary metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            result_color = "green" if game_info['WL'] == 'W' else "red"
            st.markdown(f"**Result:** :{result_color}[{game_info['WL']}]")
        with col2:
            st.metric("Team Points", f"{int(game_info['PTS'])}")
        with col3:
            st.metric("Assists", f"{int(game_info['AST'])}")
        with col4:
            st.metric("Rebounds", f"{int(game_info['REB'])}")
        with col5:
            st.metric("FG%", f"{game_info['FG_PCT']:.1f}%")
        
        st.markdown("---")
        
        # Player stats table
        player_cols = ['PLAYER_NAME', 'MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 
                       'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT']
        
        if 'FANTASY_PTS' in game_players.columns:
            player_cols.append('FANTASY_PTS')
        
        available_cols = [col for col in player_cols if col in game_players.columns]
        pct_cols = [col for col in available_cols if 'PCT' in col]
        other_numeric = [col for col in available_cols if col not in ['PLAYER_NAME'] and col not in pct_cols]
        
        format_dict = {col: "{:.1f}" for col in other_numeric}
        format_dict.update({col: "{:.1%}" for col in pct_cols})
        
        st.dataframe(
            game_players[available_cols].style.format(format_dict),
            use_container_width=True, height=400
        )
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            fig_pie = px.pie(
                game_players.head(8), values='PTS', names='PLAYER_NAME',
                title='Points Distribution (Top 8 Players)', hole=0.3
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            top_players = game_players.head(5)
            fig_bar = px.bar(
                top_players, x='PLAYER_NAME', y=['PTS', 'REB', 'AST'],
                title='Top 5 Players - Key Stats', barmode='group',
                labels={'value': 'Count', 'variable': 'Stat'}
            )
            fig_bar.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_bar, use_container_width=True)

def render_roster(team_name):
    coaches, players = load_roster_data()
    if coaches.empty and players.empty:
        st.warning("No roster data found. Please update your CSV files in `latest_data/`.")
        return

    # ---- Toggle between Coaches and Players ----
    view_option = st.radio("View:", options=["Coaches", "Players"], horizontal=True)

    if view_option == "Coaches":
        team_coaches = coaches[coaches['TEAM_NAME'] == team_name]
        if not team_coaches.empty:
            st.subheader("Coaching Staff")
            st.dataframe(team_coaches, use_container_width=True)
        else:
            st.info("No coach data available for this team.")
    else:
        team_players = players[players['TEAM_NAME'] == team_name]
        if not team_players.empty:
            st.subheader("Player Roster")
            st.dataframe(team_players, use_container_width=True)
        else:
            st.info("No player data available for this team.")


def render_single_team_tab():
    """Main function to render the team analytics tab with a toggle for Coaches/Players."""
    df, last_updated = load_data(DATA_FILE)

    if df.empty:
        st.info("No player data available. Fetch latest data first.")
        return

    st.markdown("### 🏀 Single Team Analytics")

    # Team selection
    team_name = st.selectbox(
        "Select Team:",
        options=sorted(df['TEAM_NAME'].unique()),
        key='team_selector'
    )

    # ---- Prepare team game data ----
    team_df, team_game_stats = prepare_team_data(df, team_name)
    if team_df.empty:
        st.warning("No games found for this team.")
        return

    # ---- Season Overview ----
    render_season_overview(team_game_stats)



    # ---- Detailed Analysis Tabs ----
    st.markdown("---")
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "👥 Roster",
        "📈 Scoring Trends",
        "🎯 Shooting Efficiency",
        "📊 Team Statistics",
        "📅 All Games",
        "🏀 Individual Game"
    ])


    wins = len(team_game_stats[team_game_stats['WL'] == 'W'])
    losses = len(team_game_stats[team_game_stats['WL'] == 'L'])

    with tab1:
        render_roster(team_name=team_name)

    with tab2:
        st.markdown("##### Points per Game Trends")
        st.plotly_chart(create_points_trend_chart(team_game_stats, team_name), use_container_width=True)
        col1, col2 = st.columns(2)
        fig_pts, fig_stats = create_win_loss_comparison(team_game_stats, wins, losses)
        with col1:
            st.plotly_chart(fig_pts, use_container_width=True)
        with col2:
            st.plotly_chart(fig_stats, use_container_width=True)

    with tab3:
        st.markdown("##### Shooting Efficiency Analysis")
        col1, col2 = st.columns(2)
        fig_fg, fig_3pt = create_shooting_charts(team_game_stats)
        with col1:
            st.plotly_chart(fig_fg, use_container_width=True)
        with col2:
            st.plotly_chart(fig_3pt, use_container_width=True)
        st.markdown("##### Win/Loss Shooting Comparison")
        st.plotly_chart(create_shooting_comparison(team_game_stats, wins, losses), use_container_width=True)

    with tab4:
        st.markdown("##### Team Statistics Overview")
        st.plotly_chart(create_team_stats_chart(team_game_stats), use_container_width=True)
        render_top_performers(team_df)

    with tab5:
    # ---- All Games Table ----
        st.markdown("#### All Games")
        filtered_stats = apply_table_filters(team_game_stats)
        render_games_table(filtered_stats, team_name)

    with tab6:
        render_individual_game_analysis(team_df, team_game_stats)



