"""NBA Game Simulator - Monte Carlo simulation for team matchups."""

import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Optional

# Constants
DATA_FILE = Path("latest_data/games/team_stats.csv")
POSSESSIONS_PER_QUARTER = 25
AND_ONE_PROBABILITY = 0.15
DEFAULT_SIMULATIONS = 1000


def load_data() -> pd.DataFrame:
    """Load team stats data from CSV file."""
    if DATA_FILE.exists():
        return pd.read_csv(DATA_FILE)
    return pd.DataFrame()


def calculate_team_stats(df: pd.DataFrame, team_name: str) -> Optional[Dict[str, float]]:
    """
    Calculate average statistical profile for a team.
    
    Args:
        df: DataFrame containing team game stats
        team_name: Name of the team to analyze
        
    Returns:
        Dictionary of averaged team statistics or None if team not found
    """
    team_data = df[df['TEAM_NAME'] == team_name]
    
    if team_data.empty:
        return None
    
    # Calculate shooting percentages from totals for accuracy
    total_fgm = team_data['FGM'].sum()
    total_fga = team_data['FGA'].sum()
    total_fg3m = team_data['FG3M'].sum()
    total_fg3a = team_data['FG3A'].sum()
    total_ftm = team_data['FTM'].sum()
    total_fta = team_data['FTA'].sum()
    
    return {
        'PTS': team_data['PTS'].mean(),
        'FG_PCT': (total_fgm / total_fga * 100) if total_fga > 0 else 0,
        'FG3_PCT': (total_fg3m / total_fg3a * 100) if total_fg3a > 0 else 0,
        'FT_PCT': (total_ftm / total_fta * 100) if total_fta > 0 else 0,
        'REB': team_data['REB'].mean(),
        'AST': team_data['AST'].mean(),
        'STL': team_data['STL'].mean(),
        'BLK': team_data['BLK'].mean(),
        'TOV': team_data['TOV'].mean(),
        'FGA': team_data['FGA'].mean(),
        'FG3A': team_data['FG3A'].mean(),
        'FTA': team_data['FTA'].mean(),
    }


def simulate_possession(team_stats: Dict[str, float]) -> int:
    """
    Simulate a single offensive possession.
    
    Args:
        team_stats: Dictionary containing team statistics
        
    Returns:
        Points scored on the possession (0, 1, 2, or 3)
    """
    # Check for turnover
    turnover_rate = team_stats['TOV'] / POSSESSIONS_PER_QUARTER / 4
    if np.random.random() < turnover_rate:
        return 0
    
    # Determine shot type based on attempt distribution
    three_point_rate = team_stats['FG3A'] / team_stats['FGA']
    is_three_pointer = np.random.random() < three_point_rate
    
    if is_three_pointer:
        # Three-point attempt
        if np.random.random() < (team_stats['FG3_PCT'] / 100):
            return 3
    else:
        # Two-point attempt - calculate 2P% from overall FG% and 3P%
        fg2_pct = (
            (team_stats['FG_PCT'] - team_stats['FG3_PCT'] * three_point_rate) /
            (1 - three_point_rate)
        )
        
        if np.random.random() < (fg2_pct / 100):
            points = 2
            # Check for and-one opportunity
            if np.random.random() < AND_ONE_PROBABILITY:
                if np.random.random() < (team_stats['FT_PCT'] / 100):
                    points += 1
            return points
    
    return 0


def simulate_quarter(team_a_stats: Dict[str, float], 
                     team_b_stats: Dict[str, float]) -> tuple[int, int]:
    """
    Simulate a single quarter of play.
    
    Args:
        team_a_stats: Statistics for team A
        team_b_stats: Statistics for team B
        
    Returns:
        Tuple of (team_a_points, team_b_points) for the quarter
    """
    team_a_points = sum(
        simulate_possession(team_a_stats) 
        for _ in range(POSSESSIONS_PER_QUARTER)
    )
    team_b_points = sum(
        simulate_possession(team_b_stats) 
        for _ in range(POSSESSIONS_PER_QUARTER)
    )
    
    return team_a_points, team_b_points


def simulate_game(team_a_stats: Dict[str, float], 
                  team_b_stats: Dict[str, float], 
                  quarters: int = 4) -> Dict:
    """
    Simulate a complete basketball game.
    
    Args:
        team_a_stats: Statistics for team A
        team_b_stats: Statistics for team B
        quarters: Number of quarters to simulate (default: 4)
        
    Returns:
        Dictionary containing game results and quarter-by-quarter breakdown
    """
    np.random.seed(None)  # Ensure true randomness
    
    quarter_scores_a = []
    quarter_scores_b = []
    
    for _ in range(quarters):
        q_a, q_b = simulate_quarter(team_a_stats, team_b_stats)
        quarter_scores_a.append(q_a)
        quarter_scores_b.append(q_b)
    
    team_a_total = sum(quarter_scores_a)
    team_b_total = sum(quarter_scores_b)
    
    return {
        'team_a_total': team_a_total,
        'team_b_total': team_b_total,
        'team_a_quarters': quarter_scores_a,
        'team_b_quarters': quarter_scores_b,
        'winner': 'team_a' if team_a_total > team_b_total else 'team_b'
    }


def run_monte_carlo(team_a_stats: Dict[str, float], 
                   team_b_stats: Dict[str, float], 
                   simulations: int = DEFAULT_SIMULATIONS) -> Dict:
    """
    Run Monte Carlo simulation to predict matchup outcomes.
    
    Args:
        team_a_stats: Statistics for team A
        team_b_stats: Statistics for team B
        simulations: Number of games to simulate
        
    Returns:
        Dictionary containing aggregated simulation results
    """
    team_a_wins = 0
    score_differentials = []
    team_a_scores = []
    team_b_scores = []
    
    for _ in range(simulations):
        result = simulate_game(team_a_stats, team_b_stats)
        
        if result['winner'] == 'team_a':
            team_a_wins += 1
        
        differential = result['team_a_total'] - result['team_b_total']
        score_differentials.append(differential)
        team_a_scores.append(result['team_a_total'])
        team_b_scores.append(result['team_b_total'])
    
    team_b_wins = simulations - team_a_wins
    
    return {
        'team_a_win_pct': (team_a_wins / simulations) * 100,
        'team_b_win_pct': (team_b_wins / simulations) * 100,
        'avg_differential': np.mean(score_differentials),
        'score_differentials': score_differentials,
        'avg_team_a_score': np.mean(team_a_scores),
        'avg_team_b_score': np.mean(team_b_scores),
        'team_a_scores': team_a_scores,
        'team_b_scores': team_b_scores
    }


def render_single_game_result(sim_data: Dict) -> None:
    """Render results for a single game simulation."""
    result = sim_data['result']
    team_a = sim_data['team_a']
    team_b = sim_data['team_b']
    
    st.markdown("### 🏆 Game Result")
    
    # Display final scores
    winner_a = result['winner'] == 'team_a'
    winner_b = result['winner'] == 'team_b'
    
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col1:
        st.markdown(f"#### {'🏆 ' if winner_a else ''}{team_a}")
        score_color = '#4CAF50' if winner_a else '#888'
        st.markdown(
            f"<h1 style='text-align: center; color: {score_color};'>"
            f"{result['team_a_total']}</h1>",
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            "<div style='text-align: center; padding-top: 40px; font-size: 20px;'>"
            "FINAL</div>",
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(f"#### {'🏆 ' if winner_b else ''}{team_b}")
        score_color = '#4CAF50' if winner_b else '#888'
        st.markdown(
            f"<h1 style='text-align: center; color: {score_color};'>"
            f"{result['team_b_total']}</h1>",
            unsafe_allow_html=True
        )
    
    st.markdown("---")
    
    # Quarter-by-quarter breakdown
    st.markdown("#### 📊 Quarter-by-Quarter Breakdown")
    
    quarters_df = pd.DataFrame({
        'Quarter': [f'Q{i+1}' for i in range(len(result['team_a_quarters']))] + ['TOTAL'],
        team_a: result['team_a_quarters'] + [result['team_a_total']],
        team_b: result['team_b_quarters'] + [result['team_b_total']]
    })
    
    st.dataframe(quarters_df, use_container_width=True, hide_index=True)
    
    # Scoring chart
    fig = go.Figure([
        go.Bar(
            name=team_a,
            x=[f'Q{i+1}' for i in range(len(result['team_a_quarters']))],
            y=result['team_a_quarters'],
            marker_color='#FF6B6B'
        ),
        go.Bar(
            name=team_b,
            x=[f'Q{i+1}' for i in range(len(result['team_b_quarters']))],
            y=result['team_b_quarters'],
            marker_color='#4ECDC4'
        )
    ])
    
    fig.update_layout(
        title='Scoring by Quarter',
        barmode='group',
        xaxis_title='Quarter',
        yaxis_title='Points',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_monte_carlo_result(sim_data: Dict) -> None:
    """Render results for Monte Carlo simulation."""
    result = sim_data['result']
    team_a = sim_data['team_a']
    team_b = sim_data['team_b']
    
    st.markdown("### 📈 Monte Carlo Simulation Results (1000 Games)")

    blowout_threshold = 15
    close_threshold = 5
    total_games = len(result['score_differentials'])

    # Team A metrics
    team_a_blowouts = sum(1 for diff in result['score_differentials'] if diff >= blowout_threshold)
    team_a_blowout_pct = (team_a_blowouts / total_games) * 100

    team_a_close_wins = sum(1 for diff in result['score_differentials'] if diff > 0 and diff <= close_threshold)
    team_a_close_pct = (team_a_close_wins / total_games) * 100

    team_a_margin = np.mean([diff for diff in result['score_differentials'] if diff > 0])

    # Team B metrics
    team_b_blowouts = sum(1 for diff in result['score_differentials'] if diff <= -blowout_threshold)
    team_b_blowout_pct = (team_b_blowouts / total_games) * 100

    team_b_close_wins = sum(1 for diff in result['score_differentials'] if diff < 0 and abs(diff) <= close_threshold)
    team_b_close_pct = (team_b_close_wins / total_games) * 100

    team_b_margin = np.mean([abs(diff) for diff in result['score_differentials'] if diff < 0])

    # Win probabilities
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"#### {team_a}")
        st.markdown(
            f"<h1 style='text-align: center; color: #FF6B6B;'>{result['team_a_win_pct']:.1f}%</h1>",
            unsafe_allow_html=True
        )
        st.markdown("<p style='text-align: center;'>Win Probability</p>", unsafe_allow_html=True)
        
        score_col, blowout_col, close_col, margin_col = st.columns(4)
        with score_col:
            st.metric("Avg Score", f"{result['avg_team_a_score']:.1f}")
        with blowout_col:
            st.metric("% Blowout Wins (>15)", f"{team_a_blowout_pct:.1f}%")
        with close_col:
            st.metric("% Close Wins (<5)", f"{team_a_close_pct:.1f}%")
        with margin_col:
            st.metric("Avg Margin of Victory", f"{team_a_margin:.1f} pts")

    with col2:
        st.markdown(f"#### {team_b}")
        st.markdown(
            f"<h1 style='text-align: center; color: #4ECDC4;'>{result['team_b_win_pct']:.1f}%</h1>",
            unsafe_allow_html=True
        )
        st.markdown("<p style='text-align: center;'>Win Probability</p>", unsafe_allow_html=True)
        
        score_col, blowout_col, close_col, margin_col = st.columns(4)
        with score_col:
            st.metric("Avg Score", f"{result['avg_team_b_score']:.1f}")
        with blowout_col:
            st.metric("% Blowout Wins (>15)", f"{team_b_blowout_pct:.1f}%")
        with close_col:
            st.metric("% Close Wins (<5)", f"{team_b_close_pct:.1f}%")
        with margin_col:
            st.metric("Avg Margin of Victory", f"{team_b_margin:.1f} pts")
    
    team_a_pct = result['team_a_win_pct']
    team_b_pct = 100 - team_a_pct  # ensure total = 100%

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=[team_a_pct],
        y=["Win Probability"],
        orientation='h',
        marker_color="#FF6B6B",
        text=f"{team_a_pct:.1f}%",
        textposition='inside'
    ))

    fig.add_trace(go.Bar(
        x=[team_b_pct],
        y=["Win Probability"],
        orientation='h',
        marker_color="#4ECDC4",
        text=f"{team_b_pct:.1f}%",
        textposition='inside'
    ))

    fig.update_layout(
        barmode='stack',
        xaxis=dict(range=[0, 100], title="Win Probability (%)"),
        yaxis=dict(showticklabels=False),
        height=80,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")
    
    # Distribution charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Score Distribution")
        
        fig = go.Figure([
            go.Histogram(
                x=result['team_a_scores'],
                name=team_a,
                opacity=0.7,
                marker_color='#FF6B6B',
                nbinsx=30
            ),
            go.Histogram(
                x=result['team_b_scores'],
                name=team_b,
                opacity=0.7,
                marker_color='#4ECDC4',
                nbinsx=30
            )
        ])
        
        fig.update_layout(
            barmode='overlay',
            xaxis_title='Points',
            yaxis_title='Frequency',
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Point Differential Distribution")
        
        fig = go.Figure(go.Histogram(
            x=result['score_differentials'],
            marker_color='#8B5CF6',
            nbinsx=40
        ))
        
        fig.add_vline(
            x=0,
            line_dash="dash",
            line_color="red",
            annotation_text="Even Game"
        )
        
        fig.update_layout(
            xaxis_title=f'Point Differential ({team_a} perspective)',
            yaxis_title='Frequency',
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Generate insights
    render_simulation_insights(result, team_a, team_b)
    
    # Summary statistics
    render_summary_stats(result)


def render_simulation_insights(result: Dict, team_a: str, team_b: str) -> None:
    """Generate and display simulation insights."""
    st.markdown("---")
    st.markdown("#### 🎯 Simulation Insights")
    
    insights = []
    
    # Win probability insight
    if result['team_a_win_pct'] > 65:
        insights.append(
            f"🔥 **{team_a}** is heavily favored with "
            f"{result['team_a_win_pct']:.1f}% win probability"
        )
    elif result['team_b_win_pct'] > 65:
        insights.append(
            f"🔥 **{team_b}** is heavily favored with "
            f"{result['team_b_win_pct']:.1f}% win probability"
        )
    else:
        insights.append("⚖️ This is projected to be a **close matchup** - either team can win")
    
    # Score margin insight
    avg_diff = abs(result['avg_differential'])
    if avg_diff < 3:
        insights.append(f"🎲 Expected to be a **nail-biter** (avg margin: {avg_diff:.1f} pts)")
    elif avg_diff > 10:
        insights.append(
            f"💪 One team has a significant statistical edge (avg margin: {avg_diff:.1f} pts)"
        )
    
    # Variance insight
    spread = np.std(result['score_differentials'])
    if spread > 12:
        insights.append(
            f"📊 High variance game - outcomes vary widely (std dev: {spread:.1f})"
        )
    else:
        insights.append(f"📊 Consistent outcomes across simulations (std dev: {spread:.1f})")
    
    for insight in insights:
        st.info(insight)


def render_summary_stats(result: Dict) -> None:
    """Display summary statistics from simulation."""
    st.markdown("---")
    st.markdown("#### 📋 Summary Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        median_diff = np.median(result['score_differentials'])
        st.metric("Median Differential", f"{median_diff:.1f} pts")
    
    with col2:
        score_range_a = (
            f"{int(result['avg_team_a_score'] - 5)}-"
            f"{int(result['avg_team_a_score'] + 5)}"
        )
        score_range_b = (
            f"{int(result['avg_team_b_score'] - 5)}-"
            f"{int(result['avg_team_b_score'] + 5)}"
        )
        st.metric("Most Likely Score Range", f"{score_range_a} vs {score_range_b}")
    
    with col3:
        close_games = sum(1 for diff in result['score_differentials'] if abs(diff) <= 5)
        close_pct = (close_games / len(result['score_differentials'])) * 100
        st.metric("Close Games (<5 pts)", f"{close_pct:.1f}%")


def render_simulator_tab() -> None:
    """Main function to render the simulator tab."""
    st.markdown("### 🎮 Game Simulator")
    st.markdown("Simulate matchups based on team averages and statistical modeling")
    
    # Load data
    df = load_data()
    
    if df.empty:
        st.warning("⚠️ No game data available. Please fetch data from the Games tab first.")
        return
    
    if 'TEAM_NAME' not in df.columns:
        st.error("Team data not found in the dataset.")
        return
    
    teams = sorted(df['TEAM_NAME'].dropna().unique())
    
    st.markdown("---")
    
    # Team selection
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col1:
        team_a = st.selectbox("🏀 Home Team", teams, key="sim_team_a")
    
    with col2:
        st.markdown(
            "<div style='text-align: center; padding-top: 28px; font-size: 24px;'>VS</div>",
            unsafe_allow_html=True
        )
    
    with col3:
        team_b = st.selectbox(
            "🏀 Away Team",
            teams,
            key="sim_team_b",
            index=min(1, len(teams) - 1)
        )
    
    if team_a == team_b:
        st.info("Please select two different teams to simulate.")
        return
    
    # Calculate team statistics
    team_a_stats = calculate_team_stats(df, team_a)
    team_b_stats = calculate_team_stats(df, team_b)
    
    if not team_a_stats or not team_b_stats:
        st.error("Unable to calculate stats for selected teams.")
        return
    
    st.markdown("---")
    
    # Simulation controls
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        sim_type = st.radio(
            "Simulation Type",
            ["Single Game", "Monte Carlo (1000 sims)"],
            key="sim_type"
        )
    
    with col2:
        st.write("")
        st.write("")
        simulate_button = st.button(
            "▶️ Run Simulation",
            type="primary",
            use_container_width=True
        )
    
    # Run simulation and display results
    if simulate_button:
        st.markdown("---")
        
        if sim_type == "Single Game":
            with st.spinner("Simulating game..."):
                result = simulate_game(team_a_stats, team_b_stats)
                st.session_state.last_simulation = {
                    'type': 'single',
                    'result': result,
                    'team_a': team_a,
                    'team_b': team_b
                }
        else:
            with st.spinner("Running 1000 simulations..."):
                result = run_monte_carlo(team_a_stats, team_b_stats)
                st.session_state.last_simulation = {
                    'type': 'monte_carlo',
                    'result': result,
                    'team_a': team_a,
                    'team_b': team_b
                }
    
    # Display cached results if available
    if 'last_simulation' in st.session_state:
        sim_data = st.session_state.last_simulation
        
        if sim_data['type'] == 'single':
            render_single_game_result(sim_data)
        else:
            render_monte_carlo_result(sim_data)