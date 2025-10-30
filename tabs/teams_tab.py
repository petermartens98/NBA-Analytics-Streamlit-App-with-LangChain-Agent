# teams_tab.py
import streamlit as st
from pathlib import Path
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.scrape_games import scrape_all_games_team_stats
from utils.calculate_team_averages import calculate_team_averages
from utils.calculate_defense_allowed_team_avg import calculate_defense_allowed_team_avg

data_file = Path("latest_data/games/team_stats.csv")

TEAM_ABBR_NAME_DICT = {
    "HOU": "Houston Rockets","CLE": "Cleveland Cavaliers","ORL": "Orlando Magic",
    "ATL": "Atlanta Hawks","SAC": "Sacramento Kings","IND": "Indiana Pacers",
    "NOP": "New Orleans Pelicans","POR": "Portland Trail Blazers","LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies","TOR": "Toronto Raptors","BOS": "Boston Celtics",
    "DET": "Detroit Pistons","BKN": "Brooklyn Nets","CHI": "Chicago Bulls",
    "DAL": "Dallas Mavericks","DEN": "Denver Nuggets","UTA": "Utah Jazz",
    "MIN": "Minnesota Timberwolves","MIA": "Miami Heat","CHA": "Charlotte Hornets",
    "WAS": "Washington Wizards","LAC": "Los Angeles Clippers","SAS": "San Antonio Spurs",
    "OKC": "Oklahoma City Thunder","GSW": "Golden State Warriors",
    "MIL": "Milwaukee Bucks","NYK": "New York Knicks","PHI": "Philadelphia 76ers"
}

def load_data():
    if data_file.exists():
        df = pd.read_csv(data_file)
        last_updated = datetime.fromtimestamp(data_file.stat().st_mtime)
        return df, last_updated
    return pd.DataFrame(), None

def save_data(df):
    data_file.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(data_file, index=False)

def create_team_comparison_chart(df, metric, title):
    df_sorted = df.sort_values(metric, ascending=True)
    fig = go.Figure(go.Bar(
        x=df_sorted[metric],
        y=df_sorted['TEAM_NAME'],
        orientation='h',
        marker=dict(color=df_sorted[metric], colorscale='Viridis', showscale=True),
        text=df_sorted[metric].round(1),
        textposition='outside'
    ))
    fig.update_layout(title=title, xaxis_title=metric, yaxis_title="", height=800,
                      showlegend=False, margin=dict(l=150))
    return fig

def create_shooting_efficiency_chart(df):
    fig = px.scatter(df, x='FG_PCT', y='PTS', size='FGA', color='FG3_PCT', hover_name='TEAM_NAME',
                     hover_data={'FG_PCT': ':.2%', 'FG3_PCT': ':.2%', 'PTS': ':.1f', 'FGA': ':.1f'},
                     labels={'FG_PCT': 'Field Goal %','PTS': 'Points Per Game','FG3_PCT': '3-Point %'},
                     title='Shooting Efficiency: FG% vs Points (bubble size = FGA)',
                     color_continuous_scale='RdYlGn')
    fig.update_layout(height=500)
    return fig

def create_shot_distribution_chart(df):
    """Visualize 2PT vs 3PT shot distribution"""
    fig = px.scatter(df, x='FRAC_ATT_2PT', y='FRAC_ATT_3PT', 
                     size='PTS', color='FG_PCT', hover_name='TEAM_NAME',
                     hover_data={'FRAC_ATT_2PT': ':.2%', 'FRAC_ATT_3PT': ':.2%', 
                                'PTS': ':.1f', 'FG_PCT': ':.2%'},
                     labels={'FRAC_ATT_2PT': '% of Attempts from 2PT','FRAC_ATT_3PT': '% of Attempts from 3PT',
                            'PTS': 'Points Per Game'},
                     title='Shot Selection: 2PT vs 3PT Attempt Distribution',
                     color_continuous_scale='RdYlGn')
    fig.update_layout(height=500)
    return fig

def create_scoring_breakdown_chart(df):
    """Show where teams get their points from"""
    df_sorted = df.sort_values('PTS', ascending=True)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(y=df_sorted['TEAM_NAME'], x=df_sorted['FG2_PTS'], 
                        name='2-Point FG', orientation='h', marker=dict(color='#FF6B6B')))
    fig.add_trace(go.Bar(y=df_sorted['TEAM_NAME'], x=df_sorted['FG3_PTS'], 
                        name='3-Point FG', orientation='h', marker=dict(color='#4ECDC4')))
    fig.add_trace(go.Bar(y=df_sorted['TEAM_NAME'], x=df_sorted['FTM'], 
                        name='Free Throws', orientation='h', marker=dict(color='#95E1D3')))
    
    fig.update_layout(title='Points Distribution by Source', xaxis_title='Points Per Game',
                      yaxis_title='', barmode='stack', height=800, margin=dict(l=150))
    return fig

def create_offense_defense_chart(df):
    fig = px.scatter(df, x='AST', y='TOV', size='PTS', color='PLUS_MINUS', hover_name='TEAM_NAME',
                     hover_data={'AST': ':.1f','TOV': ':.1f','PTS': ':.1f','PLUS_MINUS': ':.1f'},
                     labels={'AST': 'Assists Per Game','TOV': 'Turnovers Per Game','PLUS_MINUS': 'Plus/Minus'},
                     title='Offense Efficiency: Assists vs Turnovers (bubble size = PTS)',
                     color_continuous_scale='RdYlGn')
    fig.update_layout(height=500)
    return fig

def create_rebounding_chart(df):
    df_sorted = df.sort_values('REB', ascending=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(y=df_sorted['TEAM_NAME'], x=df_sorted['OREB'], name='Offensive Rebounds', 
                        orientation='h', marker=dict(color='#FF6B6B')))
    fig.add_trace(go.Bar(y=df_sorted['TEAM_NAME'], x=df_sorted['DREB'], name='Defensive Rebounds', 
                        orientation='h', marker=dict(color='#4ECDC4')))
    fig.update_layout(title='Rebounding Breakdown by Team', xaxis_title='Rebounds Per Game',
                      yaxis_title='', barmode='stack', height=800, margin=dict(l=150))
    return fig

def create_win_loss_chart(df):
    win_loss = df.groupby('TEAM_NAME')['WL'].value_counts().unstack(fill_value=0)
    win_loss['WIN_PCT'] = win_loss['W'] / (win_loss['W'] + win_loss['L'])
    win_loss = win_loss.sort_values('WIN_PCT', ascending=False)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(y=win_loss.index, x=win_loss['W'] if 'W' in win_loss.columns else 0, 
                        name='Wins', orientation='h', marker=dict(color='#2ECC71'),
                        text=win_loss['W'], textposition='inside'))
    fig.add_trace(go.Bar(y=win_loss.index, x=win_loss['L'] if 'L' in win_loss.columns else 0, 
                        name='Losses', orientation='h', marker=dict(color='#E74C3C'),
                        text=win_loss['L'], textposition='inside'))
    fig.update_layout(title='Win/Loss Records (Sorted by Win %)', xaxis_title='Games', yaxis_title='', 
                      barmode='stack', height=800, margin=dict(l=150))
    return fig

def create_home_away_performance(df):
    """Compare home vs away performance"""
    home_away = df.groupby(['TEAM_NAME', 'HOME_AWAY']).agg({
        'WL': lambda x: (x == 'W').sum(),
        'PTS': 'mean',
        'PLUS_MINUS': 'mean'
    }).reset_index()
    
    home_away_pivot = home_away.pivot(index='TEAM_NAME', columns='HOME_AWAY', values='PTS')
    home_away_pivot = home_away_pivot.sort_values('HOME', ascending=False)
    
    fig = go.Figure()
    if 'HOME' in home_away_pivot.columns:
        fig.add_trace(go.Bar(y=home_away_pivot.index, x=home_away_pivot['HOME'], 
                            name='Home', orientation='h', marker=dict(color='#3498DB')))
    if 'AWAY' in home_away_pivot.columns:
        fig.add_trace(go.Bar(y=home_away_pivot.index, x=home_away_pivot['AWAY'], 
                            name='Away', orientation='h', marker=dict(color='#E67E22')))
    
    fig.update_layout(title='Home vs Away Scoring Performance', xaxis_title='Points Per Game',
                      yaxis_title='', barmode='group', height=800, margin=dict(l=150))
    return fig

def create_conference_performance(df):
    """Analyze conference-based performance"""
    if 'CONFERENCE' not in df.columns or 'OPP_CONFERENCE' not in df.columns:
        return None
    
    # Same conference vs different conference
    df['MATCHUP_TYPE'] = df.apply(lambda x: 'Intra-Conference' if x['CONFERENCE'] == x['OPP_CONFERENCE'] 
                                  else 'Inter-Conference', axis=1)
    
    conf_perf = df.groupby(['TEAM_NAME', 'MATCHUP_TYPE']).agg({
        'WL': lambda x: (x == 'W').sum() / len(x) * 100,
        'PTS': 'mean'
    }).reset_index()
    
    conf_pivot = conf_perf.pivot(index='TEAM_NAME', columns='MATCHUP_TYPE', values='WL')
    conf_pivot = conf_pivot.sort_values('Intra-Conference' if 'Intra-Conference' in conf_pivot.columns else conf_pivot.columns[0], 
                                        ascending=False)
    
    fig = go.Figure()
    for col in conf_pivot.columns:
        fig.add_trace(go.Bar(y=conf_pivot.index, x=conf_pivot[col], 
                            name=col, orientation='h'))
    
    fig.update_layout(title='Win % by Matchup Type', xaxis_title='Win Percentage',
                      yaxis_title='', barmode='group', height=800, margin=dict(l=150))
    return fig

def create_monthly_trend(df):
    """Show team performance trends by month"""
    if 'MONTH' not in df.columns:
        return None
    
    month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                   7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    
    monthly = df.groupby(['TEAM_NAME', 'MONTH']).agg({
        'PTS': 'mean',
        'WL': lambda x: (x == 'W').sum() / len(x) * 100 if len(x) > 0 else 0
    }).reset_index()
    
    monthly['MONTH_NAME'] = monthly['MONTH'].map(month_names)
    
    fig = px.line(monthly, x='MONTH', y='PTS', color='TEAM_NAME',
                 labels={'PTS': 'Points Per Game', 'MONTH': 'Month'},
                 title='Team Scoring Trends by Month')
    fig.update_layout(height=600)
    return fig

def create_radar_chart(df, teams, categories=['PTS', 'REB', 'AST', 'STL', 'BLK', 'FG_PCT']):
    fig = go.Figure()
    for team in teams:
        team_data = df[df['TEAM_NAME'] == team]
        if not team_data.empty:
            values = []
            for cat in categories:
                max_val, min_val = df[cat].max(), df[cat].min()
                values.append(((team_data[cat].values[0]-min_val)/(max_val-min_val))*100)
            values.append(values[0])
            fig.add_trace(go.Scatterpolar(r=values, theta=categories+[categories[0]], fill='toself', name=team))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True,
                      title='Team Comparison (Normalized Stats)', height=500)
    return fig

def create_advanced_metrics_table(df_team_avg):
    """Calculate and display advanced metrics"""
    df_adv = df_team_avg.copy()
    
    # Assist-to-Turnover Ratio
    df_adv['AST_TO_RATIO'] = df_adv['AST'] / df_adv['TOV']
    
    # True Shooting % approximation: PTS / (2 * (FGA + 0.44 * FTA))
    df_adv['TS_PCT'] = df_adv['PTS'] / (2 * (df_adv['FGA'] + 0.44 * df_adv['FTA']))
    
    # Effective FG%: (FGM + 0.5 * FG3M) / FGA
    df_adv['EFG_PCT'] = (df_adv['FGM'] + 0.5 * df_adv['FG3M']) / df_adv['FGA']
    
    # Offensive Rebound %
    df_adv['OREB_PCT'] = df_adv['OREB'] / df_adv['REB']
    
    # Points per shot attempt
    df_adv['PTS_PER_ATT'] = df_adv['PTS'] / df_adv['FGA']
    
    return df_adv

def render_teams_tab():
    df, last_updated = load_data()

    header_col, fetch_col, download_col, last_updated_col = st.columns([4, 2, 2, 2.5])
    with header_col:
        st.markdown("### 🏀 Team Analytics Dashboard")
    
    if df.empty:
        st.info("No team data available. Click 'Fetch Latest' to load.")
        return

    # Calculate team averages
    df_team_avg = calculate_team_averages(df)
    
    # Calculate advanced metrics
    df_advanced = create_advanced_metrics_table(df_team_avg)

    # Defensive allowed averages
    df_defense_avg = calculate_defense_allowed_team_avg(df, TEAM_ABBR_NAME_DICT)

    # Enhanced tabs with more analytics
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Overview","🎯 Shooting Analysis","⚖️ Advanced Stats","🏠 Home/Away & Trends",
        "🔄 Ball Movement","📈 Records","🛡️ Defense","🔍 Team Compare"
    ])

    # ---------- Overview Tab ----------
    with tab1:
        st.markdown("### Team Performance Overview")
        
        # Key metrics in columns
        col1, col2, col3, col4, col5 = st.columns(5)
        top_scorer = df_team_avg.loc[df_team_avg['PTS'].idxmax()]
        best_fg = df_team_avg.loc[df_team_avg['FG_PCT'].idxmax()]
        best_3pt = df_team_avg.loc[df_team_avg['FG3_PCT'].idxmax()]
        top_reb = df_team_avg.loc[df_team_avg['REB'].idxmax()]
        best_pm = df_team_avg.loc[df_team_avg['PLUS_MINUS'].idxmax()]
        
        col1.metric("Top Scoring", top_scorer['TEAM_NAME'], f"{top_scorer['PTS']:.1f} PPG")
        col2.metric("Best FG%", best_fg['TEAM_NAME'], f"{best_fg['FG_PCT']:.1%}")
        col3.metric("Best 3PT%", best_3pt['TEAM_NAME'], f"{best_3pt['FG3_PCT']:.1%}")
        col4.metric("Top Rebounds", top_reb['TEAM_NAME'], f"{top_reb['REB']:.1f} RPG")
        col5.metric("Best +/-", best_pm['TEAM_NAME'], f"{best_pm['PLUS_MINUS']:+.1f}")

        st.markdown("---")
        
        # Quick stats selector
        stat_view = st.selectbox(
            "Select stat category to view:",
            ["Core Stats", "Shooting Stats", "Advanced Metrics", "All Stats"]
        )
        
        if stat_view == "Core Stats":
            display_cols = ['TEAM_NAME','PTS','REB','AST','STL','BLK','TOV','PF','PLUS_MINUS']
        elif stat_view == "Shooting Stats":
            display_cols = ['TEAM_NAME','FGM','FGA','FG_PCT','FG3M','FG3A','FG3_PCT','FTM','FTA','FT_PCT',
                          'FRAC_ATT_2PT','FRAC_ATT_3PT']
        elif stat_view == "Advanced Metrics":
            display_cols = ['TEAM_NAME','AST_TO_RATIO','TS_PCT','EFG_PCT','PTS_PER_ATT','OREB_PCT']
            df_team_avg = df_advanced
        else:
            display_cols = df_team_avg.columns.tolist()
        
        numeric_cols = df_team_avg.select_dtypes(include='number').columns.tolist()
        
        st.dataframe(
            df_team_avg[display_cols].sort_values('PTS' if 'PTS' in display_cols else display_cols[1], 
                                                   ascending=False).style.format(
                {col: "{:.1f}" if col not in ['FG_PCT','FG3_PCT','FT_PCT','TS_PCT','EFG_PCT','OREB_PCT',
                                               'FRAC_ATT_2PT','FRAC_ATT_3PT','FRAC_MK_2PT','FRAC_MK_3PT',
                                               'FRAC_PTS_2PT','FRAC_PTS_3PT','FRAC_PTS_FT'] 
                 else "{:.1%}" for col in numeric_cols}
            ), 
            use_container_width=True, 
            height=500
        )

    # ---------- Shooting Analysis Tab ----------
    with tab2:
        st.markdown("### Comprehensive Shooting Analytics")
        
        st.plotly_chart(create_shooting_efficiency_chart(df_team_avg), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            # Create custom chart with percentage values
            df_sorted_3pt = df_team_avg.sort_values('FG3_PCT', ascending=True)
            fig_3pt = go.Figure(go.Bar(
                x=df_sorted_3pt['FG3_PCT'] * 100,  # Convert to percentage
                y=df_sorted_3pt['TEAM_NAME'],
                orientation='h',
                marker=dict(color=df_sorted_3pt['FG3_PCT'] * 100, colorscale='Viridis', showscale=True),
                text=[f"{val*100:.1f}%" for val in df_sorted_3pt['FG3_PCT']],
                textposition='outside'
            ))
            fig_3pt.update_layout(
                title='3-Point % by Team',
                xaxis_title='FG3_PCT',
                yaxis_title="",
                height=800,
                showlegend=False,
                margin=dict(l=150)
            )
            fig_3pt.update_xaxes(dtick=10, range=[0, max(df_sorted_3pt['FG3_PCT']*100) + 5], ticksuffix='%')
            st.plotly_chart(fig_3pt, use_container_width=True)
        
        with col2:
            # Create custom chart with percentage values
            df_sorted_fg = df_team_avg.sort_values('FG_PCT', ascending=True)
            fig_fg = go.Figure(go.Bar(
                x=df_sorted_fg['FG_PCT'] * 100,  # Convert to percentage
                y=df_sorted_fg['TEAM_NAME'],
                orientation='h',
                marker=dict(color=df_sorted_fg['FG_PCT'] * 100, colorscale='Viridis', showscale=True),
                text=[f"{val*100:.1f}%" for val in df_sorted_fg['FG_PCT']],
                textposition='outside'
            ))
            fig_fg.update_layout(
                title='Overall FG% by Team',
                xaxis_title='FG_PCT',
                yaxis_title="",
                height=800,
                showlegend=False,
                margin=dict(l=150)
            )
            fig_fg.update_xaxes(dtick=10, range=[0, max(df_sorted_fg['FG_PCT']*100) + 5], ticksuffix='%')
            st.plotly_chart(fig_fg, use_container_width=True)
        
        st.plotly_chart(create_shot_distribution_chart(df_team_avg), use_container_width=True)
        st.plotly_chart(create_scoring_breakdown_chart(df_team_avg), use_container_width=True)

    # ---------- Advanced Stats Tab ----------
    with tab3:
        st.markdown("### Advanced Metrics & Efficiency")
        
        col1, col2, col3, col4 = st.columns(4)
        best_ts = df_advanced.loc[df_advanced['TS_PCT'].idxmax()]
        best_efg = df_advanced.loc[df_advanced['EFG_PCT'].idxmax()]
        best_ast_to = df_advanced.loc[df_advanced['AST_TO_RATIO'].idxmax()]
        best_pts_att = df_advanced.loc[df_advanced['PTS_PER_ATT'].idxmax()]
        
        col1.metric("Best True Shooting%", best_ts['TEAM_NAME'], f"{best_ts['TS_PCT']:.1%}")
        col2.metric("Best Effective FG%", best_efg['TEAM_NAME'], f"{best_efg['EFG_PCT']:.1%}")
        col3.metric("Best AST/TO Ratio", best_ast_to['TEAM_NAME'], f"{best_ast_to['AST_TO_RATIO']:.2f}")
        col4.metric("Best Pts/Attempt", best_pts_att['TEAM_NAME'], f"{best_pts_att['PTS_PER_ATT']:.2f}")
        
        st.markdown("---")
        
        # Advanced metrics charts
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_team_comparison_chart(df_advanced,'TS_PCT','True Shooting % by Team'), 
                          use_container_width=True)
        with col2:
            st.plotly_chart(create_team_comparison_chart(df_advanced,'AST_TO_RATIO','Assist/Turnover Ratio'), 
                          use_container_width=True)
        
        # Display advanced metrics table
        adv_cols = ['TEAM_NAME','TS_PCT','EFG_PCT','AST_TO_RATIO','PTS_PER_ATT','OREB_PCT']
        st.dataframe(
            df_advanced[adv_cols].sort_values('TS_PCT', ascending=False).style.format(
                {col: "{:.1%}" if 'PCT' in col else "{:.2f}" for col in adv_cols[1:]}
            ),
            use_container_width=True
        )

    # ---------- Home/Away & Trends Tab ----------
    with tab4:
        
        # Home vs Away
        st.plotly_chart(create_home_away_performance(df), use_container_width=True)
        

    # ---------- Ball Movement Tab ----------
    with tab5:
        st.markdown("### Ball Movement & Efficiency")
        st.plotly_chart(create_offense_defense_chart(df_team_avg), use_container_width=True)
        st.plotly_chart(create_rebounding_chart(df_team_avg), use_container_width=True)
        
        # AST/TO ratio chart
        st.plotly_chart(create_team_comparison_chart(df_advanced,'AST_TO_RATIO',
                                                     'Assist to Turnover Ratio'), 
                       use_container_width=True)

    # ---------- Records Tab ----------
    with tab6:
        st.markdown("### Team Records & Win Analysis")
        st.plotly_chart(create_win_loss_chart(df), use_container_width=True)
        
        # Win % analysis
        win_stats = df.groupby('TEAM_NAME').agg({
            'WL': [lambda x: (x == 'W').sum(), lambda x: (x == 'L').sum(), 
                   lambda x: (x == 'W').sum() / len(x) * 100],
            'PLUS_MINUS': 'mean',
            'PTS': 'mean'
        }).reset_index()
        win_stats.columns = ['TEAM_NAME', 'WINS', 'LOSSES', 'WIN_PCT', 'AVG_PLUS_MINUS', 'AVG_PTS']
        win_stats = win_stats.sort_values('WIN_PCT', ascending=False)
        
        st.dataframe(
            win_stats.style.format({
                'WIN_PCT': '{:.1f}%',
                'AVG_PLUS_MINUS': '{:+.1f}',
                'AVG_PTS': '{:.1f}'
            }).background_gradient(subset=['WIN_PCT'], cmap='RdYlGn'),
            use_container_width=True
        )

    # ---------- Defense Tab ----------
    with tab7:
        st.markdown("### Defensive Analytics")

        defense_metric_options = {
            'OPP_PTS': 'Points Allowed Per Game',
            'OPP_FGM': 'FG Made Allowed',
            'OPP_FGA': 'FG Attempted Allowed',
            'OPP_FG_PCT': 'FG% Allowed',
            'OPP_REB': 'Rebounds Allowed',
            'OPP_AST': 'Assists Allowed',
            'OPP_TOV': 'Turnovers Forced',
            'OPP_STL': 'Steals Allowed',
            'OPP_BLK': 'Blocks Allowed',
            'OPP_PLUS_MINUS': 'Defensive +/-'
        }

        selected_def_metric = st.selectbox(
            "Select defensive stat to visualize:",
            options=list(defense_metric_options.keys()),
            format_func=lambda x: defense_metric_options[x],
            index=0
        )

        # Sort dataframe (ascending for OPP_PTS = better defense)
        ascending = selected_def_metric not in ['OPP_TOV']  # More turnovers forced = better
        df_sorted = df_defense_avg[['TEAM_NAME'] + list(defense_metric_options.keys())].sort_values(
            selected_def_metric, ascending=not ascending
        )

        fig = px.bar(
            df_sorted,
            x=selected_def_metric,
            y='TEAM_NAME',
            orientation='h',
            text=df_sorted[selected_def_metric].round(1),
            color=selected_def_metric,
            color_continuous_scale='RdYlGn_r' if ascending else 'RdYlGn'
        )
        fig.update_layout(
            title=f"{defense_metric_options[selected_def_metric]} by Team",
            xaxis_title=defense_metric_options[selected_def_metric],
            yaxis_title="",
            height=800,
            showlegend=False,
            margin=dict(l=150)
        )

        st.plotly_chart(fig, use_container_width=True)

        numeric_cols = df_sorted.select_dtypes(include='number').columns.tolist()
        st.dataframe(
            df_sorted.style.format(
                {col:"{:.1f}" if 'PCT' not in col else "{:.1%}" for col in numeric_cols}
            ).background_gradient(subset=[selected_def_metric], 
                                 cmap='RdYlGn_r' if ascending else 'RdYlGn'),
            use_container_width=True
        )

    # ---------- Team Compare Tab ----------
    with tab8:
        st.markdown("### Head-to-Head Team Comparison")
        
        teams = st.multiselect(
            "Select teams to compare (2-5 teams recommended)", 
            options=sorted(df_team_avg['TEAM_NAME'].unique()), 
            default=sorted(df_team_avg['TEAM_NAME'].unique())[:3]
        )
        
        if len(teams) >= 2:
            df_selected = df_advanced[df_advanced['TEAM_NAME'].isin(teams)].copy()

            # Radar chart
            st.plotly_chart(create_radar_chart(df_advanced, teams, 
                           ['PTS', 'REB', 'AST', 'FG_PCT', 'TS_PCT', 'AST_TO_RATIO']), 
                          use_container_width=True)

            # Comparison categories
            comp_category = st.radio(
                "Comparison Category:",
                ["Core Stats", "Shooting", "Advanced Metrics", "Defense"],
                horizontal=True
            )
            
            if comp_category == "Core Stats":
                comp_cols = ['TEAM_NAME','PTS','REB','AST','STL','BLK','TOV','PLUS_MINUS']
            elif comp_category == "Shooting":
                comp_cols = ['TEAM_NAME','FGM','FGA','FG_PCT','FG3M','FG3A','FG3_PCT','FG2M','FG2A']
            elif comp_category == "Advanced Metrics":
                comp_cols = ['TEAM_NAME','TS_PCT','EFG_PCT','AST_TO_RATIO','PTS_PER_ATT','OREB_PCT']
            else:  # Defense
                # Merge defense stats
                df_selected = df_selected.merge(
                    df_defense_avg[['TEAM_NAME','OPP_PTS','OPP_FG_PCT','OPP_TOV','OPP_PLUS_MINUS']], 
                    on='TEAM_NAME', 
                    how='left'
                )
                comp_cols = ['TEAM_NAME','OPP_PTS','OPP_FG_PCT','OPP_TOV','OPP_PLUS_MINUS']

            # Highlight leaders
            numeric_cols = [col for col in comp_cols if col != 'TEAM_NAME']
            
            def highlight_leader(s):
                if s.name in ['TOV', 'OPP_PTS', 'OPP_FG_PCT']:  # Lower is better
                    is_leader = s == s.min()
                elif s.name in ['OPP_TOV']:  # Higher is better for turnovers forced
                    is_leader = s == s.max()
                else:  # Higher is better
                    is_leader = s == s.max()
                return ['background-color: #90EE90; font-weight: bold' if v else '' for v in is_leader]

            st.dataframe(
                df_selected[comp_cols].style.apply(highlight_leader, subset=numeric_cols)
                        .format({col: "{:.1f}" if col not in ['FG_PCT','FG3_PCT','FT_PCT','TS_PCT',
                                                               'EFG_PCT','OREB_PCT','OPP_FG_PCT'] 
                                else "{:.1%}" for col in numeric_cols}),
                use_container_width=True
            )
            
            # Side-by-side comparison charts
            st.markdown("---")
            st.markdown("#### Visual Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                metric1 = st.selectbox("Select first metric:", 
                                      ['PTS','REB','AST','FG_PCT','FG3_PCT','TS_PCT','AST_TO_RATIO'],
                                      key='metric1')
                fig1 = px.bar(df_selected, x='TEAM_NAME', y=metric1, 
                            title=f'{metric1} Comparison',
                            color=metric1, color_continuous_scale='Viridis')
                fig1.update_layout(showlegend=False)
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                metric2 = st.selectbox("Select second metric:", 
                                      ['PTS','REB','AST','FG_PCT','FG3_PCT','TS_PCT','AST_TO_RATIO'],
                                      index=1, key='metric2')
                fig2 = px.bar(df_selected, x='TEAM_NAME', y=metric2, 
                            title=f'{metric2} Comparison',
                            color=metric2, color_continuous_scale='Plasma')
                fig2.update_layout(showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)
            
            # Head-to-head matchup history if available
            st.markdown("---")
            st.markdown("#### Head-to-Head Matchup History")
            
            if len(teams) == 2:
                team1, team2 = teams[0], teams[1]
                
                # Filter head-to-head games between the two selected teams
                def get_team_abbr(team_name):
                    return next((k for k, v in TEAM_ABBR_NAME_DICT.items() if v == team_name), None)

                team1_abbr = get_team_abbr(team1)
                team2_abbr = get_team_abbr(team2)

                if team1_abbr and team2_abbr:
                    h2h_games = df[
                        ((df['TEAM_NAME'] == team1) & (df['OPP_TEAM_ABBR'] == team2_abbr)) |
                        ((df['TEAM_NAME'] == team2) & (df['OPP_TEAM_ABBR'] == team1_abbr))
                    ].copy()

                    if not h2h_games.empty:
                        h2h_games = h2h_games.sort_values('GAME_DATE')

                        # Group both teams' entries per GAME_ID to form a single record per matchup
                        combined = (
                            h2h_games.groupby('GAME_ID')
                            .apply(lambda g: pd.Series({
                                'GAME_DATE': g['GAME_DATE'].iloc[0],
                                'HOME_TEAM': g.loc[g['HOME_AWAY'] == 'HOME', 'TEAM_NAME'].values[0] if 'HOME' in g['HOME_AWAY'].values else g['TEAM_NAME'].iloc[0],
                                'AWAY_TEAM': g.loc[g['HOME_AWAY'] == 'AWAY', 'TEAM_NAME'].values[0] if 'AWAY' in g['HOME_AWAY'].values else g['TEAM_NAME'].iloc[-1],
                                'HOME_PTS': g.loc[g['HOME_AWAY'] == 'HOME', 'PTS'].values[0] if 'HOME' in g['HOME_AWAY'].values else None,
                                'AWAY_PTS': g.loc[g['HOME_AWAY'] == 'AWAY', 'PTS'].values[0] if 'AWAY' in g['HOME_AWAY'].values else None,
                                'WINNER': g.loc[g['WL'] == 'W', 'TEAM_NAME'].values[0] if 'W' in g['WL'].values else None,
                            }))
                            .reset_index(drop=True)
                        )

                        team1_wins = (combined['WINNER'] == team1).sum()
                        team2_wins = (combined['WINNER'] == team2).sum()
                        total_games = len(combined)

                        col1, col2, col3 = st.columns(3)
                        col1.metric(f"{team1} Wins", team1_wins)
                        col2.metric("Total Matchups", total_games)
                        col3.metric(f"{team2} Wins", team2_wins)

                        st.markdown("##### Recent Matchups")
                        h2h_display = combined[['GAME_DATE', 'HOME_TEAM', 'HOME_PTS', 'AWAY_TEAM', 'AWAY_PTS', 'WINNER']].tail(10)
                        st.dataframe(h2h_display, use_container_width=True)
                    else:
                        st.info("No head-to-head matchups found between these teams in the dataset.")
                else:
                    st.warning("Invalid team selection — could not find abbreviations.")


            else:
                st.info("Select exactly 2 teams to see head-to-head matchup history.")
        else:
            st.info("Please select at least 2 teams to compare.")