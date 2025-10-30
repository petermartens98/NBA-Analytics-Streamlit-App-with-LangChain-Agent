# players_tab.py
import streamlit as st
from pathlib import Path
from datetime import datetime
import pandas as pd
import plotly.express as px
from utils.scrape_player_game_stats import (
    scrape_all_games_player_stats,
    get_player_totals,
    get_player_averages
)
from utils.data_helpers.load_data import load_data

DATA_FILE = Path("latest_data/players.csv")
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
CSV_DATE_FORMAT = '%Y%m%d'

def save_data(df):
    """Save player data to CSV file."""
    try:
        DATA_FILE.parent.mkdir(exist_ok=True, parents=True)
        df.to_csv(DATA_FILE, index=False)
        return True
    except Exception as e:
        st.error(f"Error saving data: {e}")
        return False


def render_stat_leaders(df_display, view_mode):
    """Render stat leaders visualization with coaching context."""
    if df_display is None or df_display.empty:
        return

    # Define stat categories based on available columns
    stat_categories = []
    stat_map = {
        'PTS': ('Points', '🔥 Who should we gameplan for?'),
        'AST': ('Assists', '🧠 Who runs their offense?'),
        'REB': ('Rebounds', '💪 Who controls the glass?'),
        'STL': ('Steals', '🏃 Who creates turnovers?'),
        'BLK': ('Blocks', '🛡️ Who protects the rim?'),
        'FG3M': ('3-Pointers Made', '🎯 Who spreads the floor?'),
        'PLUS_MINUS': ('+/-', '⚖️ Who are the winning players?'),
        'TOV': ('Turnovers', '⚠️ Who can we pressure?'),
        'PF': ('Fouls', '🎭 Who gets in foul trouble?'),
        'FANTASY_PTS': ('Fantasy Points', '⭐ Who are the true all-around players?')
    }
    
    for col, (label, context) in stat_map.items():
        if col in df_display.columns:
            stat_categories.append((col, label, context))
    
    if not stat_categories:
        st.info("No statistical categories available for visualization.")
        return
    
    # Stat selector with coaching context
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        selected_stat_label = st.selectbox(
            "Select Stat Category",
            [label for _, label, _ in stat_categories],
            key="stat_leader_selector"
        )
    
    # Find the column name and context for selected stat
    selected_stat = None
    selected_context = ""
    for col, label, context in stat_categories:
        if label == selected_stat_label:
            selected_stat = col
            selected_context = context
            break
    
    with col3:
        top_n = st.slider("Top N", min_value=5, max_value=20, value=10, key="top_n_slider")
    
    # Get top players for selected stat
    if selected_stat and selected_stat in df_display.columns:
        # Filter out zero/null values for certain stats
        df_viz = df_display.copy()
        
        # For turnovers, show descending (most turnovers)
        if selected_stat == 'TOV':
            top_players = df_viz[df_viz[selected_stat] > 0].nlargest(top_n, selected_stat)[['PLAYER_NAME', 'TEAM_NAME', selected_stat]]
            chart_title = f"⚠️ Turnover-Prone Players (Defensive Targets)"
        elif selected_stat == 'PF':
            top_players = df_viz[df_viz[selected_stat] > 0].nlargest(top_n, selected_stat)[['PLAYER_NAME', 'TEAM_NAME', selected_stat]]
            chart_title = f"🎭 Players in Foul Trouble (Attack Them)"
        else:
            top_players = df_viz.nlargest(top_n, selected_stat)[['PLAYER_NAME', 'TEAM_NAME', selected_stat]]
            chart_title = f"Top {top_n} Players - {selected_stat_label}"
        
        if not top_players.empty:
            # Create horizontal bar chart
            color_scale = 'Reds' if selected_stat in ['TOV', 'PF'] else 'Blues'
            
            fig = px.bar(
                top_players,
                y='PLAYER_NAME',
                x=selected_stat,
                orientation='h',
                color=selected_stat,
                color_continuous_scale=color_scale,
                text=selected_stat,
                hover_data={'TEAM_NAME': True, 'PLAYER_NAME': False, selected_stat: ':.2f'}
            )
            
            fig.update_traces(
                texttemplate='%{text:.1f}',
                textposition='outside'
            )
            
            fig.update_layout(
                title=chart_title,
                xaxis_title=selected_stat_label,
                yaxis_title="",
                yaxis={'categoryorder': 'total ascending'},
                height=max(400, top_n * 35),
                showlegend=False,
                coloraxis_showscale=False
            )
            
            st.plotly_chart(fig, use_container_width=True)

def render_additional_visualizations(df_display, view_mode):
    """Render advanced statistical visualizations with coaching insights."""
    if df_display is None or df_display.empty:
        return
    
    st.markdown("---")
    st.markdown("### 📈 Advanced Analytics")
    
    viz_tabs = st.tabs([
        "📈 Statistical Leaders",
        "🎯 Shot Selection",
        "🏆 Player Archetypes",
        "⚡ Impact Players",
        "🔥 Performance Metrics"
    ])

    # Tab 1: Statistical Leaders
    with viz_tabs[0]:        
        render_stat_leaders(df_display, view_mode)
    
    # Tab 2: Shot Selection & Efficiency
    with viz_tabs[1]:
        st.markdown("#### Shot Selection Analysis")
        
        if all(col in df_display.columns for col in ['FGA', 'FGM', 'FG3A', 'FG3M']):
            # Calculate shooting percentages
            df_viz = df_display[df_display['FGA'] >= 5].copy()  # Min 5 attempts
            
            if not df_viz.empty:
                df_viz['FG_PCT'] = df_viz['FGM'] / df_viz['FGA']
                df_viz['FG3_PCT'] = df_viz['FG3M'] / df_viz['FG3A'].replace(0, 1)
                df_viz['FG2A'] = df_viz['FGA'] - df_viz['FG3A']
                df_viz['FG2M'] = df_viz['FGM'] - df_viz['FG3M']
                df_viz['FG2_PCT'] = df_viz['FG2M'] / df_viz['FG2A'].replace(0, 1)
                
                # Calculate shot distribution
                df_viz['2PT_FREQ'] = (df_viz['FG2A'] / df_viz['FGA'] * 100)
                df_viz['3PT_FREQ'] = (df_viz['FG3A'] / df_viz['FGA'] * 100)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # 2PT vs 3PT frequency with FG% color
                    fig = px.scatter(
                        df_viz.nlargest(40, 'FGA'),
                        x='3PT_FREQ',
                        y='2PT_FREQ',
                        size='FGA',
                        color='FG_PCT',
                        hover_name='PLAYER_NAME',
                        hover_data={
                            'TEAM_NAME': True, 
                            'FGA': True, 
                            'FG_PCT': ':.1%',
                            'FG2_PCT': ':.1%',
                            'FG3_PCT': ':.1%',
                            '2PT_FREQ': ':.1f',
                            '3PT_FREQ': ':.1f'
                        },
                        title='Shot Distribution: 2PT vs 3PT Frequency',
                        labels={'3PT_FREQ': '3-Point Frequency (%)', '2PT_FREQ': '2-Point Frequency (%)'},
                        color_continuous_scale='RdYlGn',
                        range_color=[0.3, 0.6]
                    )
                    fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5)
                    fig.add_vline(x=50, line_dash="dash", line_color="gray", opacity=0.5)
                    fig.update_layout(height=450)
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("**Coaching Insight**: Top-left = Paint attackers | Bottom-right = Perimeter players")
                
                with col2:
                    # True Shooting Efficiency
                    df_viz['TS_PCT'] = df_viz['PTS'] / (2 * (df_viz['FGA'] + 0.44 * df_viz['FTA']))
                    df_viz['TS_PCT'] = df_viz['TS_PCT'].clip(0, 1).fillna(0)
                    
                    top_efficient = df_viz[df_viz['FGA'] >= 10].nlargest(15, 'TS_PCT')
                    
                    if not top_efficient.empty:
                        fig = px.bar(
                            top_efficient,
                            y='PLAYER_NAME',
                            x='TS_PCT',
                            orientation='h',
                            color='TS_PCT',
                            title='True Shooting % Leaders (Min 10 FGA)',
                            labels={'TS_PCT': 'True Shooting %'},
                            color_continuous_scale='Greens',
                            text='TS_PCT'
                        )
                        fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
                        fig.update_layout(height=450, yaxis={'categoryorder': 'total ascending'}, coloraxis_showscale=False)
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption("**TS%** accounts for 3PT value and free throws - best efficiency metric")
        
        # Additional shot charts
        col1, col2 = st.columns(2)
        
        with col1:
            # 3PT shooting volume vs efficiency
            if all(col in df_display.columns for col in ['FG3A', 'FG3M']):
                df_3pt = df_display[df_display['FG3A'] >= 3].copy()
                if not df_3pt.empty:
                    df_3pt['FG3_PCT'] = df_3pt['FG3M'] / df_3pt['FG3A']
                    
                    fig = px.scatter(
                        df_3pt.nlargest(40, 'FG3M'),
                        x='FG3A',
                        y='FG3_PCT',
                        size='FG3M',
                        color='FG3M',
                        hover_name='PLAYER_NAME',
                        hover_data={'TEAM_NAME': True, 'FG3A': True, 'FG3M': True, 'FG3_PCT': ':.1%'},
                        title='3-Point Shooting: Volume vs Accuracy',
                        labels={'FG3A': '3-Point Attempts', 'FG3_PCT': '3-Point %'},
                        color_continuous_scale='Oranges'
                    )
                    league_avg = df_3pt['FG3_PCT'].mean()
                    fig.add_hline(y=league_avg, line_dash="dash", line_color="red", 
                                 annotation_text=f"League Avg: {league_avg:.1%}", opacity=0.7)
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Free throw rate (FTA/FGA) - shows who gets to the line
            if all(col in df_display.columns for col in ['FTA', 'FGA', 'FTM']):
                df_ft = df_display[df_display['FGA'] >= 5].copy()
                if not df_ft.empty:
                    df_ft['FT_RATE'] = df_ft['FTA'] / df_ft['FGA']
                    df_ft['FT_PCT'] = df_ft['FTM'] / df_ft['FTA'].replace(0, 1)
                    
                    top_ft = df_ft.nlargest(15, 'FT_RATE')
                    
                    fig = px.bar(
                        top_ft,
                        y='PLAYER_NAME',
                        x='FT_RATE',
                        orientation='h',
                        color='FT_PCT',
                        title='Free Throw Rate Leaders (FTA/FGA)',
                        labels={'FT_RATE': 'FT Rate', 'FT_PCT': 'FT%'},
                        color_continuous_scale='Blues',
                        text='FT_RATE',
                        hover_data={'FT_PCT': ':.1%'}
                    )
                    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                    fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("**High FT Rate** = Gets to the line often (aggressive attackers)")
    
    # Tab 3: Player Archetypes (Enhanced with deeper analysis)
    with viz_tabs[2]:
        st.markdown("#### Player DNA & Role Classification")
        
        if all(col in df_display.columns for col in ['PTS', 'AST', 'REB']):
            df_viz = df_display.copy()
            
            # Normalize stats to classify players
            for col in ['PTS', 'AST', 'REB', 'STL', 'BLK', 'FG3M']:
                if col in df_viz.columns:
                    col_min = df_viz[col].min()
                    col_max = df_viz[col].max()
                    if col_max > col_min:
                        df_viz[f'{col}_norm'] = (df_viz[col] - col_min) / (col_max - col_min)
                    else:
                        df_viz[f'{col}_norm'] = 0
            
            # Enhanced archetype classification with more NBA-specific roles
            def classify_archetype(row):
                pts = row.get('PTS_norm', 0)
                ast = row.get('AST_norm', 0)
                reb = row.get('REB_norm', 0)
                stl = row.get('STL_norm', 0)
                blk = row.get('BLK_norm', 0)
                fg3 = row.get('FG3M_norm', 0)
                
                # Elite scoring guard (high pts, high 3s, low reb)
                if pts > 0.7 and fg3 > 0.6 and reb < 0.4:
                    return '🎯 Sharpshooter'
                # Slashing scorer (high pts, low 3s, high steals)
                elif pts > 0.7 and fg3 < 0.4 and stl > 0.5:
                    return '⚡ Slasher'
                # Primary ball handler
                elif ast > 0.8:
                    return '🧠 Floor General'
                # Secondary playmaker
                elif ast > 0.6 and pts > 0.5:
                    return '🔀 Combo Guard'
                # Interior presence (high reb, high blk)
                elif reb > 0.7 and blk > 0.5:
                    return '🏰 Rim Protector'
                # Pure rebounder
                elif reb > 0.75:
                    return '💪 Glass Cleaner'
                # Two-way wing (balanced scoring/defense)
                elif pts > 0.6 and (stl > 0.6 or blk > 0.5):
                    return '🛡️ Two-Way Wing'
                # Versatile forward (pts, reb, ast all solid)
                elif pts > 0.5 and reb > 0.5 and ast > 0.4:
                    return '⭐ Point Forward'
                # Traditional big (reb + scoring)
                elif reb > 0.6 and pts > 0.6:
                    return '🏀 Scoring Big'
                # 3&D specialist
                elif fg3 > 0.5 and (stl > 0.5 or blk > 0.4) and pts < 0.6:
                    return '🎯 3&D Specialist'
                # Energy/hustle player
                elif (stl > 0.5 or blk > 0.5) and reb > 0.4:
                    return '⚡ Energy Guy'
                else:
                    return '🔧 Role Player'
            
            df_viz['Archetype'] = df_viz.apply(classify_archetype, axis=1)
            
            # Create archetype summary with key stats
            archetype_summary = df_viz.groupby('Archetype').agg({
                'PTS': 'mean',
                'AST': 'mean',
                'REB': 'mean',
                'PLAYER_NAME': 'count'
            }).round(1).reset_index()
            archetype_summary.columns = ['Archetype', 'Avg PTS', 'Avg AST', 'Avg REB', 'Count']
            archetype_summary = archetype_summary.sort_values('Count', ascending=False)
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                # Interactive scatter with archetype colors
                top_players = df_viz.nlargest(60, 'PTS')
                
                fig = px.scatter(
                    top_players,
                    x='PTS',
                    y='AST',
                    size='REB',
                    color='Archetype',
                    hover_name='PLAYER_NAME',
                    hover_data={
                        'TEAM_NAME': True, 
                        'PTS': ':.1f', 
                        'AST': ':.1f', 
                        'REB': ':.1f',
                        'STL': ':.1f' if 'STL' in top_players.columns else False,
                        'BLK': ':.1f' if 'BLK' in top_players.columns else False
                    },
                    title='Player Archetypes: Scoring vs Playmaking (Bubble = Rebounds)',
                    labels={'PTS': 'Points', 'AST': 'Assists'},
                    color_discrete_map={
                        '🎯 Sharpshooter': '#FF6B35',
                        '⚡ Slasher': '#FF0000',
                        '🧠 Floor General': '#004E98',
                        '🔀 Combo Guard': '#3A86FF',
                        '🏰 Rim Protector': '#2D3142',
                        '💪 Glass Cleaner': '#556B2F',
                        '🛡️ Two-Way Wing': '#8338EC',
                        '⭐ Point Forward': '#FFD700',
                        '🏀 Scoring Big': '#FF8C00',
                        '🎯 3&D Specialist': '#06D6A0',
                        '⚡ Energy Guy': '#F72585',
                        '🔧 Role Player': '#888888'
                    }
                )
                fig.update_layout(height=520, legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02))
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Archetype breakdown table
                st.markdown("##### Archetype Breakdown")
                st.dataframe(
                    archetype_summary,
                    hide_index=True,
                    use_container_width=True,
                    height=350
                )
                
                st.caption("**Coaching Strategy**: Match archetypes to your system needs")
        
        # Position-based analysis
        st.markdown("---")
        st.markdown("#### Positional Versatility Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Size vs Skill (Height proxy: rebounds vs assists)
            if all(col in df_display.columns for col in ['REB', 'AST', 'PTS']):
                df_pos = df_display[df_display['PTS'] >= 5].copy()
                
                if not df_pos.empty:
                    # Create position estimates
                    def estimate_position(row):
                        reb = row['REB']
                        ast = row['AST']
                        if reb < 3 and ast > 4:
                            return 'Guard-Type'
                        elif reb > 8:
                            return 'Center-Type'
                        elif reb > 5 and ast > 3:
                            return 'Forward-Type'
                        else:
                            return 'Wing-Type'
                    
                    df_pos['Position_Type'] = df_pos.apply(estimate_position, axis=1)
                    
                    fig = px.scatter(
                        df_pos.nlargest(50, 'PTS'),
                        x='AST',
                        y='REB',
                        size='PTS',
                        color='Position_Type',
                        hover_name='PLAYER_NAME',
                        hover_data={'TEAM_NAME': True, 'PTS': ':.1f', 'AST': ':.1f', 'REB': ':.1f'},
                        title='Positional Profile: Playmaking vs Size',
                        labels={'AST': 'Assists (Playmaking)', 'REB': 'Rebounds (Size/Positioning)'},
                        color_discrete_map={
                            'Guard-Type': '#FF6B6B',
                            'Wing-Type': '#4ECDC4',
                            'Forward-Type': '#45B7D1',
                            'Center-Type': '#96CEB4'
                        }
                    )
                    fig.update_layout(height=450)
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("**Positionless Basketball**: Modern NBA values versatility over traditional positions")
        
        with col2:
            # Offensive load distribution
            if all(col in df_display.columns for col in ['FGA', 'FTA', 'AST']):
                df_load = df_display[df_display['PTS'] >= 5].copy()
                
                if not df_load.empty:
                    # Calculate usage proxies
                    df_load['Usage_Proxy'] = df_load['FGA'] + 0.44 * df_load['FTA'] + df_load['AST']
                    df_load['Scorer_Role'] = df_load['FGA'] / (df_load['FGA'] + df_load['AST'] + 1)
                    df_load['Facilitator_Role'] = df_load['AST'] / (df_load['FGA'] + df_load['AST'] + 1)
                    
                    top_usage = df_load.nlargest(40, 'Usage_Proxy')
                    
                    fig = px.scatter(
                        top_usage,
                        x='Scorer_Role',
                        y='Facilitator_Role',
                        size='Usage_Proxy',
                        color='PTS',
                        hover_name='PLAYER_NAME',
                        hover_data={'TEAM_NAME': True, 'PTS': ':.1f', 'FGA': True, 'AST': True},
                        title='Offensive Role: Scorer vs Facilitator',
                        labels={'Scorer_Role': 'Shooting Role', 'Facilitator_Role': 'Passing Role'},
                        color_continuous_scale='Reds'
                    )
                    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.3)
                    fig.add_vline(x=0.5, line_dash="dash", line_color="gray", opacity=0.3)
                    fig.update_layout(height=450)
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("**Top-right** = High usage scorers | **Bottom-right** = Pure shooters | **Top-left** = Pure facilitators")
        
        # Detailed comparison table
        st.markdown("---")
        st.markdown("#### Find Your Archetype")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            if 'Archetype' in df_viz.columns:
                archetypes_list = ['All'] + sorted(df_viz['Archetype'].unique().tolist())
                selected_archetype = st.selectbox(
                    "Filter by Archetype",
                    archetypes_list,
                    key="archetype_filter"
                )
        
        with col2:
            if selected_archetype != 'All':
                filtered_df = df_viz[df_viz['Archetype'] == selected_archetype].copy()
            else:
                filtered_df = df_viz.copy()
            
            # Show top players in this archetype
            display_cols = ['PLAYER_NAME', 'TEAM_NAME', 'Archetype', 'PTS', 'AST', 'REB']
            if 'STL' in filtered_df.columns:
                display_cols.append('STL')
            if 'BLK' in filtered_df.columns:
                display_cols.append('BLK')
            if 'FG3M' in filtered_df.columns:
                display_cols.append('FG3M')
            
            available_cols = [col for col in display_cols if col in filtered_df.columns]
            
            st.dataframe(
                filtered_df.nlargest(20, 'PTS')[available_cols],
                hide_index=True,
                use_container_width=True,
                height=300
            )
            
            st.caption(f"Showing top 20 players {f'in {selected_archetype}' if selected_archetype != 'All' else 'across all archetypes'}")
    
    # Tab 4: Impact Players (Advanced Metrics)
    with viz_tabs[3]:
        st.markdown("#### High-Impact Player Identification")
        
        if all(col in df_display.columns for col in ['PTS', 'AST', 'REB', 'STL', 'BLK', 'TOV']):
            df_viz = df_display.copy()
            
            # Calculate impact score
            df_viz['Impact_Score'] = (
                df_viz['PTS'] * 1.0 +
                df_viz['AST'] * 1.5 +
                df_viz['REB'] * 1.2 +
                df_viz['STL'] * 2.0 +
                df_viz['BLK'] * 2.0 -
                df_viz['TOV'] * 1.5
            )
            
            # Calculate per-minute efficiency
            if 'MIN' in df_viz.columns:
                df_viz['PTS_PER_MIN'] = df_viz['PTS'] / df_viz['MIN'].replace(0, 1)
                df_viz['Impact_Per_Min'] = df_viz['Impact_Score'] / df_viz['MIN'].replace(0, 1)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Top impact players
                top_impact = df_viz.nlargest(15, 'Impact_Score')
                
                fig = px.bar(
                    top_impact,
                    y='PLAYER_NAME',
                    x='Impact_Score',
                    orientation='h',
                    color='Impact_Score',
                    title='Overall Impact Score Leaders',
                    labels={'Impact_Score': 'Impact Score'},
                    color_continuous_scale='Reds',
                    text='Impact_Score'
                )
                fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
                fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'}, coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)
                st.caption("**Impact Score** = Weighted combination of all positive contributions")
            
            with col2:
                # Defensive impact (Stocks - Steals + Blocks)
                df_viz['Stocks'] = df_viz['STL'] + df_viz['BLK']
                defensive_leaders = df_viz[df_viz['Stocks'] > 0].nlargest(15, 'Stocks')
                
                if not defensive_leaders.empty:
                    fig = px.bar(
                        defensive_leaders,
                        y='PLAYER_NAME',
                        x='Stocks',
                        orientation='h',
                        color='Stocks',
                        title='Defensive Playmakers (Stocks)',
                        labels={'Stocks': 'Steals + Blocks'},
                        color_continuous_scale='Blues',
                        text='Stocks'
                    )
                    fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
                    fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'}, coloraxis_showscale=False)
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("**'Stocks'** = Steals + Blocks")
        
        # Fantasy and efficiency metrics
        col1, col2 = st.columns(2)
        
        with col1:
            if 'FANTASY_PTS' in df_display.columns:
                top_fantasy = df_display.nlargest(15, 'FANTASY_PTS')
                
                fig = px.bar(
                    top_fantasy,
                    y='PLAYER_NAME',
                    x='FANTASY_PTS',
                    orientation='h',
                    color='FANTASY_PTS',
                    title='Fantasy Points Leaders',
                    labels={'FANTASY_PTS': 'Fantasy Points'},
                    color_continuous_scale='Purples',
                    text='FANTASY_PTS'
                )
                fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
                fig.update_layout(height=450, yaxis={'categoryorder': 'total ascending'}, coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)
                st.caption("**Fantasy Pts** = Overall statistical production")
        
        with col2:
            if 'MIN' in df_display.columns and 'PTS' in df_display.columns:
                df_eff = df_display[df_display['MIN'] >= 10].copy()
                if not df_eff.empty:
                    df_eff['PTS_PER_MIN'] = df_eff['PTS'] / df_eff['MIN']
                    top_eff = df_eff.nlargest(15, 'PTS_PER_MIN')
                    
                    fig = px.bar(
                        top_eff,
                        y='PLAYER_NAME',
                        x='PTS_PER_MIN',
                        orientation='h',
                        color='PTS_PER_MIN',
                        title='Points Per Minute Leaders (Min 10 MIN)',
                        labels={'PTS_PER_MIN': 'Points Per Minute'},
                        color_continuous_scale='Oranges',
                        text='PTS_PER_MIN'
                    )
                    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                    fig.update_layout(height=450, yaxis={'categoryorder': 'total ascending'}, coloraxis_showscale=False)
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("**Pts/Min** = Scoring efficiency (great for bench impact players)")
    
    # Tab 5: Performance Metrics
    with viz_tabs[4]:
        st.markdown("#### Performance Consistency & Efficiency")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Usage vs Efficiency
            if all(col in df_display.columns for col in ['FGA', 'FGM', 'PTS']):
                df_viz = df_display[df_display['FGA'] >= 5].copy()
                
                if not df_viz.empty:
                    df_viz['FG_PCT'] = df_viz['FGM'] / df_viz['FGA']
                    
                    fig = px.scatter(
                        df_viz.nlargest(40, 'PTS'),
                        x='FGA',
                        y='FG_PCT',
                        size='PTS',
                        color='PTS',
                        hover_name='PLAYER_NAME',
                        hover_data={'TEAM_NAME': True, 'FGA': True, 'FG_PCT': ':.1%', 'PTS': ':.1f'},
                        title='Volume vs Efficiency: The Sweet Spot',
                        labels={'FGA': 'Field Goal Attempts (Usage)', 'FG_PCT': 'Field Goal %'},
                        color_continuous_scale='Hot'
                    )
                    # Add quadrant lines
                    median_fga = df_viz['FGA'].median()
                    median_pct = df_viz['FG_PCT'].median()
                    fig.add_hline(y=median_pct, line_dash="dash", line_color="white", opacity=0.3)
                    fig.add_vline(x=median_fga, line_dash="dash", line_color="white", opacity=0.3)
                    fig.update_layout(height=450)
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("**Top-Right Quadrant** = High volume, high efficiency (franchise players)")
        
        with col2:
            # Plus/Minus leaders
            if 'PLUS_MINUS' in df_display.columns:
                pm_leaders = df_display.nlargest(15, 'PLUS_MINUS')
                
                fig = px.bar(
                    pm_leaders,
                    y='PLAYER_NAME',
                    x='PLUS_MINUS',
                    orientation='h',
                    color='PLUS_MINUS',
                    title='Plus/Minus Leaders (Net Impact)',
                    labels={'PLUS_MINUS': 'Plus/Minus'},
                    color_continuous_scale='RdYlGn',
                    text='PLUS_MINUS'
                )
                fig.update_traces(texttemplate='%{text:+.1f}', textposition='outside')
                fig.update_layout(height=450, yaxis={'categoryorder': 'total ascending'}, coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)
                st.caption("**+/-** = Point differential when player is on court")
        
        # Additional performance metrics
        col1, col2 = st.columns(2)
        
        with col1:
            # Scoring efficiency (Points per shot attempt)
            if all(col in df_display.columns for col in ['PTS', 'FGA', 'FTA']):
                df_eff = df_display[df_display['FGA'] >= 5].copy()
                
                if not df_eff.empty:
                    df_eff['PTS_PER_ATTEMPT'] = df_eff['PTS'] / (df_eff['FGA'] + 0.44 * df_eff['FTA'])
                    top_eff = df_eff.nlargest(15, 'PTS_PER_ATTEMPT')
                    
                    fig = px.bar(
                        top_eff,
                        y='PLAYER_NAME',
                        x='PTS_PER_ATTEMPT',
                        orientation='h',
                        color='PTS_PER_ATTEMPT',
                        title='Points Per Shooting Possession (Efficiency)',
                        labels={'PTS_PER_ATTEMPT': 'Points Per Attempt'},
                        color_continuous_scale='Greens',
                        text='PTS_PER_ATTEMPT'
                    )
                    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                    fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'}, coloraxis_showscale=False)
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("**Elite efficiency** = ~1.20+ points per possession")
        
        with col2:
            # Games played consistency
            if 'GP' in df_display.columns:
                gp_dist = df_display['GP'].value_counts().sort_index().reset_index()
                gp_dist.columns = ['Games Played', 'Player Count']
                
                fig = px.bar(
                    gp_dist,
                    x='Games Played',
                    y='Player Count',
                    title='Games Played Distribution',
                    labels={'Games Played': 'Games Played', 'Player Count': 'Number of Players'},
                    color='Player Count',
                    color_continuous_scale='Blues'
                )
                fig.update_layout(height=400, showlegend=False, coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)
                st.caption("**Availability** = The best ability is availability")

def render_data_view(df):
    """Render the data view with totals/averages toggle."""
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        view_mode = st.radio(
            "Select View",
            ["Totals", "Averages"],
            horizontal=True,
            key="view_mode"
        )
    
    # Team filter
    selected_team = "All Teams"
    with col2:
        if 'TEAM_NAME' in df.columns:
            teams = sorted(df['TEAM_NAME'].dropna().unique())
            team_options = ["All Teams"] + list(teams)
            selected_team = st.selectbox("🏀 Team", team_options, key="team_filter")
            if selected_team != "All Teams":
                df = df[df['TEAM_NAME'] == selected_team]
    
    # Player search
    with col3:
        search_term = ""
        if 'PLAYER_NAME' in df.columns:
            search_term = st.text_input("🔍 Search Player", placeholder="Enter player name...")
            if search_term:
                df = df[df['PLAYER_NAME'].str.contains(search_term, case=False, na=False)]
    
    st.markdown("---")

    df_display = pd.DataFrame()
    try:
        if view_mode == "Totals":
            st.markdown("#### 📊 Player Totals")
            df_display = get_player_totals(df)
        else:
            st.markdown("#### 🧮 Player Averages (Per Game)")
            df_display = get_player_averages(df)
        
        if df_display.empty:
            st.warning("No data to display after filtering.")
            return None, view_mode
        
        st.dataframe(
            df_display,
            use_container_width=True,
            height=550,
            hide_index=True
        )
        
        st.caption(f"Showing {len(df_display)} player(s)")

        # Right-aligned last updated under the table
        if 'last_updated' in st.session_state and st.session_state.last_updated:
            last_updated_text = st.session_state.last_updated.strftime('%Y-%m-%d %H:%M:%S')
            st.markdown(
                f"<div style='text-align: right; font-size: 13px; color: gray; margin-top: 0.5rem;'>"
                f"Last retrieved: {last_updated_text}"
                f"</div>",
                unsafe_allow_html=True
            )
        
        # Add visualizations below the table
        render_additional_visualizations(df_display, view_mode)
        
        return df_display, view_mode
        
    except Exception as e:
        st.error(f"❌ Error processing data: {e}")
        with st.expander("🔍 View Error Details"):
            st.exception(e)
        return None, view_mode

def render_players_tab():
    st.markdown("### 🏀 Player Analytics")
    """Main function to render the players tab."""
    df, last_updated = load_data(data_file=DATA_FILE)

    # Initialize session state for df_display if not exists
    if 'df_display' not in st.session_state:
        st.session_state.df_display = None
    if 'current_view_mode' not in st.session_state:
        st.session_state.current_view_mode = "Raw"

    # Check if data exists
    if df.empty:
        st.info("📭 No player data available. Click **'Fetch Player Data'** to load.")

    # Render data view and update session state
    df_display, view_mode = render_data_view(df)
    st.session_state.df_display = df_display
    st.session_state.current_view_mode = view_mode