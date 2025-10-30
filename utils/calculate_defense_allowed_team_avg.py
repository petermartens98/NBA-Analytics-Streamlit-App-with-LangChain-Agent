import pandas as pd

def calculate_defense_allowed_team_avg(df: pd.DataFrame, team_abbr_name_dict: dict) -> pd.DataFrame:
    """
    Calculate the average defensive stats allowed by each team,
    including a league average row.
    """
    df = df.copy()
    
    # Keep only numeric columns for aggregation
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    
    # Pivot / group by opponent team
    pivot_defense = df.groupby('OPP_TEAM_ABBR')[numeric_cols].mean().reset_index()
    
    # Add league average row
    league_avg = pivot_defense[numeric_cols].mean()
    league_avg_row = pd.Series(league_avg)
    league_avg_row['OPP_TEAM_ABBR'] = 'NBA'
    pivot_defense.loc[len(pivot_defense)] = league_avg_row
    
    # Map full team names
    pivot_defense['TEAM_NAME'] = pivot_defense['OPP_TEAM_ABBR'].map(team_abbr_name_dict)
    # Explicitly set league average row name
    pivot_defense.loc[pivot_defense['OPP_TEAM_ABBR'] == 'NBA', 'TEAM_NAME'] = 'League Average'
    
    # Reorder columns
    cols_to_move = ['TEAM_NAME', 'OPP_TEAM_ABBR']
    pivot_defense = pivot_defense[cols_to_move + [c for c in pivot_defense.columns if c not in cols_to_move]]
    
    # Prefix numeric columns with 'OPP_' only if not already prefixed
    for col in numeric_cols:
        if not col.startswith('OPP_'):
            pivot_defense.rename(columns={col: f'OPP_{col}'}, inplace=True)
    
    # Rename opponent column to TEAM_ABBR
    pivot_defense.rename(columns={'OPP_TEAM_ABBR': 'TEAM_ABBR'}, inplace=True)
    
    # Ensure column names are unique
    pivot_defense = pivot_defense.loc[:, ~pivot_defense.columns.duplicated()]
    
    return pivot_defense
