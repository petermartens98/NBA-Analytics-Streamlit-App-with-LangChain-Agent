import pandas as pd

def calculate_team_averages(df):
    # Keep only numeric columns for aggregation
    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    # Pivot table using only numeric columns
    pivot_team_avg = pd.pivot_table(df, index=['TEAM_NAME','TEAM_ABBREVIATION'], values=numeric_cols, aggfunc='mean')
    df_team_avg = pivot_team_avg.reset_index()

    # Add league average row
    league_avg = df_team_avg[numeric_cols].mean()
    league_avg['TEAM_NAME'] = 'League Average'
    league_avg['TEAM_ABBREVIATION'] = 'NBA'
    df_team_avg.loc[len(df_team_avg)] = league_avg

    # Rename column
    df_team_avg = df_team_avg.rename(columns={'TEAM_ABBREVIATION':'TEAM_ABBR'})

    # Drop month/year if they exist
    df_team_avg = df_team_avg.drop([col for col in ['MONTH','YEAR'] if col in df_team_avg.columns], axis=1)

    # Reorder columns: TEAM_NAME, TEAM_ABBR, PTS first
    cols_to_move = ['TEAM_NAME', 'TEAM_ABBR', 'PTS']
    df_team_avg = df_team_avg[cols_to_move + [col for col in df_team_avg.columns if col not in cols_to_move]]

    return df_team_avg
