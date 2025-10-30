import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

def scrape_all_games_player_stats(season="2025-26"):
    lg_log = leaguegamelog.LeagueGameLog(season=season, player_or_team_abbreviation="P")
    df = lg_log.get_data_frames()[0]

    df['FG2M'] = df['FGM'] - df['FG3M']
    df['FG2A'] = df['FGA'] - df['FG3A']
    df['FG2_PTS'] = df['FG2M'] * 2
    df['FG3_PTS'] = df['FG3M'] * 3

    df = df.fillna(0)
    return df


def get_player_totals(df_players):
    pivot_player_totals = pd.pivot_table(
        df_players,
        index=['PLAYER_NAME', 'PLAYER_ID', 'TEAM_ABBREVIATION', 'TEAM_NAME', 'TEAM_ID'],
        aggfunc='sum'
    ).reset_index()

    drop_cols = {
        'FT_PCT', 'FG3_PCT', 'FG_PCT',
        'FRAC_ATT_2PT', 'FRAC_ATT_3PT', 'FRAC_MK_2PT', 'FRAC_MK_3PT',
        'FRAC_PTS_2PT', 'FRAC_PTS_3PT', 'FRAC_PTS_FT', 'VIDEO_AVAILABLE',
        'GAME_DATE', 'GAME_ID', 'MATCHUP', 'SEASON_ID', 'WL'
    }

    df_player_totals = pivot_player_totals.drop(columns=drop_cols, errors='ignore')
    return df_player_totals


def get_player_averages(df_players):
    games_played = df_players.groupby(['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ABBREVIATION']).size().reset_index(name='GP')
    totals = get_player_totals(df_players)
    merged = totals.merge(games_played, on=['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ABBREVIATION'], how='left')

    numeric_cols = merged.select_dtypes(include='number').columns.drop('GP', errors='ignore')
    cols_to_exclude = ['GP', 'PLAYER_ID', 'TEAM_ID']
    cols_to_average = [col for col in numeric_cols if col not in cols_to_exclude]

    # Calculate averages
    for col in cols_to_average:
        merged[col] = merged[col] / merged['GP']

    merged = merged.round(2)
    return merged
