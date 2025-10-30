import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

team_abbr_name_dict = {
    'GSW':'Golden State Warriors', 'POR':'Portland Trail Blazers', 'SAC':'Sacramento Kings',
    'UTA':'Utah Jazz', 'MIA':'Miami Heat', 'DEN':'Denver Nuggets', 'MIN':'Minnesota Timberwolves',
    'PHI':'Philadelphia 76ers', 'NOP':'New Orleans Pelicans', 'ORL':'Orlando Magic', 'MIL':'Milwaukee Bucks',
    'CHI':'Chicago Bulls', 'DET':'Detroit Pistons', 'TOR':'Toronto Raptors', 'PHX':'Phoenix Suns',
    'LAL':'Los Angeles Lakers', 'ATL':'Atlanta Hawks', 'WAS':'Washington Wizards', 'MEM':'Memphis Grizzlies',
    'CLE':'Cleveland Cavaliers', 'LAC':'LA Clippers', 'BOS':'Boston Celtics', 'NYK':'New York Knicks',
    'IND':'Indiana Pacers', 'CHA':'Charlotte Hornets', 'SAS':'San Antonio Spurs', 'HOU':'Houston Rockets',
    'DAL':'Dallas Mavericks', 'OKC':'Oklahoma City Thunder', 'BKN':'Brooklyn Nets'
}

conferences = {
    'GSW':'WEST', 'POR':'WEST', 'SAC':'WEST', 'UTA':'WEST', 'DEN':'WEST', 'MIN':'WEST',
    'PHX':'WEST', 'LAL':'WEST', 'MEM':'WEST', 'LAC':'WEST', 'SAS':'WEST', 'HOU':'WEST', 'DAL':'WEST', 'OKC':'WEST',
    'MIA':'EAST', 'PHI':'EAST', 'NOP':'EAST', 'ORL':'EAST', 'MIL':'EAST', 'CHI':'EAST', 'DET':'EAST', 'TOR':'EAST',
    'ATL':'EAST', 'WAS':'EAST', 'CLE':'EAST', 'BOS':'EAST', 'NYK':'EAST', 'IND':'EAST', 'CHA':'EAST', 'BKN':'EAST'
}

def home_or_away(matchup):
    if matchup[4] == '@':
        return 'AWAY'
    elif matchup[4] == 'v':
        return 'HOME'
    return None

def scrape_all_games_team_stats(season="2025-26"):
    lg_log = leaguegamelog.LeagueGameLog(season=season, player_or_team_abbreviation="T")
    df = lg_log.get_data_frames()[0]

    # Basic derived stats
    df['FG2M'] = df['FGM'] - df['FG3M']
    df['FG2A'] = df['FGA'] - df['FG3A']
    df['FG2_PTS'] = df['FG2M'] * 2
    df['FG3_PTS'] = df['FG3M'] * 3

    # Fractions
    df['FRAC_ATT_2PT'] = df['FG2A'] / df['FGA']
    df['FRAC_ATT_3PT'] = df['FG3A'] / df['FGA']
    df['FRAC_MK_2PT'] = df['FG2M'] / df['FGM']
    df['FRAC_MK_3PT'] = df['FG3M'] / df['FGM']
    df['FRAC_PTS_2PT'] = df['FG2_PTS'] / df['PTS']
    df['FRAC_PTS_3PT'] = df['FG3_PTS'] / df['PTS']
    df['FRAC_PTS_FT'] = df['FTM'] / df['PTS']

    # Opponent info
    df['OPP_TEAM_ABBR'] = df['MATCHUP'].str.strip().str[-3:]
    df['OPP_PTS'] = df['PTS'] - df['PLUS_MINUS']

    # Date info
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df['MONTH'] = df['GAME_DATE'].dt.month
    df['YEAR'] = df['GAME_DATE'].dt.year

    # Home/Away
    df['HOME_AWAY'] = df['MATCHUP'].apply(home_or_away)

    # Conferences
    df['CONFERENCE'] = df['TEAM_ABBREVIATION'].map(conferences)
    df['OPP_CONFERENCE'] = df['OPP_TEAM_ABBR'].map(conferences)

    # Team names
    df['TEAM_NAME'] = df['TEAM_ABBREVIATION'].map(team_abbr_name_dict)

    # Optional: formatted matchup date
    df['DATE_MATCHUP'] = df['GAME_DATE'].dt.strftime('%m-%d') + ' ' + df['MATCHUP'].str[4:]

    df = df.fillna(0)
    return df