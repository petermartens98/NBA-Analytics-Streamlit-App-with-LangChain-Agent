import pandas as pd
from nba_api.stats.endpoints.commonallplayers import CommonAllPlayers

def scrape_all_players():
    # Initialize and automatically fetch data
    res = CommonAllPlayers(is_only_current_season=1)
    
    # Get the first DataFrame returned (CommonAllPlayers)
    df = res.get_data_frames()[0]
    
    print(df.head())
    return df

if __name__ == "__main__":
    df_players = scrape_all_players()
