import pandas as pd
from nba_api.stats.endpoints.commonplayerinfo import CommonPlayerInfo

def scrape_commmon_player_info():
    common_player_info = CommonPlayerInfo()
    print(common_player_info)

if __name__ == "__main__":
    scrape_commmon_player_info()