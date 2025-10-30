import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time
from nba_api.stats.library.http import NBAStatsHTTP
from nba_api.stats.static import teams
from nba_api.stats.endpoints._base import Endpoint
from nba_api.stats.library.parameters import Season, LeagueIDNullable


class CommonTeamRoster(Endpoint):
    endpoint = "commonteamroster"
    expected_data = {
        "Coaches": [
            "TEAM_ID", "SEASON", "COACH_ID", "FIRST_NAME", "LAST_NAME",
            "COACH_NAME", "IS_ASSISTANT", "COACH_TYPE", "SORT_SEQUENCE",
        ],
        "CommonTeamRoster": [
            "TeamID", "SEASON", "LeagueID", "PLAYER", "PLAYER_SLUG",
            "NUM", "POSITION", "HEIGHT", "WEIGHT", "BIRTH_DATE", "AGE",
            "EXP", "SCHOOL", "PLAYER_ID",
        ],
    }

    def __init__(
        self,
        team_id,
        season=Season.default,
        league_id_nullable=LeagueIDNullable.default,
        proxy=None,
        headers=None,
        timeout=5,
        get_request=True,
    ):
        self.proxy = proxy
        self.headers = headers
        self.timeout = timeout
        self.parameters = {
            "TeamID": team_id,
            "Season": season,
            "LeagueID": league_id_nullable,
        }
        if get_request:
            self.get_request()

    def get_request(self):
        self.nba_response = NBAStatsHTTP().send_api_request(
            endpoint=self.endpoint,
            parameters=self.parameters,
            proxy=self.proxy,
            headers=self.headers,
            timeout=self.timeout,
        )
        self.load_response()

    def load_response(self):
        data_sets = self.nba_response.get_data_sets()
        self.coaches = Endpoint.DataSet(data=data_sets["Coaches"])
        self.common_team_roster = Endpoint.DataSet(data=data_sets["CommonTeamRoster"])


def scrape_all_team_rosters(season="2025-26", max_retries=3):
    Path("latest_data").mkdir(exist_ok=True)
    players_list, coaches_list = [], []

    all_teams = teams.get_teams()
    retry_queue = []
    print(f"Fetching rosters for {season}")

    for team in tqdm(all_teams, desc="Initial fetch"):
        team_id, team_name = team["id"], team["full_name"]

        try:
            start_time = time.time()
            data = CommonTeamRoster(team_id=team_id, season=season, timeout=5)
            elapsed = time.time() - start_time
            if elapsed > 5:
                raise TimeoutError(f"Took too long ({elapsed:.1f}s)")

            players = data.common_team_roster.get_data_frame()
            coaches = data.coaches.get_data_frame()

            if not players.empty:
                players["TEAM_NAME"], players["TEAM_ID"] = team_name, team_id
                players_list.append(players)
            if not coaches.empty:
                coaches["TEAM_NAME"], coaches["TEAM_ID"] = team_name, team_id
                coaches_list.append(coaches)

            time.sleep(0.5)

        except Exception as e:
            print(f"⚠️ {team_name} failed: {e}")
            retry_queue.append(team)

    # Retry pass
    if retry_queue:
        print(f"\n🔁 Retrying {len(retry_queue)} failed teams...")
        for attempt in range(1, max_retries + 1):
            still_failed = []
            for team in retry_queue:
                team_id, team_name = team["id"], team["full_name"]
                try:
                    data = CommonTeamRoster(team_id=team_id, season=season, timeout=5)
                    players = data.common_team_roster.get_data_frame()
                    coaches = data.coaches.get_data_frame()
                    if not players.empty:
                        players["TEAM_NAME"], players["TEAM_ID"] = team_name, team_id
                        players_list.append(players)
                    if not coaches.empty:
                        coaches["TEAM_NAME"], coaches["TEAM_ID"] = team_name, team_id
                        coaches_list.append(coaches)
                except Exception as e:
                    print(f"❌ Retry {attempt} failed for {team_name}: {e}")
                    still_failed.append(team)
                time.sleep(0.5)
            if not still_failed:
                break
            retry_queue = still_failed

    df_players = pd.concat(players_list, ignore_index=True) if players_list else pd.DataFrame()
    df_coaches = pd.concat(coaches_list, ignore_index=True) if coaches_list else pd.DataFrame()

    df_players.to_csv("latest_data/team_rosters_players.csv", index=False)
    df_coaches.to_csv("latest_data/team_rosters_coaches.csv", index=False)

    print(f"✅ Saved {len(df_players)} players")
    print(f"✅ Saved {len(df_coaches)} coaches")

    if retry_queue:
        print("⚠️ Still failed teams:")
        for t in retry_queue:
            print(f" - {t['full_name']}")

    return df_players, df_coaches


if __name__ == "__main__":
    scrape_all_team_rosters("2025-26")
