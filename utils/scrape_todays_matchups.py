import pandas as pd
import requests
from datetime import datetime

def scrape_todays_matchups():
    today_str = datetime.today().strftime("%Y%m%d")
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={today_str}"
    headers = {"User-Agent": "Mozilla/5.0"}

    r = requests.get(url, headers=headers)
    r.raise_for_status()
    data = r.json()

    events = data.get("events", [])

    games_data = []
    for event in events:
        game = {
            "Date": datetime.today().strftime("%m-%d-%y"),
            "Day": datetime.today().strftime("%A"),
            "Away": "",
            "Home": "",
            "Away Score": 0,
            "Home Score": 0,
            "Time": "TBD"
        }

        competitors = event.get("competitions", [])[0].get("competitors", [])
        for comp in competitors:
            team_name = comp.get("team", {}).get("shortDisplayName")
            score = int(comp.get("score") or 0)
            if comp.get("homeAway") == "away":
                game["Away"] = team_name
                game["Away Score"] = score
            else:
                game["Home"] = team_name
                game["Home Score"] = score

        status = event.get("competitions", [])[0].get("status", {}).get("type", {})
        game["Time"] = status.get("shortDetail", "TBD")


        games_data.append(game)

    return pd.DataFrame(games_data)
