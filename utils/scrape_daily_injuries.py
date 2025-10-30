import pandas as pd
import requests
from bs4 import BeautifulSoup

def scrape_daily_injuries(url="https://www.cbssports.com/nba/injuries/"):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.text, "html.parser")

    team_wrappers = soup.find_all("div", class_="TableBaseWrapper")
    team_data = []

    for team in team_wrappers:
        team_name = team.find("div", class_="TeamLogoNameLockup-name").get_text(strip=True)
        players = team.find_all('tr', class_="TableBase-bodyTr")

        for player in players:
            cells = player.find_all('td', class_="TableBase-bodyTd")
            player_data = {
                "Team": team_name,
                "Name": player.find('span', class_="CellPlayerName--long").get_text(strip=True),
                "POS": cells[1].get_text(strip=True),
                "Updated": player.find('span', class_="CellGameDate").get_text(strip=True),
                "Injury": cells[3].get_text(strip=True),
                "Status": cells[4].get_text(strip=True)
            }
            team_data.append(player_data)

    return pd.DataFrame(team_data).sort_values("Team")
