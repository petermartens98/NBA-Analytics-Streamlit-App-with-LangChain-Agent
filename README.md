# NBA Analytics Streamlit App with LangChain Agent

### Project Overview

The **NBA Analytics Streamlit App** provides interactive dashboards for player, team, and game stats. It integrates a **LangChain Agent** for AI-powered insights and natural language queries, enabling users to explore performance metrics, head-to-head comparisons, and game-level analytics in real time.

The Project is Currently in Continuing Development, and Open to Collaboration.

---

### **Features**

| Feature                  | Description                                                                                                                    |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------ |
| **Today's Matchups**     | View today’s games with key stats and predictions.                                                                             |
| **Injuries**             | Track current player injuries and impact to team.                                                                              |
| **All Players**          | Overview of all player stats, performance trends, shot selection, player archetypes, and impact players; filterable on team.                                             |
| **Single Player**        | Detailed profile and analytics for an individual player; including game logs, career stats, performance trends, shooting analysis, and player impact.                    |
| **All Teams**          | Overview of all teams stats, shooting analysis, ball movement, home/away trends, defense and a team comparison tool                                        |
| **Single Team**        | Detailed overview and analytics for an individual team; including rosters, game logs, scoring efficiency and trends.                    |
| **Simulation**           | Simulate games or matchups based on player/team data.                                                                          |
| **Chat**                 | Ask questions and get AI-generated insights via LangChain agent.                                                                     |
---

### **Dependencies & Purpose**

| Library                 | Purpose / Usage in Project                                                                                   |
| ----------------------- | ------------------------------------------------------------------------------------------------------------ |
| **streamlit**           | Powers the interactive web app, including dashboards, tabs, charts, and UI components.                       |
| **beautifulsoup4**      | Enables web scraping of additional NBA or media data not available via official APIs.                        |
| **pandas**              | Core data manipulation library for cleaning, filtering, and structuring player, team, and game data.         |
| **nba_api**             | Provides access to official NBA statistics for real-time and historical data.                                |
| **plotly**              | Creates interactive visualizations such as performance trends and matchup dashboards.                        |
| **matplotlib**          | Supports static or advanced plotting for detailed analytics visualizations.                                  |
| **langchain**           | Framework for AI reasoning workflows; connects NBA data to AI agents for generating insights.                |
| **langchain-openai**    | Integrates OpenAI models for natural language queries, powering the Chat tab.                                |
| **langchain-community** | Adds community-supported tools and connectors to enhance AI workflows and data integration within LangChain. |
---

### Getting Started

1. **Clone the repository**

```bash
git clone https://github.com/petermartens98/NBA-Analytics-Streamlit-App-with-LangChain-Agent
cd NBA-Analytics-Streamlit-App-with-LangChain-Agent
```

2. **Create a virtual environment**

```bash
python -m venv venv
```

3. **Activate the virtual environment**

* **Windows:**

```bash
venv\Scripts\activate
```

* **Mac/Linux:**

```bash
source venv/bin/activate
```

4. **Install dependencies**

```bash
pip install -r requirements.txt
```

5. **Run the app**

```bash
streamlit run streamlit_app.py
```

---

### Screenshots

#### Today's Matchups Tab
<img width="1500" height="1237" alt="image" src="https://github.com/user-attachments/assets/8dd3b0eb-401b-42f7-a469-d57ee0f7a3ae" />

#### Injuries Tab
<img width="1531" height="1207" alt="image" src="https://github.com/user-attachments/assets/54775ee9-fe69-4f21-9861-adf343f33046" />

#### Players Tab
<img width="1511" height="899" alt="image" src="https://github.com/user-attachments/assets/087bb48d-a6ce-4b97-891f-16ad3a570aee" />
<img width="1523" height="541" alt="image" src="https://github.com/user-attachments/assets/08cf519b-50b3-44af-a1c1-c08632d2d523" />
<img width="1518" height="978" alt="image" src="https://github.com/user-attachments/assets/72130373-c0a2-43ce-ba14-a3d152a11324" />
<img width="1559" height="1267" alt="image" src="https://github.com/user-attachments/assets/4849fc59-d4e0-41a3-9d3e-a3ab3f7f8a6e" />
<img width="1537" height="886" alt="image" src="https://github.com/user-attachments/assets/44ba6854-03e7-4f45-9f5e-b8ec317983cc" />
<img width="1548" height="789" alt="image" src="https://github.com/user-attachments/assets/ed409ebb-9df6-4787-8df7-ee45e8e625b3" />

#### Single Player Tab
<img width="1514" height="857" alt="image" src="https://github.com/user-attachments/assets/779bee6f-aac8-4c68-a9f7-7e91fa893cb8" />
<img width="1527" height="907" alt="image" src="https://github.com/user-attachments/assets/b12d4ccd-9f8e-474b-913b-c4ef55d83c12" />
<img width="1551" height="1267" alt="image" src="https://github.com/user-attachments/assets/143cc38c-55bb-4fda-8e95-2f9582c11664" />
<img width="1525" height="918" alt="image" src="https://github.com/user-attachments/assets/670dba74-87b7-48f7-a056-379b94626a8a" />
<img width="1519" height="715" alt="image" src="https://github.com/user-attachments/assets/09247af6-dcad-46ba-b9e9-9b32e6f76b1c" />
<img width="1520" height="1067" alt="image" src="https://github.com/user-attachments/assets/b55f1791-8a35-4af7-87fe-4fd7b3112dab" />
<img width="1543" height="1202" alt="image" src="https://github.com/user-attachments/assets/9b3d43d4-fdb0-495a-8f80-37a0535d27ce" />
<img width="1503" height="647" alt="image" src="https://github.com/user-attachments/assets/b422cbbd-8ac0-4953-a4b6-345d6a20bba1" />


#### Teams Tab

#### Single Team Tab

#### Simulation Tab

#### Chat Tab

---
