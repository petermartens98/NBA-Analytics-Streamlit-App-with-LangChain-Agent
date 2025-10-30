import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.tools import tool
from langchain.agents import create_agent
import pandas as pd
from pathlib import Path

DATA_FILE = Path("latest_data/injuries.csv")
PLAYER_FILE = Path("latest_data/players.csv")
TEAM_FILE = Path("latest_data/games/team_stats.csv")

# ---------------- Tools ----------------
@tool
def get_injury_data_tool(query: str = "") -> str:
    """Get current NBA injury data. Optionally filter by team or player name."""
    if not DATA_FILE.exists():
        return "No injury data available."
    
    try:
        df = pd.read_csv(DATA_FILE)
        if query:
            filtered = df[
                df['Team'].str.contains(query, case=False, na=False) |
                df['Name'].str.contains(query, case=False, na=False)
            ]
            if not filtered.empty:
                return filtered.to_string(index=False)
            return f"No injury data found for '{query}'"
        return df.head(20).to_string(index=False)
    except Exception as e:
        return f"Error reading injury data: {str(e)}"

@tool
def get_player_stats_tool(query: str = "") -> str:
    """Get NBA player statistics. Optionally filter by player name."""
    if not PLAYER_FILE.exists():
        return "No player data available."
    
    try:
        df = pd.read_csv(PLAYER_FILE)
        player_col = next((c for c in df.columns if 'player_name' in c.lower()), df.columns[0])
        
        if query:
            filtered = df[df[player_col].str.contains(query, case=False, na=False)]
            if not filtered.empty:
                return filtered.to_string(index=False)
            return f"No player stats found for '{query}'"
        return df.head(20).to_string(index=False)
    except Exception as e:
        return f"Error reading player data: {str(e)}"

@tool
def get_team_stats_tool(query: str = "") -> str:
    """Get NBA team statistics. Optionally filter by team name."""
    if not TEAM_FILE.exists():
        return "No team data available."
    
    try:
        df = pd.read_csv(TEAM_FILE)
        if query:
            filtered = df[df['Team'].str.contains(query, case=False, na=False)]
            if not filtered.empty:
                return filtered.to_string(index=False)
            return f"No team stats found for '{query}'"
        return df.to_string(index=False)
    except Exception as e:
        return f"Error reading team data: {str(e)}"

# ---------------- Agent ----------------
@st.cache_resource
def create_agent_with_tools(api_key: str):
    """Create and cache the agent to avoid recreation on every interaction."""
    llm = ChatOpenAI(
        model="deepseek-chat",
        openai_api_key=api_key,
        openai_api_base="https://api.deepseek.com",
        temperature=0.7
    )
    tools = [get_injury_data_tool, get_player_stats_tool, get_team_stats_tool]
    agent = create_agent(llm, tools=tools)
    return agent

# ---------------- Streamlit UI ----------------
def initialize_session_state():
    """Initialize all session state variables."""
    if "deepseek_api_key" not in st.session_state:
        st.session_state.deepseek_api_key = ""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "agent_initialized" not in st.session_state:
        st.session_state.agent_initialized = False

def render_chat_tab():
    st.markdown("### 💬 NBA Analytics Chat Assistant")
    st.markdown("Ask questions about NBA injuries, player stats, and team performance!")

    # Initialize session state
    initialize_session_state()

    # API Key input section
    with st.expander("⚙️ API Configuration", expanded=not st.session_state.deepseek_api_key):
        api_key_input = st.text_input(
            "DeepSeek API Key",
            type="password",
            value=st.session_state.deepseek_api_key,
            help="Enter your DeepSeek API key. Get one at https://platform.deepseek.com",
            key="api_key_input"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("💾 Save", use_container_width=True):
                if api_key_input and api_key_input.strip():
                    st.session_state.deepseek_api_key = api_key_input.strip()
                    st.session_state.agent_initialized = False  # Reset agent
                    st.success("✅ API key saved!")
                    st.rerun()
                else:
                    st.error("❌ Please enter a valid API key")
        
        with col2:
            if st.session_state.deepseek_api_key and st.button("🗑️ Clear Key", use_container_width=True):
                st.session_state.deepseek_api_key = ""
                st.session_state.agent_initialized = False
                st.session_state.chat_history = []
                st.rerun()

    # Check if API key is set
    if not st.session_state.deepseek_api_key:
        st.info("👆 Please enter your DeepSeek API key in the configuration section above to start chatting")
        return

    # Initialize agent (only once, cached)
    if not st.session_state.agent_initialized:
        try:
            with st.spinner("Initializing AI agent..."):
                st.session_state.agent = create_agent_with_tools(st.session_state.deepseek_api_key)
                st.session_state.system_prompt = SystemMessage(
                    content="You are a helpful NBA analytics assistant with access to real-time injury reports, "
                           "player statistics, and team performance data. Provide concise, accurate, and insightful "
                           "responses. When using tools, interpret the data clearly for the user."
                )
                st.session_state.agent_initialized = True
        except Exception as e:
            st.error(f"❌ Failed to initialize agent: {str(e)}")
            st.info("💡 Please check your API key and try again")
            return

    # Display chat history
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.write(message.content)

    # Chat input
    if prompt := st.chat_input("Ask about NBA injuries, players, or teams..."):
        # Add user message
        st.session_state.chat_history.append(HumanMessage(content=prompt))

        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    messages = [st.session_state.system_prompt] + st.session_state.chat_history
                    response = st.session_state.agent.invoke({"messages": messages})
                    
                    # Extract assistant's response
                    last_msg = response["messages"][-1]
                    assistant_response = last_msg.content
                    
                    # Save response
                    st.session_state.chat_history.append(AIMessage(content=assistant_response))
                    st.rerun()
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append(AIMessage(content=error_msg))

    # Action buttons
    if st.session_state.chat_history:
        st.divider()
        col1, col2, col3 = st.columns([5, 1, 1])
        with col3:
            if st.button("🔄 Clear", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

if __name__ == "__main__":
    # Set page config
    st.set_page_config(
        page_title="NBA Analytics Chat",
        page_icon="🏀",
        layout="wide"
    )
    render_chat_tab()