# ============================================================
# ReAct Agentic AI - LangGraph + Groq + DuckDuckGo + Streamlit
# Classic Chat UI - Single Python File
# ============================================================
#
# Install:
# pip install -U streamlit langgraph langchain langchain-groq langchain-community duckduckgo-search
#
# Configure Groq:
#
# Option 1 - Streamlit secrets:
# Create .streamlit/secrets.toml
# GROQ_API_KEY = "your_groq_api_key"
#
# Option 2 - Environment variable:
# Windows:
#   set GROQ_API_KEY=your_groq_api_key
#
# Linux/Mac:
#   export GROQ_API_KEY="your_groq_api_key"
#
# Run:
# streamlit run react_langgraph_streamlit.py
#
# ============================================================

import os
import streamlit as st

from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage,
)

from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="ReAct Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CLASSIC UI CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main application */
    .stApp {
        background-color: #ffffff;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #f5f5f5;
        border-right: 1px solid #dddddd;
    }

    /* Header */
    .app-header {
        background-color: #343541;
        color: white;
        padding: 18px 25px;
        border-radius: 8px;
        margin-bottom: 20px;
    }

    .app-header h1 {
        margin: 0;
        font-size: 25px;
    }

    .app-header p {
        margin: 5px 0 0 0;
        color: #d1d5db;
        font-size: 14px;
    }

    /* Welcome */
    .welcome {
        text-align: center;
        margin-top: 100px;
        color: #444444;
    }

    .welcome h2 {
        font-size: 30px;
    }

    .welcome p {
        color: #777777;
    }

    /* Tool execution */
    .tool-box {
        background-color: #f7f7f8;
        border: 1px solid #dddddd;
        border-radius: 6px;
        padding: 10px;
        margin: 8px 0;
        font-size: 13px;
    }

    /* Status */
    .status {
        padding: 8px 12px;
        border-radius: 5px;
        background-color: #eeeeee;
        font-size: 13px;
    }

    /* Chat width */
    div[data-testid="stChatMessage"] {
        max-width: 900px;
        margin-left: auto;
        margin-right: auto;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "execution_trace" not in st.session_state:
    st.session_state.execution_trace = []


# ============================================================
# GROQ API KEY
# ============================================================

def get_groq_key():

    try:
        secret_key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        secret_key = ""

    return secret_key or os.getenv("GROQ_API_KEY", "")


# ============================================================
# CREATE AGENT
# ============================================================

@st.cache_resource
def create_agent():

    api_key = get_groq_key()

    if not api_key:
        return None

    # --------------------------------------------------------
    # Groq LLM
    # --------------------------------------------------------

    llm = ChatGroq(
    model="qwen/qwen3.6-27b",
    temperature=0,
    max_tokens=700,
    groq_api_key=api_key,
)

    # --------------------------------------------------------
    # DuckDuckGo Search Tool
    # --------------------------------------------------------

    search_tool = DuckDuckGoSearchRun()

    tools = [search_tool]

    # --------------------------------------------------------
    # Bind tools to LLM
    # --------------------------------------------------------

    llm_with_tools = llm.bind_tools(tools)

    # --------------------------------------------------------
    # Agent Node
    # --------------------------------------------------------

    def agent(state: MessagesState):

        system_message = SystemMessage(
            content="""
You are a helpful ReAct-style research assistant.

You have access to a DuckDuckGo web search tool.

Use the search tool when:
- The user asks about current information.
- The user asks about recent events.
- The user asks about a person, company, product, technology,
  news, price, release, or other information that may have changed.
- You need external information to answer accurately.



After receiving search results:
- Analyze the results.
- Answer the user's question clearly.
- Do not expose hidden chain-of-thought.
- Briefly mention that web search was used when appropriate.
"""
        )

        messages = state["messages"]

        response = llm_with_tools.invoke(
            [system_message] + messages
        )

        return {
            "messages": [response]
        }

    # --------------------------------------------------------
    # Tool Node
    # --------------------------------------------------------

    tool_node = ToolNode(tools)

    # --------------------------------------------------------
    # LangGraph
    # --------------------------------------------------------

    graph_builder = StateGraph(MessagesState)

    graph_builder.add_node(
        "agent",
        agent
    )

    graph_builder.add_node(
        "tools",
        tool_node
    )

    graph_builder.add_edge(
        START,
        "agent"
    )

    graph_builder.add_conditional_edges(
        "agent",
        tools_condition
    )

    graph_builder.add_edge(
        "tools",
        "agent"
    )

    return graph_builder.compile()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("##  ReAct Agent")

    st.caption(
        "LangGraph + Groq + DuckDuckGo"
    )

    st.divider()

    # API status
    if get_groq_key():
        st.success("Groq API key detected")
    else:
        st.error("Groq API key not configured")

    st.divider()

    st.markdown("### Agent Architecture")

    st.markdown(
        """
        **LLM**

        `openai/gpt-oss-20b`

        **Tool**

        `DuckDuckGo Search`

        **Framework**

        `LangGraph`

        **Pattern**

        `ReAct / Tool Calling`
        """
    )

    st.divider()

    st.markdown("### Agent Flow")

    st.code(
        """
User
 ↓
Agent / LLM
 ↓
Need tool?
 ├── No → Final Answer
 │
 └── Yes
       ↓
DuckDuckGo
       ↓
Tool Result
       ↓
Agent / LLM
       ↓
Final Answer
""",
        language="text",
    )

    st.divider()

    if st.button(
        "➕ New Chat",
        use_container_width=True,
    ):

        st.session_state.messages = []
        st.session_state.execution_trace = []

        st.rerun()

    st.caption(
        "Classic Streamlit Agent UI"
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="app-header">
        <h1> ReAct Agentic AI</h1>
        <p>
            LangGraph Agent + Groq LLM + DuckDuckGo Search Tool
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CREATE AGENT
# ============================================================

agent_graph = create_agent()


if agent_graph is None:

    st.warning(
        """
        **Groq API key is missing.**

        Add your key to:

        `.streamlit/secrets.toml`

        Example:

        `GROQ_API_KEY = "your-api-key"`

        Then restart Streamlit.
        """
    )

    st.stop()


# ============================================================
# WELCOME SCREEN
# ============================================================

if not st.session_state.messages:

    st.markdown(
        """
        <div class="welcome">

        <h2>How can I help you?</h2>

        <p>
        Ask a question. The agent will decide whether
        it needs DuckDuckGo web search.
        </p>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )


# ============================================================
# USER INPUT
# ============================================================

user_question = st.chat_input(
    "Ask the ReAct Agent..."
)


if user_question:

    # --------------------------------------------------------
    # Display user message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question,
        }
    )

    with st.chat_message("user"):

        st.markdown(user_question)

    # --------------------------------------------------------
    # Execute Agent
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        status_placeholder = st.empty()

        answer_placeholder = st.empty()

        try:

            status_placeholder.markdown(
                """
                <div class="status">
                Agent is analyzing the question...
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Convert UI messages to LangChain messages

            langchain_messages = []

            for msg in st.session_state.messages:

                if msg["role"] == "user":

                    langchain_messages.append(
                        HumanMessage(
                            content=msg["content"]
                        )
                    )

                elif msg["role"] == "assistant":

                    langchain_messages.append(
                        AIMessage(
                            content=msg["content"]
                        )
                    )

            # ------------------------------------------------
            # Run graph
            # ------------------------------------------------

            result = agent_graph.invoke(
                {
                    "messages": langchain_messages
                }
            )

            # ------------------------------------------------
            # Analyze execution
            # ------------------------------------------------

            final_answer = ""

            trace = []

            for msg in result["messages"]:

                # AI message
                if isinstance(msg, AIMessage):

                    if msg.tool_calls:

                        for tool_call in msg.tool_calls:

                            tool_name = tool_call.get(
                                "name",
                                "unknown"
                            )

                            tool_args = tool_call.get(
                                "args",
                                {}
                            )

                            trace.append(
                                f" Tool: {tool_name}\n"
                                f"Arguments: {tool_args}"
                            )

                    elif msg.content:

                        final_answer = msg.content

                # Tool response
                elif isinstance(msg, ToolMessage):

                    trace.append(
                        f" Tool Result Received\n"
                        f"{str(msg.content)[:500]}"
                    )

            # ------------------------------------------------
            # Display tool execution
            # ------------------------------------------------

            if trace:

                status_placeholder.markdown(
                    """
                    <div class="status">
                     Agent used DuckDuckGo Search
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                with st.expander(
                    " View Agent Tool Execution"
                ):

                    for item in trace:

                        st.markdown(
                            f"""
                            <div class="tool-box">
                            {item.replace(chr(10), "<br>")}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            else:

                status_placeholder.markdown(
                    """
                    <div class="status">
                    Agent answered without using a tool
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # ------------------------------------------------
            # Final answer
            # ------------------------------------------------

            answer_placeholder.markdown(
                final_answer
            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": final_answer,
                }
            )

            st.session_state.execution_trace = trace

        except Exception as e:

            status_placeholder.empty()

            st.error(
                f"Agent Error: {e}"
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div style="
        text-align:center;
        color:#999999;
        font-size:12px;
        margin-top:30px;
        padding:15px;
    ">
        ReAct Agent • LangGraph • Groq • DuckDuckGo
    </div>
    """,
    unsafe_allow_html=True,
)
