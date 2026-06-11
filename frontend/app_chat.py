import streamlit as st
import requests

st.set_page_config(
    page_title="Enterprise Credit Assistant",
    page_icon="🤖",
    layout="wide"
)

# Dark-mode ChatGPT theme injection with absolute white text overrides
st.markdown("""
    <style>
    /* Global Application Canvas Background */
    .stApp { 
        background-color: #343541; 
        color: #FFFFFF !important; 
    }
    
    /* Target all Streamlit chat markdown message text wrappers to be crisp white */
    .stChatMessage div, .stChatMessage p, .stMarkdown p {
        color: #10a37f !important;
    }
    
    /* Custom style for the session sidebar button */
    div.stButton > button:first-child {
        background-color: #10a37f !important;
        color: white !important;
        border-radius: 6px;
        border: none;
    }
    
    /* Document/Policy Citation card formatting with readable text */
    .citation-card {
        background-color: #444654;
        color: #FFFFFF !important;
        border-radius: 8px;
        padding: 12px;
        margin-top: 8px;
        border-left: 4px solid #10a37f;
    }
    .citation-card b {
        color: #444654 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 AI Chat Assistant Workspace")
st.caption("Chat with your Vector DB or RDBMS databases with session continuation tracking.")
st.markdown("---")

# Session-based context initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Presentation layer
if "agent_memory" not in st.session_state:
    st.session_state.agent_memory = []  # Retained layer

# Sidebar controls
st.sidebar.title("💳 Session Management")
if st.sidebar.button("🗑️ Clear Active Chat Thread"):
    st.session_state.chat_history = []
    st.session_state.agent_memory = []
    st.rerun()

# Render all previous chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sql" in message and message["sql"]:
            with st.expander("🛠️ View Executed Analytical SQL String"):
                st.code(message["sql"], language="sql")
        if "citations" in message and message["citations"] != "N/A" and message["citations"]:
            st.markdown(f"<div class='citation-card'><b>📋 Policy Citations:</b><br>{message['citations']}</div>", unsafe_allow_html=True)

# Process incoming conversation inputs
if user_prompt := st.chat_input("How can I help with your credit analysis tasks today?"):
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        with st.spinner("Processing through your internal databases..."):
            try:
                # FASTAPI BACKEND TARGET ENDPOINT 
                FASTAPI_QUERY_URL = "http://127.0.0.1:8000/api/v1/query"
                
                # Matches your QueryRequest Pydantic schema structure
                payload = {"query": user_prompt}
                
                # Fire the POST request to the backend API
                response = requests.post(FASTAPI_QUERY_URL, json=payload)
                
                if response.status_code == 200:
                    response_payload = response.json()
                    
                    # Map variables according to your AIResponse Pydantic Model fields
                    answer_text = response_payload.get("answer", "No processing response generated.")
                    executed_sql = response_payload.get("sql_query_executed", "")
                    citations = response_payload.get("policy_citations", "N/A")
                    
                    # Render response components directly to UI container frames
                    response_placeholder.markdown(answer_text)
                    if executed_sql:
                        with st.expander("🛠️ View Executed Analytical SQL String"):
                            st.code(executed_sql, language="sql")
                    if citations and citations != "N/A":
                        st.markdown(f"<div class='citation-card'><b>📋 Policy Citations:</b><br>{citations}</div>", unsafe_allow_html=True)
                    
                    # Append everything to the current presentation session sequence
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer_text,
                        "sql": executed_sql,
                        "citations": citations
                    })
                else:
                    response_placeholder.error(f"❌ Backend Error {response.status_code}: {response.text}")
                
            except requests.exceptions.ConnectionError:
                response_placeholder.error("❌ Connection Refused: Could not connect to the FastAPI server. Verify it's running on port 8000!")
            except Exception as error:
                response_placeholder.error(f"An execution variance occurred: `{error}`")