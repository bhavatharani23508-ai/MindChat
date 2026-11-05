
import streamlit as st
import uuid
import time

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="MindChat AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# NEW THEME: OBSIDIAN + NEON BLUE
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* GLOBAL */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #090d14 !important;
    color: #e8edf5 !important;
}

/* REMOVE DEFAULT HEADER */
header[data-testid="stHeader"] {
    background: transparent !important;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: #0d131d !important;
    border-right: 1px solid #1e2938 !important;
}

section[data-testid="stSidebar"] > div {
    background: #0d131d !important;
}

section[data-testid="stSidebar"] * {
    color: #dbe5f2;
}

/* BRAND */
.brand {
    font-size: 34px;
    font-weight: 800;
    letter-spacing: -1.5px;
    color: #38bdf8;
}

.brand-sub {
    color: #64748b !important;
    font-size: 10px;
    letter-spacing: 3px;
    margin-top: 3px;
}

/* HERO */
.hero {
    text-align: center;
    padding: 45px 0 20px;
}

.hero h1 {
    font-size: 48px;
    font-weight: 800;
    color: #f8fafc;
    letter-spacing: -2px;
}

.hero h1 span {
    color: #38bdf8;
}

.hero p {
    color: #8493a8;
    font-size: 15px;
}

/* CHAT MESSAGES */
div[data-testid="stChatMessage"] {
    background: #111a27 !important;
    border: 1px solid #233246 !important;
    border-radius: 16px !important;
    padding: 16px 18px !important;
    margin-bottom: 14px !important;
}

div[data-testid="stChatMessage"] p,
div[data-testid="stChatMessage"] li {
    color: #e2e8f0 !important;
    line-height: 1.75 !important;
}

/* FIX CHAT INPUT - ALL LAYERS */
[data-testid="stBottom"] {
    background: #090d14 !important;
}

[data-testid="stBottom"] > div {
    background: #090d14 !important;
}

[data-testid="stBottom"] [data-testid="stChatInput"] {
    background: #111a27 !important;
    border: 1px solid #334155 !important;
    border-radius: 16px !important;
    box-shadow: 0 0 0 1px rgba(56,189,248,0.08) !important;
}

/* INPUT WRAPPER */
[data-testid="stChatInput"] div,
[data-testid="stChatInput"] textarea {
    background-color: transparent !important;
}

/* INPUT TEXT */
[data-testid="stChatInput"] textarea {
    color: #f8fafc !important;
    -webkit-text-fill-color: #f8fafc !important;
    caret-color: #38bdf8 !important;
    font-size: 16px !important;
}

/* PLACEHOLDER */
[data-testid="stChatInput"] textarea::placeholder {
    color: #94a3b8 !important;
    -webkit-text-fill-color: #94a3b8 !important;
    opacity: 1 !important;
}

/* INPUT FOCUS */
[data-testid="stChatInput"]:focus-within {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 2px rgba(56,189,248,0.12) !important;
}

/* SEND BUTTON */
[data-testid="stChatInput"] button {
    background: #0284c7 !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
}

[data-testid="stChatInput"] button:hover {
    background: #0ea5e9 !important;
}

/* NORMAL BUTTONS */
.stButton > button,
.stDownloadButton > button {
    background: #111a27 !important;
    color: #dbeafe !important;
    border: 1px solid #26384d !important;
    border-radius: 11px !important;
    min-height: 42px;
    transition: 0.2s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: #172638 !important;
    border-color: #38bdf8 !important;
    color: #ffffff !important;
}

/* SELECT BOX */
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background: #111a27 !important;
    border-color: #26384d !important;
    color: #f8fafc !important;
}

/* METRICS */
div[data-testid="stMetric"] {
    background: #111a27 !important;
    border: 1px solid #26384d !important;
    border-radius: 13px !important;
    padding: 14px !important;
}

div[data-testid="stMetric"] label {
    color: #94a3b8 !important;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #38bdf8 !important;
}

/* STATUS CARD */
.status-card {
    background: #111a27;
    border: 1px solid #26384d;
    border-radius: 12px;
    padding: 14px;
    color: #e2e8f0;
}

.status-card small {
    color: #94a3b8;
}

/* THINKING */
div[data-testid="stSpinner"] p {
    color: #38bdf8 !important;
}

/* FOOTER */
.footer {
    text-align: center;
    color: #64748b;
    font-size: 12px;
    padding: 25px;
}

/* DIVIDER */
hr {
    border-color: #1e2938 !important;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "chat_store" not in st.session_state:
    st.session_state.chat_store = {}

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

if "model_name" not in st.session_state:
    st.session_state.model_name = "llama3.2"


# ============================================================
# OLLAMA MODEL
# ============================================================

@st.cache_resource
def load_model(model_name):
    return ChatOllama(
        model=model_name,
        temperature=0.7
    )


# ============================================================
# LANGCHAIN MEMORY
# ============================================================

def get_session_history(session_id):

    if session_id not in st.session_state.chat_store:
        st.session_state.chat_store[session_id] = (
            InMemoryChatMessageHistory()
        )

    return st.session_state.chat_store[session_id]


# ============================================================
# PROMPT
# ============================================================

prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are MindChat, an intelligent AI assistant.

        - Give accurate and helpful answers.
        - Remember details from the current conversation.
        - Use previous messages for context.
        - Explain difficult topics simply.
        - Format answers clearly.
        - Be friendly and professional.
        - Admit uncertainty when necessary.
        """
    ),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])


# ============================================================
# CHAT CHAIN
# ============================================================

@st.cache_resource
def create_chat_chain(model_name):

    llm = load_model(model_name)

    chain = prompt_template | llm

    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history"
    )


chat_chain = create_chat_chain(st.session_state.model_name)


# ============================================================
# HELPERS
# ============================================================

def new_conversation():
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.pending_prompt = None


def clear_conversation():
    st.session_state.chat_store[
        st.session_state.session_id
    ] = InMemoryChatMessageHistory()


def message_text(content):

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        result = ""

        for item in content:
            if isinstance(item, dict):
                result += item.get("text", "")
            else:
                result += str(item)

        return result

    return str(content)


def export_conversation(history):

    output = ["MindChat Conversation", "=" * 40, ""]

    for message in history.messages:

        speaker = (
            "You" if message.type == "human"
            else "MindChat"
        )

        output.append(
            f"{speaker}:\n{message_text(message.content)}\n"
        )

    return "\n".join(output)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="brand">MindChat</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="brand-sub">LOCAL AI ASSISTANT</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("### Workspace")

    if st.button("＋  New Conversation", use_container_width=True):
        new_conversation()
        st.rerun()

    if st.button("⌫  Clear Current Chat", use_container_width=True):
        clear_conversation()
        st.rerun()

    st.markdown("---")

    st.markdown("### AI Configuration")

    st.selectbox(
        "Local LLM",
        ["llama3.2"],
        key="model_name"
    )

    st.markdown("""
    <div class="status-card">
        🟢 <b>Local Inference</b><br>
        <small>AI responses generated on your computer</small>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    history = get_session_history(st.session_state.session_id)

    st.markdown("### Conversation Insights")

    st.metric("Total Messages", len(history.messages))

    st.metric(
        "Questions Asked",
        sum(m.type == "human" for m in history.messages)
    )

    st.markdown("---")

    st.markdown("### About MindChat")

    st.caption("Local AI powered by Ollama and LangChain.")

    st.caption("🔒 Private local inference")


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown("""
<div class="hero">
    <h1>Meet <span>MindChat</span></h1>
    <p>Your intelligent assistant for learning, thinking and creating.</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# WELCOME SUGGESTIONS
# ============================================================

history = get_session_history(st.session_state.session_id)

if not history.messages:

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📚 Learn Something", use_container_width=True):
            st.session_state.pending_prompt = (
                "Explain artificial intelligence simply."
            )
            st.rerun()

    with col2:
        if st.button("💻 Coding Assistant", use_container_width=True):
            st.session_state.pending_prompt = (
                "Teach me Python with an example."
            )
            st.rerun()

    with col3:
        if st.button("✍ Writing Assistant", use_container_width=True):
            st.session_state.pending_prompt = (
                "Help me write a professional introduction."
            )
            st.rerun()


# ============================================================
# CHAT HISTORY
# ============================================================

for message in history.messages:

    role = "user" if message.type == "human" else "assistant"

    avatar = "👤" if role == "user" else "🤖"

    with st.chat_message(role, avatar=avatar):
        st.markdown(message_text(message.content))


# ============================================================
# CHAT INPUT
# ============================================================

user_input = st.chat_input("Message MindChat...")

if st.session_state.pending_prompt:

    user_input = st.session_state.pending_prompt

    st.session_state.pending_prompt = None


# ============================================================
# STREAM RESPONSE
# ============================================================

if user_input:

    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="🤖"):

        placeholder = st.empty()

        full_response = ""

        start_time = time.time()

        try:

            with st.spinner("MindChat is thinking..."):

                stream = chat_chain.stream(
                    {"input": user_input},
                    config={
                        "configurable": {
                            "session_id": st.session_state.session_id
                        }
                    }
                )

                for chunk in stream:

                    full_response += message_text(chunk.content)

                    placeholder.markdown(
                        full_response + " ▌"
                    )

            placeholder.markdown(full_response)

            elapsed = time.time() - start_time

            st.caption(
                f"⚡ Generated in {elapsed:.2f} seconds"
            )

        except Exception as error:

            st.error("Could not connect to Ollama.")

            st.info(
                "Run 'ollama run llama3.2' in another terminal."
            )

            st.code(str(error))

            st.stop()

    st.rerun()


# ============================================================
# DOWNLOAD CHAT
# ============================================================

history = get_session_history(st.session_state.session_id)

if history.messages:

    st.markdown("---")

    st.download_button(
        "⬇ Download Conversation",
        data=export_conversation(history),
        file_name="MindChat_Conversation.txt",
        mime="text/plain"
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">
    MindChat AI &nbsp; • &nbsp; Ollama &nbsp; • &nbsp;
    LangChain Memory &nbsp; • &nbsp; Local Intelligence
</div>
""", unsafe_allow_html=True)
