import uuid
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig


from main import (
    chatbot,
    register_user,
    login_user,
    upsert_thread,
    retrieve_all_threads,
    thread_document_metadata,
    retrieve_user_documents,
    set_thread_title,
    get_thread_title,
    ingest_pdf,
    summarize_thread,
)

# ==========================
# Session defaults
# ==========================
def ensure_session_defaults():
    if "user" not in st.session_state:
        st.session_state["user"] = None
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = None
    if "thread_id" not in st.session_state:
        st.session_state["thread_id"] = None
    if "message_history" not in st.session_state:
        st.session_state["message_history"] = []
    if "chat_threads" not in st.session_state:
        st.session_state["chat_threads"] = []
    if "thread_titles" not in st.session_state:
        st.session_state["thread_titles"] = {}
    if "ingested_docs" not in st.session_state:
        st.session_state["ingested_docs"] = {}

ensure_session_defaults()
selected_thread = None
docs = None

def generate_thread_id() -> str:
    return str(uuid.uuid4())

def get_config():
    return RunnableConfig(
        configurable={
            "thread_id": st.session_state["thread_id"],
            "user_id": st.session_state["user_id"],
        },
        metadata={
            "thread_id": st.session_state["thread_id"],
            "user_id": st.session_state["user_id"],
        },
        run_name="chat_turn",
    )

def load_conversation(thread_id: str, user_id: int):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id, "user_id": user_id}})
    return state.values.get("messages", [])

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    upsert_thread(thread_id, st.session_state["user_id"])
    st.session_state["message_history"] = []
    st.session_state["thread_titles"][thread_id] = "New Chat"
    set_thread_title(thread_id, "New Chat", st.session_state["user_id"])

def load_user_threads():
    if st.session_state["user_id"] is not None:
        st.session_state["chat_threads"] = retrieve_all_threads(st.session_state["user_id"])

def get_title_for_thread(thread_id: str) -> str:
    user_id = st.session_state["user_id"]
    local = st.session_state["thread_titles"].get(thread_id)
    if local:
        return local
    title = get_thread_title(thread_id, user_id)
    if title:
        st.session_state["thread_titles"][thread_id] = title
        return title
    return "New Chat"

# ==========================
# Auth UI
# ==========================
if st.session_state["user"] is None or st.session_state["user_id"] is None:
    st.title("🔐 Login / Register")

    tab_login, tab_register = st.tabs(["Login", "Register"])
    with tab_register:
        r_user = st.text_input("Username (register)", key="register_username")
        r_pass = st.text_input("Password", type="password", key="register_password")
        if st.button("Register"):
            msg = register_user(r_user.strip(), r_pass)
            if "success" in msg.lower():
                st.success(msg)
            else:
                st.error(msg)

    with tab_login:
        
        l_user = st.text_input("Username (login)", key="login_username")
        l_pass = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login"):
            user_id = login_user(l_user.strip(), l_pass)
            if user_id:
                st.session_state["user"] = l_user.strip()
                st.session_state["user_id"] = user_id
                # Initialize first thread if none
                load_user_threads()
                if not st.session_state["chat_threads"]:
                    reset_chat()
                else:
                    # pick most recent thread
                    st.session_state["thread_id"] = st.session_state["chat_threads"][0]
                st.rerun()
            else:
                st.error("Invalid credentials")
    st.stop()

# ==========================
# Logged-in UI
# ==========================
load_user_threads()
if st.session_state["thread_id"] is None:
    reset_chat()

thread_key = st.session_state["thread_id"]
user_id = st.session_state["user_id"]

st.sidebar.title("MohuGPT")
st.sidebar.success(f"Logged in as {st.session_state['user']}")
if st.sidebar.button("Logout", use_container_width=True):
    # Clear session
    for k in ["user", "user_id", "thread_id", "message_history", "chat_threads", "thread_titles", "ingested_docs"]:
        st.session_state[k] = None if k in ["user", "user_id", "thread_id"] else []
    ensure_session_defaults()
    st.rerun()

# Current thread header with title
current_title = get_title_for_thread(thread_key)
st.sidebar.markdown(f"**Current chat:** {current_title}")
st.sidebar.markdown(f"Thread ID: `{thread_key}`")

if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

# PDF status for current thread
doc_meta = thread_document_metadata(thread_key, user_id)
if doc_meta:
    st.sidebar.success(
        f"Using `{doc_meta.get('filename')}` "
        f"(chunks: {doc_meta.get('chunks')}, pages: {doc_meta.get('documents')})"
    )
else:
    st.sidebar.info("No PDF indexed yet for this chat.")

# Upload PDF
uploaded_pdf = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_pdf:
    summary = ingest_pdf(
        uploaded_pdf.getvalue(),
        thread_id=thread_key,
        user_id=user_id,
        filename=uploaded_pdf.name,
    )
    st.session_state["ingested_docs"].setdefault(thread_key, {})[uploaded_pdf.name] = summary
    st.sidebar.success(f"Ingested `{uploaded_pdf.name}`")

# Past conversations (user-specific)
st.sidebar.subheader("Past conversations")
threads_desc = [(tid, get_title_for_thread(tid)) for tid in st.session_state["chat_threads"]]

if not threads_desc:
    st.sidebar.write("No past conversations yet.")
else:
    for tid, title in (threads_desc):
        btn_label = f"{title}"
        if st.sidebar.button(btn_label, key=f"side-thread-{tid}"):
            selected_thread = tid

# My Documents (all PDFs for this user)
st.sidebar.subheader("📄 My Documents")
docs = retrieve_user_documents(user_id)
if not docs:
    st.sidebar.write("No documents uploaded yet.")
else:
    for i, doc in enumerate(docs[:50]):
        doc_label = f"{doc['filename']} - Thread: {doc['thread_id'][:8]}"
        if st.sidebar.button(doc_label, key=f"doc-{i}-{doc['thread_id']}"):
            st.session_state["thread_id"] = doc["thread_id"]
            st.rerun()

# ==========================
# Main chat area
# ==========================
st.title("Multi Utility Chatbot")

# Render message history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask about your document or use tools")

if user_input:
    # Append & render user message
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    config = get_config()
    with st.chat_message("assistant"):
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, _ in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="messages",
            ):
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    # Append assistant message
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )

    # Generate & persist thread title (first exchange)
    title = summarize_thread(thread_key, user_id)
    st.session_state["thread_titles"][thread_key] = title

    # Show document meta again
    doc_meta = thread_document_metadata(thread_key, user_id)
    if doc_meta:
        st.caption(
            f"Document indexed: {doc_meta.get('filename')} "
            f"(chunks: {doc_meta.get('chunks')}, pages: {doc_meta.get('documents')})"
        )

st.divider()

# Load a selected thread from sidebar
if selected_thread:
    st.session_state["thread_id"] = selected_thread
    messages = load_conversation(selected_thread, st.session_state["user_id"])

    temp_messages = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        else:
            # Ignore ToolMessage in display; optionally render tool feedback separately
            continue
        temp_messages.append({"role": role, "content": msg.content})

    st.session_state["message_history"] = temp_messages
    st.session_state["ingested_docs"].setdefault(selected_thread, {})
    st.rerun()
