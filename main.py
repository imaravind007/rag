import os, json, requests
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from streamlit_lottie import st_lottie

try:
    from pinecone import Pinecone
    from langchain_community.chat_models import ChatOpenAI
    from langchain.chains import ConversationChain
    from langchain.chains.conversation.memory import ConversationBufferWindowMemory
    from langchain.prompts import (
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate,
        ChatPromptTemplate,
        MessagesPlaceholder,
    )
    from langchain_openai import OpenAIEmbeddings
    LANGCHAIN_OK = True
except ImportError:
    LANGCHAIN_OK = False

# local helpers (stored in bot.py)
from bot import (
    find_match,
    load_docs,
    query_refiner,
    get_conversation_string,
    get_project_names,
)
# --------------------------------------------------------------------------
# 1 · App‑wide config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Renewable Dataroom Copilot",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# 2 · Runtime settings & API‑keys panel (sidebar)
# --------------------------------------------------------------------------
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=False), override=False)

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "OPENAI_API_KEY" not in st.session_state:
    st.session_state.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if "PINE_API_KEY" not in st.session_state:
    st.session_state.PINE_API_KEY = os.getenv("PINE_API_KEY", "")

with st.sidebar:
    st.divider()
    st.markdown("### 🔑 API Keys")
    st.session_state.OPENAI_API_KEY = st.text_input(
        "OpenAI API Key", value=st.session_state.OPENAI_API_KEY, type="password"
    )
    st.session_state.PINE_API_KEY = st.text_input(
        "Pinecone API Key", value=st.session_state.PINE_API_KEY, type="password"
    )
    if st.button("Save keys / rerun"):
        st.experimental_rerun()

openai_key = st.session_state.OPENAI_API_KEY.strip()
pine_key   = st.session_state.PINE_API_KEY.strip()

# --------------------------------------------------------------------------
# 3 · Global CSS (dark‑mode, bubbles, FAB)
# --------------------------------------------------------------------------
primary = "#14E39C"
bg_dark, bg_dark_2, text_dark = "#0E1117", "#161B22", "#E6EDF3"
css = f"""
<style>
body, .stApp {{
    {'background:'+bg_dark+'; color:'+text_dark+';' if st.session_state.dark_mode else ''}
}}
/* glass bubbles */
.st-chat-message .st-chat-message-content {{
    backdrop-filter:blur(10px);
    border-radius:14px; padding:1rem;
    border:1px solid rgba(255,255,255,0.15);
}}
.st-chat-message:nth-child(even) .st-chat-message-content {{
    background:rgba(20,227,156,0.08);
    border:1px solid rgba(20,227,156,0.25);
}}
/* floating action button */
#fab {{
  position:fixed; bottom:24px; right:28px; z-index:1000;
}}
#fab button {{
  height:52px;width:52px;border-radius:50%; background:{primary};
  border:none;color:{bg_dark}; font-size:28px;
  box-shadow:0 4px 12px rgba(20,227,156,0.45);
}}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# --------------------------------------------------------------------------
# 4 · Hero banner + Lottie
# --------------------------------------------------------------------------
def load_lottie(url: str):
    try:
        return requests.get(url).json()
    except Exception:
        return None

with st.container():
    st.markdown(
        f"""
        <div style="background:linear-gradient(90deg,{primary} 0%,#0fa47f 100%);
                    padding:2.5rem 1rem;border-radius:16px;margin-bottom:1rem;
                    text-align:center;color:{bg_dark}">
          <h1 style="font-size:2.4rem;margin:0">🌿 Renewable Dataroom Copilot</h1>
          <p style="font-size:1.25rem;margin:0.5rem 0 0">
            AI answers&nbsp;|&nbsp;Audit‑ready sources&nbsp;|&nbsp;Zero manual copy‑paste
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    lottie_json = load_lottie(
        "https://lottie.host/0cfa906e-d8b0-4b0e-9a67-56b7e7e82ca5/solar-panels.json"
    )
    if lottie_json:
        st_lottie(lottie_json, height=140, speed=0.5, key="solar")

# --------------------------------------------------------------------------
# 5 · Initialise or placeholder‑init models if keys absent
# --------------------------------------------------------------------------
if openai_key and pine_key and LANGCHAIN_OK:
    # Pinecone / LangChain objects
    pc = Pinecone(api_key=pine_key)
    index = pc.Index("energy-chunk")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large", openai_api_key=openai_key
    )
    llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=openai_key)

    prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                "Answer truthfully using the given context. "
                "If context doesn't have the answer, say “I don’t know”."
            ),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]
    )
    conversation = ConversationChain(
        memory=ConversationBufferWindowMemory(k=5, return_messages=True),
        prompt=prompt_template,
        llm=llm,
    )
    namespaces = get_project_names(index)
else:
    pc = index = embeddings = llm = conversation = None
    namespaces = []

# --------------------------------------------------------------------------
# 6 · Session state
# --------------------------------------------------------------------------
st.session_state.setdefault(
    "responses",
    ["Hi there 👋 — upload a document to begin."],
)
st.session_state.setdefault("requests", [])
st.session_state.setdefault("namespaces", namespaces)
st.session_state.setdefault("file_uploaded", False)

# --------------------------------------------------------------------------
# 7 · Floating‑action button anchor
# --------------------------------------------------------------------------
st.markdown('<a id="fab" href="#namespaces"><button>⚙️</button></a>', unsafe_allow_html=True)

# --------------------------------------------------------------------------
# 8 · Main UI (Upload & Chat tabs)
# --------------------------------------------------------------------------
tab_upload, tab_chat = st.tabs(["📄 Upload", "💬 Chat"])

# ---------------- 8.a Upload
with tab_upload:
    st.subheader("Upload your documents")
    if not (openai_key and pine_key and LANGCHAIN_OK):
        st.info("➡️ Enter your OpenAI & Pinecone keys in the sidebar to enable upload.")
    else:
        file_col, meta_col = st.columns([2, 1])
        with file_col:
            uploaded_files = st.file_uploader(
                "Supported: PDF · DOCX · TXT · XLS*",
                type=["pdf", "docx", "txt", "xlsx", "xlsm", "xlsb", "xls"],
                label_visibility="collapsed",
                accept_multiple_files=True
            )
        with meta_col:
            ns_input = st.text_input("Namespace", placeholder="e.g. Solar-Lease-Agreement.pdf")
            ns_clean = ns_input.strip().replace(" ", "-")

        if uploaded_files and st.button("Embed & index ↗", use_container_width=True):
            if not ns_clean:
                st.warning("Please supply a namespace.")
                st.stop()

            status = st.empty()  # timeline
            try:
                for uploaded_file in uploaded_files:

                    status.markdown(f"🔍 **Reading file** {uploaded_file.name} …")
                    content = uploaded_file.read()
                    
                    status.markdown("📑 **Chunking with Docling** …")
                    docs  = load_docs(
                            content,
                            os.path.splitext(uploaded_file.name)[1]
                            )               
                    texts = [d.page_content for d in docs]
                    
                    status.markdown("🧠 **Embedding** …")
                    vecs = embeddings.embed_documents(texts)
                    
                    status.markdown("🚀 **Upserting to Pinecone** …")
                    index.upsert(
                        vectors=[
                            {
                                "id": f"{uploaded_file.name}_{i}",
                                "values": vecs[i],
                                "metadata": {"text": texts[i], "title": uploaded_file.name},
                            }
                            for i in range(len(texts))
                        ],
                        namespace=ns_clean,
                    )

                st.session_state.namespaces = get_project_names(index)
                status.success("✅ All done — switch to **Chat** ➡️")

            except Exception as e:
                status.error(f"Upload failed: {e}")

# ---------------- 8.b Chat
with tab_chat:
    st.subheader("Ask questions about your documents")
    if not (openai_key and pine_key and LANGCHAIN_OK):
        st.info("➡️ Enter your API keys in the sidebar to enable chat.")
    else:
        if st.session_state.namespaces:
            selected_ns = st.selectbox("Namespace", st.session_state.namespaces)
        else:
            st.info("No namespaces yet — upload a document first.")
            selected_ns = None

        # history
        with st.chat_message("assistant"):
            st.markdown(st.session_state.responses[0])

        for i in range(1, len(st.session_state.responses)):
            with st.chat_message("user"):
                st.markdown(st.session_state.requests[i - 1])
            with st.chat_message("assistant"):
                st.markdown(st.session_state.responses[i])

        # new prompt
        user_prompt = st.chat_input("Ask something about the document…")
        if user_prompt and selected_ns:
            with st.chat_message("user"):
                st.markdown(user_prompt)

            with st.spinner("Thinking …"):
                try:
                    refined = query_refiner(get_conversation_string(), user_prompt)
                    context,sources = find_match(refined, selected_ns)
                    answer = conversation.predict(
                        input=f"Context:\n{context}\n\nQuery:\n{user_prompt}"
                    )

                    st.session_state.requests.append(user_prompt)
                    st.session_state.responses.append(answer)

                    with st.chat_message("assistant"):
                        st.markdown(answer)
                        if sources: 
                            with st.expander("🔗 Sources"):
                                st.markdown("\n".join(f"{i}. {src}" for i, src in enumerate(sources, 1)))
                except Exception as e:
                    st.error(f"Error: {e}")

# --------------------------------------------------------------------------
# 9 · Namespace manager
# --------------------------------------------------------------------------
st.markdown('<span id="namespaces"></span>', unsafe_allow_html=True)
with st.container():
    st.subheader("Namespace manager")
    if not (openai_key and pine_key and LANGCHAIN_OK):
        st.info("Enter keys to manage namespaces.")
    elif st.session_state.namespaces:
        chosen_ns = st.selectbox("Select namespace", st.session_state.namespaces, key="ns_mgr")
        col_del, col_refresh = st.columns([1, 1])
        if col_del.button("🗑️ Delete namespace"):
            pc.delete_namespace(index_name="energy-chunk", namespace=chosen_ns)
            st.session_state.namespaces = get_project_names(index)
            st.success("Deleted.")
        if col_refresh.button("🔄 Refresh list"):
            st.session_state.namespaces = get_project_names(index)
            st.success("Refreshed.")
    else:
        st.info("No namespaces yet.")
