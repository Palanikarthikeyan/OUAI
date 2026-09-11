import os
import streamlit as st
from pathlib import Path

# pip install streamlit langchain langchain-community langchain-groq pypdf docx2txt

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

try:
    from langchain_community.document_loaders import (
        DirectoryLoader,
        TextLoader,
        PyPDFLoader,
        Docx2txtLoader,
    )
except ImportError:
    DirectoryLoader = None


# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="Directory Chat",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CSS - ChatGPT-like UI
# ============================================================
st.markdown(
    """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        .stApp {
            background: #ffffff;
        }

        section[data-testid="stSidebar"] {
            background: #f7f7f8;
            border-right: 1px solid #e5e5e5;
        }

        .chat-title {
            font-size: 30px;
            font-weight: 700;
            text-align: center;
            margin-top: 20px;
            margin-bottom: 8px;
        }

        .chat-subtitle {
            text-align: center;
            color: #6b7280;
            margin-bottom: 25px;
        }

        .welcome-box {
            max-width: 760px;
            margin: 80px auto 20px auto;
            text-align: center;
        }

        .welcome-box h1 {
            font-size: 34px;
            margin-bottom: 8px;
        }

        .welcome-box p {
            color: #6b7280;
            font-size: 16px;
        }

        div[data-testid="stChatMessage"] {
            max-width: 850px;
            margin-left: auto;
            margin-right: auto;
        }

        .status-box {
            padding: 10px 12px;
            border-radius: 8px;
            background: #ececec;
            font-size: 13px;
            margin-top: 10px;
        }

        .file-info {
            font-size: 13px;
            color: #555;
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

if "documents" not in st.session_state:
    st.session_state.documents = []

if "directory" not in st.session_state:
    st.session_state.directory = ""

if "llm" not in st.session_state:
    st.session_state.llm = None


# ============================================================
# HELPERS
# ============================================================
def get_api_key():
    """Read Groq API key from Streamlit secrets or environment."""
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        key = ""

    return key or os.getenv("GROQ_API_KEY", "")


def load_directory(directory_path):
    """
    Load TXT, PDF and DOCX files from a directory.
    Returns LangChain Documents.
    """
    path = Path(directory_path)

    if not path.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory_path}")

    if not path.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory_path}")

    documents = []

    # TXT files
    txt_files = list(path.rglob("*.txt"))
    for file in txt_files:
        try:
            loader = TextLoader(str(file), encoding="utf-8")
            documents.extend(loader.load())
        except Exception as e:
            st.warning(f"Could not load {file.name}: {e}")

    # PDF files
    pdf_files = list(path.rglob("*.pdf"))
    for file in pdf_files:
        try:
            loader = PyPDFLoader(str(file))
            documents.extend(loader.load())
        except Exception as e:
            st.warning(f"Could not load {file.name}: {e}")

    # DOCX files
    docx_files = list(path.rglob("*.docx"))
    for file in docx_files:
        try:
            loader = Docx2txtLoader(str(file))
            documents.extend(loader.load())
        except Exception as e:
            st.warning(f"Could not load {file.name}: {e}")

    return documents


def build_context(documents, question, max_chars=12000):
    """
    Simple directory-based context selection.

    This intentionally keeps the example simple:
    - no vector database
    - no embeddings
    - no RAG chain

    For a production RAG application, replace this with
    FAISS/Chroma + embeddings + similarity search.
    """
    if not documents:
        return ""

    # Very simple keyword-based selection.
    # It avoids sending the entire directory to the LLM.
    words = {
        w.lower().strip(".,!?;:()[]{}")
        for w in question.split()
        if len(w) >= 3
    }

    scored = []

    for doc in documents:
        text = doc.page_content or ""
        lower_text = text.lower()

        score = sum(1 for word in words if word in lower_text)
        scored.append((score, text, doc.metadata))

    scored.sort(key=lambda x: x[0], reverse=True)

    context_parts = []
    total = 0

    for score, text, metadata in scored:
        if score == 0 and context_parts:
            continue

        source = metadata.get("source", "unknown")
        page = metadata.get("page")

        if page is not None:
            source_info = f"{source}, page {page + 1}"
        else:
            source_info = source

        chunk = f"\nSOURCE: {source_info}\n{text}\n"

        if total + len(chunk) > max_chars:
            break

        context_parts.append(chunk)
        total += len(chunk)

    return "\n".join(context_parts)


def create_llm():
    api_key = get_api_key()

    if not api_key:
        return None

    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        max_tokens=700,
        reasoning_effort="low",
        groq_api_key=api_key,
    )


def answer_question(question):
    llm = st.session_state.llm

    if llm is None:
        raise ValueError(
            "Groq API key not configured. Add GROQ_API_KEY to "
            ".streamlit/secrets.toml or set it as an environment variable."
        )

    context = build_context(
        st.session_state.documents,
        question,
        max_chars=12000,
    )

    if not context:
        context = "No relevant directory content was found."

    system_prompt = """
You are a helpful document assistant.

Answer the user's question using the supplied directory content.

Rules:
1. Use the supplied context when it contains the answer.
2. Do not invent facts that are not supported by the context.
3. If the answer cannot be found, clearly say:
   "I couldn't find that information in the loaded documents."
4. Keep the answer concise but useful.
5. Mention the source file when possible.
"""

    user_prompt = f"""
DIRECTORY CONTEXT:
{context}

USER QUESTION:
{question}
"""

    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )

    return response.content


def reset_chat():
    st.session_state.messages = []


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("##  Directory Chat")
    st.caption("Chat with files from a local directory")

    st.divider()

    api_key_present = bool(get_api_key())

    if api_key_present:
        st.success("Groq API key detected")
    else:
        st.error("Groq API key missing")

    st.markdown("###  Directory")

    directory = st.text_input(
        "Directory path",
        value=st.session_state.directory,
        placeholder=r"C:\my_documents",
        help="Enter the directory containing TXT, PDF or DOCX files.",
    )

    col1, col2 = st.columns(2)

    with col1:
        load_clicked = st.button(
            " Load",
            use_container_width=True,
            type="primary",
        )

    with col2:
        clear_clicked = st.button(
            " Clear",
            use_container_width=True,
        )

    if load_clicked:
        if not directory.strip():
            st.warning("Enter a directory path.")
        else:
            with st.spinner("Loading documents..."):
                try:
                    docs = load_directory(directory.strip())

                    st.session_state.directory = directory.strip()
                    st.session_state.documents = docs
                    st.session_state.messages = []
                    st.session_state.llm = create_llm()

                    st.success(f"Loaded {len(docs)} document pages/chunks.")

                except Exception as e:
                    st.error(str(e))

    if clear_clicked:
        st.session_state.directory = ""
        st.session_state.documents = []
        st.session_state.messages = []
        st.session_state.llm = None
        st.rerun()

    st.divider()

    if st.session_state.documents:
        st.markdown("###  Loaded Content")
        st.write(f"**Documents/pages:** {len(st.session_state.documents)}")

        source_files = sorted(
            {
                str(doc.metadata.get("source", "Unknown"))
                for doc in st.session_state.documents
            }
        )

        st.write(f"**Files:** {len(source_files)}")

        with st.expander("View files"):
            for source in source_files:
                st.markdown(
                    f'<div class="file-info"> {source}</div>',
                    unsafe_allow_html=True,
                )

    st.divider()

    if st.button("➕ New Chat", use_container_width=True):
        reset_chat()
        st.rerun()

    st.caption("Supported: PDF • TXT • DOCX")


# ============================================================
# MAIN HEADER
# ============================================================
st.markdown(
    '<div class="chat-title">Directory Assistant</div>',
    unsafe_allow_html=True,
)

if st.session_state.directory:
    st.markdown(
        f'<div class="chat-subtitle"> {st.session_state.directory}</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="chat-subtitle">Load a directory and start asking questions</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# WELCOME SCREEN
# ============================================================
if not st.session_state.messages:
    st.markdown(
        """
        <div class="welcome-box">
            <h1>How can I help you?</h1>
            <p>
                Load a directory containing PDF, TXT or DOCX files,
                then ask questions about the documents.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.documents:
        st.info(
            " Enter your directory path in the sidebar and click **Load**."
        )


# ============================================================
# CHAT HISTORY
# ============================================================
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]

    with st.chat_message(role):
        st.markdown(content)


# ============================================================
# CHAT INPUT
# ============================================================
question = st.chat_input(
    "Message Directory Assistant...",
    disabled=not bool(st.session_state.documents),
)

if question:
    # User message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer = answer_question(question)
                st.markdown(answer)

            except Exception as e:
                answer = f" **Error:** {e}"
                st.error(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )
