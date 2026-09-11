import os
import pandas as pd
import streamlit as st

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama


# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------

EMBED_MODEL = "nomic-embed-text:latest"
LLM_MODEL = "gemma2:2b"

VECTOR_DB_PATH = "vectorstore"

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Excel RAG Assistant",
    layout="wide"
)

st.title("Excel RAG Assistant")
st.write("Ask questions from Excel files stored in folders/subfolders")


# ---------------------------------------------------
# LOAD EXCEL FILES
# ---------------------------------------------------

def load_excel_documents(folder_path):

    documents = []

    for root, dirs, files in os.walk(folder_path):

        for file in files:

            if file.lower().endswith((".xlsx", ".xls")):

                file_path = os.path.join(root, file)

                try:

                    df = pd.read_excel(file_path)

                    if len(df) == 0:
                        continue

                    # Row-level documents
                    for _, row in df.iterrows():

                        text = "\n".join(
                            [
                                f"{col}: {row[col]}"
                                for col in df.columns
                            ]
                        )

                        documents.append(
                            Document(
                                page_content=text,
                                metadata={
                                    "source": file_path
                                }
                            )
                        )

                except Exception as e:

                    st.error(f"Error reading {file_path}")
                    st.error(str(e))

    return documents


# ---------------------------------------------------
# CREATE VECTOR DB
# ---------------------------------------------------

def build_vector_store(folder_path):

    docs = load_excel_documents(folder_path)

    if len(docs) == 0:

        st.error(
            "No Excel records found. Check folder path."
        )

        return None

    embedding = OllamaEmbeddings(
        model=EMBED_MODEL
    )

    vector_db = FAISS.from_documents(
        docs,
        embedding
    )

    vector_db.save_local(
        VECTOR_DB_PATH
    )

    return len(docs)


# ---------------------------------------------------
# LOAD VECTOR DB
# ---------------------------------------------------

def load_vector_store():

    embedding = OllamaEmbeddings(
        model=EMBED_MODEL
    )

    db = FAISS.load_local(
        VECTOR_DB_PATH,
        embedding,
        allow_dangerous_deserialization=True
    )

    return db


# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.header("Settings")

folder_path = st.sidebar.text_input(
    "Excel Folder Path",
    r"C:\\Users\karth\RAG-Projcts\Excel-RAG\Data\customers"
)

if st.sidebar.button("Build Vector DB"):

    with st.spinner("Scanning Excel files..."):

        total_docs = build_vector_store(
            folder_path
        )

    if total_docs:

        st.sidebar.success(
            f"Indexed {total_docs} records"
        )

# ---------------------------------------------------
# QA SECTION
# ---------------------------------------------------

question = st.text_input(
    "Ask Question",
    placeholder="List all transaction above 5000 balance"
)

if st.button("Ask"):

    if not os.path.exists(VECTOR_DB_PATH):

        st.error(
            "Please build Vector DB first."
        )

    else:

        with st.spinner("Searching..."):

            db = load_vector_store()

            retriever = db.as_retriever(
                search_kwargs={"k": 5}
            )

            docs = retriever.invoke(
                question
            )

            context = "\n\n".join(
                [
                    doc.page_content
                    for doc in docs
                ]
            )

            llm = ChatOllama(
                model=LLM_MODEL,
                temperature=0
            )

            prompt = f"""
You are an Excel assistant.

Answer ONLY using the provided context.

If the answer is not found,
say "Information not found."

Context:
{context}

Question:
{question}

Answer:
"""

            response = llm.invoke(
                prompt
            )

            st.subheader("Answer")

            st.write(
                response.content
            )

            st.subheader(
                "Retrieved Records"
            )

            for doc in docs:

                with st.expander(
                    doc.metadata.get(
                        "source",
                        "Unknown"
                    )
                ):

                    st.text(
                        doc.page_content
                    )