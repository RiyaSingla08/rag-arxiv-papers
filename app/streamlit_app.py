"""
A simple chat interface for the RAG system, built with Streamlit.

Run from the project root:
    streamlit run app/streamlit_app.py

Then open the URL it prints (usually http://localhost:8501) in your browser.
"""

import sys
from pathlib import Path

import streamlit as st

# Make sure src/ is importable regardless of the exact directory Streamlit
# runs the script from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generate import generate_answer  # noqa: E402

st.set_page_config(page_title="RAG over arXiv Papers", page_icon="📚", layout="wide")

# ------------------------------------------------------------------
# Sidebar: settings + project context
# ------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Number of sources to retrieve", min_value=1, max_value=10, value=5)
    use_reranking = st.checkbox("Use cross-encoder re-ranking", value=True)

    st.divider()
    st.markdown(
        """
        **About this project**

        A Retrieval-Augmented Generation system built end to end:
        arXiv paper ingestion → chunking → local embeddings → vector
        search (Chroma) → optional cross-encoder re-ranking → local LLM
        generation (Ollama, `llama3.2`).

        Answers are grounded **only** in the retrieved paper content —
        the model is instructed to say so explicitly if the context
        doesn't contain an answer, rather than guessing.

        See `eval/results.json` and the README for retrieval quality
        metrics (Hit Rate@k, MRR) measured against a hand-labeled test set.
        """
    )

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

# ------------------------------------------------------------------
# Main chat area
# ------------------------------------------------------------------
st.title("📚 RAG over arXiv Papers on Retrieval-Augmented Generation")
st.caption(
    "Ask a question about the 15 papers in the knowledge base. "
    "The system will only answer from what it actually retrieves."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay chat history on every rerun (Streamlit reruns the whole script
# top-to-bottom on each interaction, so this is how history persists visually).
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.markdown(f"- {source}")

user_question = st.chat_input("Ask a question about the papers...")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant chunks and generating answer..."):
            result = generate_answer(user_question, top_k=top_k, use_reranking=use_reranking)
        st.markdown(result["answer"])
        if result["sources"]:
            with st.expander("Sources"):
                for source in result["sources"]:
                    st.markdown(f"- {source}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })
