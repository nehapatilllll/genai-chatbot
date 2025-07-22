import os
import uuid
import json
import datetime
import streamlit as st
from openai import OpenAI
from langchain.schema import Document
from modules.backend_store import get_backend_vectorstore, get_backend_graph
from modules.user_docs import ingest_user_file
from modules.forms import solution_form
from modules.llm import chat_completion

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
BACKEND_INDEX_DIR = "faiss_backend"   # FAISS folder

# ------------------------------------------------------------
# STREAMLIT PAGE
# ------------------------------------------------------------
st.set_page_config(page_title="GenAI Solutions Chatbot", layout="wide")

# ------------------------------------------------------------
# SIDEBAR (ad-hoc uploads)
# ------------------------------------------------------------
with st.sidebar:
    st.title("📁 Ad-hoc Document")
    uploaded = st.file_uploader("Upload PDF / DOCX / TXT", type=["pdf", "docx", "txt"])
    if uploaded:
        with st.spinner("Processing..."):
            vs, full_text = ingest_user_file(uploaded, uploaded.name)
            st.session_state["adhoc_vectorstore"] = vs
            st.session_state["adhoc_text"] = full_text
            st.success("Ready for questions / summary.")
    if st.button("Reset chat"):
        st.session_state.messages = []

# ------------------------------------------------------------
# ENSURE BACKEND KNOWLEDGE EXISTS
# ------------------------------------------------------------
if not os.path.isdir(BACKEND_INDEX_DIR):
    st.error("Backend knowledge not found. Run `python backend/ingest_backend.py` locally first.")
    st.stop()

# ------------------------------------------------------------
# MAIN KNOWLEDGE CHAT  (answers ONLY from vector DB)
# ------------------------------------------------------------
st.title("💬 Digital Solutions Platform – Knowledge Chat")

# session memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# display history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# user prompt
prompt = st.chat_input("Ask anything…")
if prompt:
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 1. Retrieve from vector DB
    vs = get_backend_vectorstore()
    docs = vs.similarity_search(prompt, k=4)
    context = "\n".join([d.page_content for d in docs])

    # 2. Build prompt that **forces** the LLM to use only the context
    system_msg = (
        "You are a helpful assistant. "
        "Answer **only** using the snippets provided below. "
        "If the snippet does not contain the answer, say: 'No relevant information found.'"
    )
    user_msg = f"Context:\n{context}\n\nQuestion: {prompt}"

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]

    with st.spinner("Searching knowledge base…"):
        answer = chat_completion(messages)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)

# ------------------------------------------------------------
# UPLOADED-DOC TAB (unchanged)
# ------------------------------------------------------------
with st.expander("📄 Uploaded Document Q&A"):
    if "adhoc_vectorstore" not in st.session_state:
        st.info("Please upload a document in the sidebar first.")
    else:
        prompt_doc = st.chat_input("Ask / Summarize / Analyze…", key="doc_prompt")
        if prompt_doc:
            st.chat_message("user").write(prompt_doc)
            vs = st.session_state["adhoc_vectorstore"]
            docs = vs.similarity_search(prompt_doc, k=4)
            context = "\n".join([d.page_content for d in docs])
            msgs = [
                {"role": "system", "content": "Use only the provided context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {prompt_doc}"}
            ]
            with st.spinner("Thinking…"):
                answer = chat_completion(msgs)
            st.chat_message("assistant").write(answer)
        if st.button("Summarize uploaded document"):
            with st.spinner("Summarizing…"):
                msgs = [
                    {"role": "system", "content": "Summarize the following text in 3–5 bullet points."},
                    {"role": "user", "content": st.session_state["adhoc_text"]}
                ]
                summary = chat_completion(msgs)
            st.markdown("**Summary:**")
            st.write(summary)

# ------------------------------------------------------------
# SUBMIT SOLUTION FORM
# ------------------------------------------------------------
with st.expander("🚀 Submit New Solution / Enhancement"):
    solution_form()