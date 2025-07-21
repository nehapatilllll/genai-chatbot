import streamlit as st
from modules.llm import chat_completion
from modules.backend_store import get_backend_vectorstore, get_backend_graph
from modules.user_docs import ingest_user_file
from modules.forms import solution_form
import os

st.set_page_config(page_title="Digital Solutions Chatbot", layout="wide")

# ---------- 1. Backend knowledge must exist ----------
if not os.path.isdir("chroma_backend"):
    st.error("Backend knowledge store not found. "
             "Please run `python backend/ingest_backend.py` first.")
    st.stop()

# ---------- 2. Sidebar ----------
with st.sidebar:
    st.title("📁 Ad-hoc Document")
    uploaded = st.file_uploader("Upload PDF / DOCX / TXT", type=["pdf", "docx", "txt"],
                                key="adhoc_uploader")
    if uploaded:
        with st.spinner("Processing..."):
            vs, full_text = ingest_user_file(uploaded, uploaded.name)
            st.session_state["adhoc_vectorstore"] = vs
            st.session_state["adhoc_text"] = full_text
            st.success("Ready for questions / summary.")

    if st.button("Reset chat"):
        st.session_state.messages = []

# ---------- 3. Chat Memory ----------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system",
         "content": "You are an expert assistant helping employees with digital solutions, technologies, industries and pain points."}
    ]

# ---------- 4. Chat UI ----------
st.title("💬 Digital Solutions Platform")
tab_main, tab_doc = st.tabs(["Main Knowledge Chat", "Uploaded Document Q&A"])

with tab_main:
    for msg in st.session_state.messages[1:]:
        st.chat_message(msg["role"]).write(msg["content"])

    prompt_main = st.chat_input("Ask anything...", key="main_prompt")
    if prompt_main:
        st.session_state.messages.append({"role": "user", "content": prompt_main})
        st.chat_message("user").write(prompt_main)

        # retrieve from backend
        vs_back = get_backend_vectorstore()
        docs = vs_back.similarity_search(prompt_main, k=4)
        context = "\n".join([d.page_content for d in docs])

        temp_msgs = st.session_state.messages + [
            {"role": "user",
             "content": f"User question: {prompt_main}\n\nRelevant snippets:\n{context}"}]
        with st.spinner("Thinking..."):
            answer = chat_completion(temp_msgs)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)

with tab_doc:
    st.subheader("📄 Questions about the uploaded file")
    if "adhoc_vectorstore" not in st.session_state:
        st.info("Please upload a document in the sidebar first.")
    else:
        prompt_doc = st.chat_input("Ask / Summarize / Analyze...", key="doc_prompt")
        if prompt_doc:
            st.chat_message("user").write(prompt_doc)

            vs = st.session_state["adhoc_vectorstore"]
            docs = vs.similarity_search(prompt_doc, k=4)
            context = "\n".join([d.page_content for d in docs])

            messages = [
                {"role": "system",
                 "content": "You are a helpful assistant. Use only the provided context to answer."},
                {"role": "user",
                 "content": f"Context:\n{context}\n\nQuestion: {prompt_doc}"}
            ]
            with st.spinner("Thinking..."):
                answer = chat_completion(messages)
            st.chat_message("assistant").write(answer)

        if st.button("Summarize uploaded document"):
            with st.spinner("Summarizing..."):
                messages = [
                    {"role": "system",
                     "content": "Summarize the following text in 3–5 bullet points."},
                    {"role": "user", "content": st.session_state["adhoc_text"]}
                ]
                summary = chat_completion(messages)
            st.markdown("**Summary:**")
            st.write(summary)

# ---------- 5. Submit Solution ----------
with st.expander("🚀 Submit New Solution / Enhancement"):
    solution_form()