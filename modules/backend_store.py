import os, uuid, networkx as nx
import streamlit as st   
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


CHROMA_BACKEND_DIR = "chroma_backend"
EMBEDDING = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_backend_vectorstore():
    os.makedirs(CHROMA_BACKEND_DIR, exist_ok=True)
    return Chroma(persist_directory=CHROMA_BACKEND_DIR, embedding_function=EMBEDDING)

@st.cache_resource
def get_backend_graph():
    if "backend_graph" not in st.session_state:
        st.session_state.backend_graph = nx.DiGraph()
    return st.session_state.backend_graph

def ingest_into_backend(file_obj, filename):
    """Called from backend/ingest_backend.py"""
    text = ""
    if filename.endswith(".pdf"):
        from pypdf2 import PdfReader
        text = "\n".join(p.extract_text() or "" for p in PdfReader(file_obj).pages)
    elif filename.endswith(".docx"):
        from docx import Document as DocxDocument
        doc = DocxDocument(file_obj)
        text = "\n".join(p.text for p in doc.paragraphs)
    else:  # TXT
        text = file_obj.read().decode()

    if not text.strip():
        return 0

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = [Document(page_content=chunk,
                     metadata={"source": filename,
                               "chunk_id": str(uuid.uuid4())})
            for chunk in splitter.split_text(text)]

    vs = get_backend_vectorstore()
    vs.add_documents(docs)
    

    G = get_backend_graph()
    file_node = f"file:{filename}"
    G.add_node(file_node, type="file", name=filename)
    for d in docs:
        chunk_node = d.metadata["chunk_id"]
        G.add_node(chunk_node, type="chunk", text=d.page_content[:100]+"...")
        G.add_edge(file_node, chunk_node)
    return len(docs)