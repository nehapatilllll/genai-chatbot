import os, uuid, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from modules.backend_store import ingest_into_backend

KNOWLEDGE_DIR = os.path.join(os.path.dirname(__file__), "knowledge")
if not os.path.isdir(KNOWLEDGE_DIR):
    os.makedirs(KNOWLEDGE_DIR)
    print(f"✅ Place your backend documents in {KNOWLEDGE_DIR} then run this script again.")
    exit()

for fname in os.listdir(KNOWLEDGE_DIR):
    fpath = os.path.join(KNOWLEDGE_DIR, fname)
    if os.path.isfile(fpath):
        with open(fpath, "rb") as f:
            ingest_into_backend(f, fname)
print("Backend knowledge store ready!")