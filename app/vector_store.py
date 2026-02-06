import faiss
import numpy as np
import json
import os
from typing import List, Dict

DATA_DIR = "data"
INDEX_PATH = f"{DATA_DIR}/faiss.index"
META_PATH = f"{DATA_DIR}/metadata.json"
DIMENSION = 1536

os.makedirs(DATA_DIR, exist_ok=True)

index = faiss.IndexFlatL2(DIMENSION)

metadata: List[Dict] = []

if os.path.exists(INDEX_PATH):
    index = faiss.read_index(INDEX_PATH)

if os.path.exists(META_PATH):
    with open(META_PATH, "r") as f:
        metadata = json.load(f)

def save_state():
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

def add_embedding(embedding: list, meta: dict):
    index.add(np.array([embedding], dtype="float32"))
    metadata.append(meta)
    save_state()

def search_embedding(embedding: list, k: int = 3):
    distances, indices = index.search(
        np.array([embedding], dtype="float32"), k
    )
    return [metadata[i] for i in indices[0] if i < len(metadata)]

def list_documents():
    seen = {}
    for m in metadata:
        seen[m["doc_id"]] = m["title"]
    return [{"doc_id": k, "title": v} for k, v in seen.items()]

def delete_document(doc_id: str):
    global metadata
    metadata = [m for m in metadata if m["doc_id"] != doc_id]
    save_state()
