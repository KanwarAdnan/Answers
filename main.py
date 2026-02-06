from fastapi import FastAPI
from uuid import uuid4

from app.schemas import Document, QueryRequest
from app.openai_client import get_embedding, generate_answer
from app.vector_store import (
    add_embedding,
    search_embedding,
    list_documents,
    delete_document
)
from app.utils import chunk_text

app = FastAPI(title="Document Q&A API")

@app.post("/documents")
async def upload_documents(documents: list[Document]):
    for doc in documents:
        doc_id = str(uuid4())
        chunks = chunk_text(doc.content)

        for chunk in chunks:
            embedding = get_embedding(chunk)
            add_embedding(
                embedding,
                {
                    "doc_id": doc_id,
                    "title": doc.title,
                    "content": chunk
                }
            )
    return {"status": "Documents indexed successfully"}

@app.get("/documents")
async def get_documents():
    return list_documents()

@app.post("/query")
async def query_documents(request: QueryRequest):
    query_embedding = get_embedding(request.question)
    results = search_embedding(query_embedding)

    unique = {}
    for r in results:
        unique[r["content"]] = r

    clean_sources = list(unique.values())
    context = "\n".join(r["content"] for r in clean_sources)

    answer = generate_answer(context, request.question)

    return {
        "answer": answer,
        "sources": results
    }

@app.delete("/documents/{doc_id}")
async def remove_document(doc_id: str):
    delete_document(doc_id)
    return {"status": "Document deleted"}
