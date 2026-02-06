# Document-Based Question Answering API

This project implements a semantic document Q&A system using FastAPI,
FAISS vector search, and OpenAI LLMs.

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=your_api_key
uvicorn app.main:app --reload
