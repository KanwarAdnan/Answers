from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding

def generate_answer(context: str, question: str) -> str:
    prompt = f"""
Use the following context to answer the question.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{question}
"""
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

