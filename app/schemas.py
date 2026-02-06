from pydantic import BaseModel
from typing import List

class Document(BaseModel):
    title: str
    content: str

class QueryRequest(BaseModel):
    question: str
