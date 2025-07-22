from pydantic import BaseModel


class Embedding(BaseModel):
    chunk: str
    embedding: str


class Chat(BaseModel):
    query: str
