from pydantic import BaseModel
from database.connection import database


class Embedding(BaseModel):
    chunk: str
    embedding: str


async def store_embeddings(embedded_value: list, chunk: str):
    query = """
    INSERT INTO rag_demo (chunk, embedding) 
    VALUES ($1, $2)
    """
    async with database.pool.acquire() as connection:
        await connection.execute(query, chunk, embedded_value)


async def get_embeddings(embedded_value: list, top_k: int = 5) -> list[Embedding]:
    # Convert embedding list to string format for PostgreSQL vector
    vector_str = "[" + ",".join(map(str, embedded_value)) + "]"

    sql_query = """
    SELECT chunk, embedding, 
           (embedding <=> $1::vector) as similarity
    FROM rag_demo 
    ORDER BY similarity 
    LIMIT $2;
    """
    async with database.pool.acquire() as connection:
        result = await connection.fetch(sql_query, vector_str, top_k)
        return [
            Embedding(
                chunk=row["chunk"],
                embedding=row["embedding"],  # Use vector string directly from DB
            )
            for row in result
        ]
