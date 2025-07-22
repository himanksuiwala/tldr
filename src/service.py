import logging
from typing import List

import ollama

from .connection import database
from .models import Embedding
from .settings import settings

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = settings.model_embedding


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


def get_vector_embeddings(text: str) -> List[float]:
    """
    Generate vector embeddings for the given text using Ollama.
    """
    try:
        logger.debug(f"Generating embeddings for text of length: {len(text)}")

        response = ollama.embed(model=EMBEDDING_MODEL, input=text)
        embedding = response["embeddings"][0]

        logger.debug(f"Generated embedding with dimension: {len(embedding)}")
        return embedding

    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        raise Exception(f"Embedding generation failed: {str(e)}")
