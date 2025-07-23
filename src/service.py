import logging
from typing import List

import ollama

from .connection import database
from .models import Embedding
from .settings import settings

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = settings.model_embedding


def generate_stringified_embedding(embedding: list) -> str:
    return "[" + ",".join(map(str, embedding)) + "]"


async def store_embeddings(embedding: list, chunk: str):
    query = """
    INSERT INTO rag_demo (chunk, embedding) 
    VALUES ($1, $2)
    """
    vector_str = generate_stringified_embedding(embedding)
    async with database.pool.acquire() as connection:
        await connection.execute(query, chunk, vector_str)


async def get_embeddings(embedding: list, top_k: int = 5) -> list[Embedding]:
    vector_str = generate_stringified_embedding(embedding)

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


async def store_chunks_with_embeddings(chunks: List[str]):
    """
    Store multiple text chunks with their generated embeddings in the database.
    """
    try:
        logger.info(f"Processing {len(chunks)} chunks for storage")

        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue

            logger.debug(f"Processing chunk {i + 1}/{len(chunks)}")

            embedding = get_vector_embeddings(chunk)
            await store_embeddings(embedding, chunk)

        logger.info(f"Successfully stored {len(chunks)} chunks with embeddings")

    except Exception as e:
        logger.error(f"Failed to store chunks with embeddings: {e}")
        raise Exception(f"Chunk storage failed: {str(e)}")
