import logging
from contextlib import asynccontextmanager

import ollama
from fastapi import FastAPI
from pydantic import BaseModel

from database.connection import database
from model.Embeddings import get_embeddings
from services.embedding import get_vector_embeddings
from settings import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            await database.connect()
            logger.info("Connected to the database successfully.")
            break
        except Exception as e:
            logger.error(
                f"Attempt {attempt + 1} to connect to the database failed: {e}"
            )
            if attempt == max_retries - 1:
                logger.error("Max retries reached. Unable to connect to the database.")
                raise
    yield
    await database.disconnect()
    logger.info("Disconnected from the database.")


class QueryModel(BaseModel):
    query: str


app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/api/upload")
async def upload(file):
    return {"message": "File uploaded successfully"}


@app.post("/api/query")
async def query(request: QueryModel):
    try:
        generated_embeddings = get_vector_embeddings(request.query)

        retrieved_knowledge = await get_embeddings(generated_embeddings)
        logger.debug(f"Retrieved {len(retrieved_knowledge)} chunks from database")

        context_chunks = [embedding.chunk for embedding in retrieved_knowledge]

        instruction_prompt = f"""You are a helpful chatbot.
        Use only the following pieces of context to answer the question. Don't make up any new information:
        {chr(10).join([f" - {chunk}" for chunk in context_chunks])}
        """

        response = ollama.chat(
            model=settings.model_language,
            messages=[
                {"role": "system", "content": instruction_prompt},
                {"role": "user", "content": request.query},
            ],
        )

        return {
            "query": request.query,
            "response": response["message"]["content"],
        }

    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        return {"error": f"Query processing failed: {str(e)}"}
