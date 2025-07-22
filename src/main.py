import asyncio
import logging
from contextlib import asynccontextmanager

import ollama
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from .connection import database
from .models import Chat
from .service import get_embeddings, get_vector_embeddings
from .settings import settings

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


app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/api/upload")
async def upload(file):
    return {"message": "File uploaded successfully"}


@app.post("/api/query")
async def query(request: Chat):
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


@app.websocket("/api/chat")
async def chat(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            query_text = await websocket.receive_text()

            try:
                generated_embeddings = get_vector_embeddings(query_text)

                retrieved_knowledge = await get_embeddings(generated_embeddings)
                logger.debug(
                    f"Retrieved {len(retrieved_knowledge)} chunks from database"
                )

                context_chunks = [embedding.chunk for embedding in retrieved_knowledge]

                instruction_prompt = f"""You are a helpful chatbot.
        Use only the following pieces of context to answer the question. Don't make up any new information:
        {chr(10).join([f" - {chunk}" for chunk in context_chunks])}
        """
                buffer = []
                flush_interval = 0.2

                async def flush_buffer():
                    nonlocal buffer
                    if buffer:
                        await websocket.send_text("".join(buffer))
                        buffer.clear()

                response = ollama.chat(
                    model=settings.model_language,
                    messages=[
                        {"role": "system", "content": instruction_prompt},
                        {"role": "user", "content": query_text},
                    ],
                    stream=True,
                )

                for chunk in response:
                    content = chunk["message"]["content"]
                    buffer.append(content)
                    if len(buffer) >= 5:
                        await flush_buffer()
                    else:
                        await asyncio.sleep(flush_interval)

                await flush_buffer()
                logger.info(f"Completed response for query: {query_text}")

            except Exception as e:
                logger.error(f"Chat processing failed: {e}")
                error_message = f"Error processing your message: {str(e)}"
                await websocket.send_text(error_message)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
        await websocket.close()
