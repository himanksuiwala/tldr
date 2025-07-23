import asyncio
import logging
import tempfile
from contextlib import asynccontextmanager

import ollama
import pymupdf
from fastapi import FastAPI, UploadFile, WebSocket, WebSocketDisconnect

from .connection import database
from .models import Chat
from .service import get_embeddings, get_vector_embeddings, store_chunks_with_embeddings
from .settings import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    maximum_connection_retries = 3
    for connection_attempt in range(maximum_connection_retries):
        try:
            await database.connect()
            logger.info("Connected to the database successfully.")
            break
        except Exception as database_connection_exception:
            logger.error(
                f"Attempt {connection_attempt + 1} to connect to the database failed: {database_connection_exception}"
            )
            if connection_attempt == maximum_connection_retries - 1:
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
async def upload(uploaded_file: UploadFile):
    temporary_file_path = None
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temporary_file:
        file_contents = await uploaded_file.read()
        temporary_file.write(file_contents)
        temporary_file_path = temporary_file.name

    extracted_document_text = ""
    pdf_document = pymupdf.open(temporary_file_path)
    for page_number, current_page in enumerate(pdf_document):
        raw_page_text = current_page.get_text()

        processed_page_text = str.strip(raw_page_text)

        extracted_document_text += processed_page_text
        text_lines = extracted_document_text.split("\n")

    processed_text_chunks = [
        text_chunk.strip() for text_chunk in text_lines if text_chunk.strip()
    ]

    await store_chunks_with_embeddings(processed_text_chunks)

    return {
        "message": f"Successfully uploaded and stored {len(processed_text_chunks)} chunks"
    }


@app.post("/api/query")
async def query(chat_request: Chat):
    try:
        user_query_embeddings = get_vector_embeddings(chat_request.query)

        similar_knowledge_chunks = await get_embeddings(user_query_embeddings)
        logger.debug(f"Retrieved {len(similar_knowledge_chunks)} chunks from database")

        contextual_text_chunks = [
            knowledge_embedding.chunk
            for knowledge_embedding in similar_knowledge_chunks
        ]

        system_instruction_prompt = f"""You are a helpful chatbot.
        Use only the following pieces of context to answer the question. Don't make up any new information:
        {chr(10).join([f" - {text_chunk}" for text_chunk in contextual_text_chunks])}
        """

        llm_response = ollama.chat(
            model=settings.model_language,
            messages=[
                {"role": "system", "content": system_instruction_prompt},
                {"role": "user", "content": chat_request.query},
            ],
        )

        return {
            "query": chat_request.query,
            "response": llm_response["message"]["content"],
        }

    except Exception as query_exception:
        logger.error(f"Query processing failed: {query_exception}")
        return {"error": f"Query processing failed: {str(query_exception)}"}


@app.websocket("/api/chat")
async def chat(websocket_connection: WebSocket):
    await websocket_connection.accept()
    try:
        while True:
            user_message_text = await websocket_connection.receive_text()

            try:
                user_message_embeddings = get_vector_embeddings(user_message_text)

                similar_knowledge_chunks = await get_embeddings(user_message_embeddings)
                logger.debug(
                    f"Retrieved {len(similar_knowledge_chunks)} chunks from database"
                )

                contextual_text_chunks = [
                    knowledge_embedding.chunk
                    for knowledge_embedding in similar_knowledge_chunks
                ]

                system_instruction_prompt = f"""You are a helpful chatbot.
        Use only the following pieces of context to answer the question. Don't make up any new information,
        If the answer is not contained in the context, say "I don't know":
        {chr(10).join([f" - {text_chunk}" for text_chunk in contextual_text_chunks])}
        """
                response_buffer = []
                buffer_flush_interval = 0.2

                async def flush_response_buffer():
                    nonlocal response_buffer
                    if response_buffer:
                        await websocket_connection.send_text("".join(response_buffer))
                        response_buffer.clear()

                streaming_llm_response = ollama.chat(
                    model=settings.model_language,
                    messages=[
                        {"role": "system", "content": system_instruction_prompt},
                        {"role": "user", "content": user_message_text},
                    ],
                    stream=True,
                )

                for response_chunk in streaming_llm_response:
                    chunk_content = response_chunk["message"]["content"]
                    response_buffer.append(chunk_content)
                    if len(response_buffer) >= 5:
                        await flush_response_buffer()
                    else:
                        await asyncio.sleep(buffer_flush_interval)

                await flush_response_buffer()
                logger.info(f"Completed response for query: {user_message_text}")

            except Exception as chat_processing_exception:
                logger.error(f"Chat processing failed: {chat_processing_exception}")
                error_message = (
                    f"Error processing your message: {str(chat_processing_exception)}"
                )
                await websocket_connection.send_text(error_message)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
        await websocket_connection.close()
