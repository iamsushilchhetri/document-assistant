
import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from uuid import uuid4

import pymupdf
from fastapi import FastAPI, UploadFile, HTTPException, Form
from pydantic import BaseModel, Field
from redis.exceptions import RedisError

from text_extraction import extract_txt, extract_pdf
from chunking import fixed_chunking, sentence_chunking
from embeddings import generate_embeddings
from vector_store import store_chunks, SearchResult
from database import init_database, save_document
from rag import generate_answer
from llm import LLMError
from memory import get_history, save_turn
from booking import handle_booking


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    init_database()
    yield


app = FastAPI(
    title="Document Assistant",
    version="0.1.0",
    lifespan=lifespan
)

MAX_FILE_SIZE = 5 * 1024 * 1024


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[SearchResult]


class IngestResponse(BaseModel):
    document_id: str
    filename: str
    strategy: Literal["fixed", "sentence"]
    character_count: int
    chunk_count: int
    chunks: list[str]
    embedding_count: int
    embedding_dimension: int


@app.post("/chat")
def chat(request: ChatRequest) -> ChatResponse:

    session_id = request.session_id.strip()
    message = request.message.strip()

    if not session_id:
        raise HTTPException(
            status_code=400,
            detail="Session ID cannot be empty"
        )

    if not message:
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty"
        )

    try:
        history = get_history(session_id)

        booking_answer = handle_booking(
            session_id,
            message,
            history
        )

        sources: list[SearchResult]

        if booking_answer is not None:
            answer = booking_answer
            sources = []
        else:
            answer, sources = generate_answer(
                message,
                history
            )

        save_turn(
            session_id=session_id,
            question=message,
            answer=answer
        )

    except RedisError as error:
        raise HTTPException(
            status_code=503,
            detail="Chat memory service unavailable"
        ) from error

    except LLMError as error:
        raise HTTPException(
            status_code=503,
            detail="Language model service unavailable"
        ) from error

    except sqlite3.Error as error:
        raise HTTPException(
            status_code=503,
            detail="Database service unavailable"
        ) from error

    return ChatResponse(
        session_id=session_id,
        answer=answer,
        sources=sources
    )


@app.post("/ingest")
def ingest_document(
    file: UploadFile,
    strategy: Literal["fixed", "sentence"] = Form("fixed")
) -> IngestResponse:

    filename = file.filename or ""
    extension = Path(filename).suffix.lower()

    if extension not in (".txt", ".pdf"):
        raise HTTPException(
            status_code=415,
            detail="Only PDF and TXT files are supported"
        )

    content = file.file.read(MAX_FILE_SIZE + 1)

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File is too large"
        )

    if not content:
        raise HTTPException(
            status_code=400,
            detail="File is empty"
        )

    try:
        if extension == ".txt":
            text = extract_txt(content)
        else:
            text = extract_pdf(content)

    except (ValueError, pymupdf.FileDataError) as error:
        raise HTTPException(
            status_code=400,
            detail="Unable to extract text from file"
        ) from error

    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="No readable text found in document"
        )

    if strategy == "fixed":
        chunks = fixed_chunking(text)
    else:
        chunks = sentence_chunking(text)

    vectors = generate_embeddings(chunks)

    document_id = str(uuid4())

    store_chunks(
        document_id=document_id,
        filename=filename,
        chunks=chunks,
        vectors=vectors
    )

    save_document(
        document_id=document_id,
        filename=filename,
        strategy=strategy,
        character_count=len(text),
        chunk_count=len(chunks)
    )

    return IngestResponse(
        document_id=document_id,
        filename=filename,
        strategy=strategy,
        character_count=len(text),
        chunk_count=len(chunks),
        chunks=chunks,
        embedding_count=len(vectors),
        embedding_dimension=len(vectors[0]) if vectors else 0
    )
