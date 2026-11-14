
# Document Assistant

A FastAPI backend for document ingestion, conversational
RAG and interview booking.

## Features

- PDF and TXT document extraction
- Fixed-size and sentence-based chunking
- Semantic embeddings using all-MiniLM-L6-v2
- Persistent vector storage using Qdrant
- Document metadata storage using SQLite
- Custom retrieval-augmented generation
- Redis-based conversation memory
- Multi-turn interview booking using an LLM
- Booking validation and confirmation

## Requirements

- Python 3.14 (tested with 3.14.7)
- uv
- Docker
- Groq API key

## Installation

Clone the repository:

```bash
git clone https://github.com/iamsushilchhetri/document-assistant.git
cd document-assistant
uv sync
```

Create a `.env` file using `.env.example` as a template
and provide your own Groq API key.

Start Redis:

```bash
docker run -d --name document-redis -p 127.0.0.1:6379:6379 -v document_redis_data:/data redis:8 redis-server --appendonly yes
```

If the container already exists:

```bash
docker start document-redis
```

Start the backend:

```bash
uv run uvicorn main:app --reload
```

API documentation:
http://127.0.0.1:8000/docs

## API Endpoints

### POST /ingest

Upload a PDF or TXT document using multipart form data.

Parameters:
- file: document to upload
- strategy: fixed or sentence

Maximum file size: 5 MiB.

### POST /chat

Example request:

```json
{
    "session_id": "user1",
    "message": "What skills are required?"
}
```

Upload test.txt through /ingest before asking questions
about its contents.

## Interview Booking

Send a message requesting an interview booking.

The assistant collects the candidate's name, email,
preferred date and time across multiple turns.

Reply "confirm" to save the booking or "cancel"
to discard it.

## Architecture

Ingestion:
Document -> Extraction -> Chunking -> Embeddings
-> Qdrant + SQLite

Chat:
Query -> Redis History -> Intent Detection
-> Retrieval + Context -> LLM Response

Booking:
Intent -> Field Extraction -> Redis State
-> Validation -> Confirmation -> SQLite

## Testing

```bash
uv run python -m pytest -v
```

Seven local automated tests cover document extraction,
chunking, metadata storage and booking storage.
