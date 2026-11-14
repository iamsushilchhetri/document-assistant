
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from uuid import uuid4

from pydantic import BaseModel
from qdrant_client import QdrantClient, models


DB_PATH = Path(__file__).resolve().parent / "qdrant_data"
COLLECTION_NAME = "documents"
VECTOR_SIZE = 384

db_lock = Lock()


class SearchResult(BaseModel):
    document_id: str
    filename: str
    chunk_index: int
    text: str
    score: float


@contextmanager
def open_database() -> Iterator[QdrantClient]:
    with db_lock:
        client = QdrantClient(path=str(DB_PATH))

        try:
            if not client.collection_exists(COLLECTION_NAME):
                client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=models.VectorParams(
                        size=VECTOR_SIZE,
                        distance=models.Distance.COSINE
                    )
                )

            yield client

        finally:
            client.close()


def store_chunks(
    document_id: str,
    filename: str,
    chunks: list[str],
    vectors: list[list[float]]
) -> None:

    if len(chunks) != len(vectors):
        raise ValueError("Chunk and vector counts must match")

    points: list[models.PointStruct] = []

    for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
        point = models.PointStruct(
            id=str(uuid4()),
            vector=vector,
            payload={
                "document_id": document_id,
                "filename": filename,
                "chunk_index": index,
                "text": chunk
            }
        )

        points.append(point)

    with open_database() as client:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True
        )


def search_chunks(
    query_vector: list[float],
    limit: int = 3
) -> list[SearchResult]:

    with open_database() as client:
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=limit,
            with_payload=True
        ).points

    return [
        SearchResult.model_validate({
            **(result.payload or {}),
            "score": result.score
        })
        for result in results
    ]
