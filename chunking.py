
import re


def fixed_chunking(
    text: str,
    chunk_size: int = 400,
    overlap: int = 50
) -> list[str]:

    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("Invalid chunk size or overlap")

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def sentence_chunking(
    text: str,
    chunk_size: int = 400
) -> list[str]:

    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive")

    if not text.strip():
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    chunks: list[str] = []
    current = ""

    for sentence in sentences:

        if len(sentence) > chunk_size:
            if current:
                chunks.append(current)
                current = ""

            chunks.extend(
                fixed_chunking(sentence, chunk_size, overlap=0)
            )
            continue

        candidate = f"{current} {sentence}".strip()

        if len(candidate) > chunk_size:
            chunks.append(current)
            current = sentence
        else:
            current = candidate

    if current:
        chunks.append(current)

    return chunks
