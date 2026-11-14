
import sqlite3
from contextlib import closing

import pymupdf

import database
from chunking import fixed_chunking, sentence_chunking
from text_extraction import extract_txt, extract_pdf


def test_txt_extraction():
    content = b"Hello World"
    assert extract_txt(content) == "Hello World"


def test_pdf_extraction():
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), "Hello World")
        content = document.tobytes()

    assert "Hello World" in extract_pdf(content)


def test_fixed_chunking():
    chunks = fixed_chunking(
        "ABCDEFGHIJKLMNOPQRST",
        chunk_size=10,
        overlap=3
    )

    assert chunks == [
        "ABCDEFGHIJ",
        "HIJKLMNOPQ",
        "OPQRST"
    ]


def test_sentence_chunking():
    text = "Python is easy. Java supports OOP."

    chunks = sentence_chunking(text, chunk_size=20)

    assert chunks == [
        "Python is easy.",
        "Java supports OOP."
    ]


def test_empty_text():
    assert fixed_chunking("") == []
    assert sentence_chunking("") == []


def test_document_storage(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"

    monkeypatch.setattr(database, "DB_PATH", db_path)

    database.init_database()

    database.save_document(
        document_id="doc-1",
        filename="test.txt",
        strategy="fixed",
        character_count=840,
        chunk_count=3
    )

    with closing(sqlite3.connect(db_path)) as conn:
        result = conn.execute(
            "SELECT filename, chunk_count FROM documents"
        ).fetchone()

    assert result == ("test.txt", 3)


def test_booking_storage(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"

    monkeypatch.setattr(database, "DB_PATH", db_path)

    database.init_database()

    database.save_booking(
        booking_id="booking-1",
        session_id="user1",
        name="Alex Sharma",
        email="alex@example.com",
        interview_date="2030-01-10",
        interview_time="11:00"
    )

    with closing(sqlite3.connect(db_path)) as conn:
        result = conn.execute(
            "SELECT name, email FROM bookings"
        ).fetchone()

    assert result == ("Alex Sharma", "alex@example.com")
