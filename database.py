
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Literal


DB_PATH = Path(__file__).resolve().parent / "assistant.db"


def init_database() -> None:

    with closing(sqlite3.connect(DB_PATH)) as conn, conn:

        conn.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY NOT NULL,
                filename TEXT NOT NULL,
                strategy TEXT NOT NULL
                    CHECK (strategy IN ('fixed', 'sentence')),
                character_count INTEGER NOT NULL
                    CHECK (character_count >= 0),
                chunk_count INTEGER NOT NULL
                    CHECK (chunk_count > 0),
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS bookings (
                booking_id TEXT PRIMARY KEY NOT NULL,
                session_id TEXT NOT NULL,
                name TEXT NOT NULL,
                email TEXT NOT NULL,
                interview_date TEXT NOT NULL,
                interview_time TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
        """)


def save_document(
    document_id: str,
    filename: str,
    strategy: Literal["fixed", "sentence"],
    character_count: int,
    chunk_count: int
) -> None:

    with closing(sqlite3.connect(DB_PATH)) as conn, conn:

        conn.execute("""
            INSERT INTO documents (
                document_id,
                filename,
                strategy,
                character_count,
                chunk_count
            )
            VALUES (?, ?, ?, ?, ?)
        """, (
            document_id,
            filename,
            strategy,
            character_count,
            chunk_count
        ))


def save_booking(
    booking_id: str,
    session_id: str,
    name: str,
    email: str,
    interview_date: str,
    interview_time: str
) -> None:

    with closing(sqlite3.connect(DB_PATH)) as conn, conn:

        conn.execute("""
            INSERT INTO bookings (
                booking_id,
                session_id,
                name,
                email,
                interview_date,
                interview_time
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(booking_id) DO NOTHING
        """, (
            booking_id,
            session_id,
            name,
            email,
            interview_date,
            interview_time
        ))
