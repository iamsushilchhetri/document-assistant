
import os
from functools import lru_cache
from typing import Literal, cast

from dotenv import load_dotenv
from pydantic import BaseModel
from redis import Redis


load_dotenv()

REDIS_URL = os.getenv(
    "REDIS_URL", "redis://localhost:6379/0"
)

MAX_MESSAGES = 12
SESSION_TTL = 24 * 60 * 60


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


@lru_cache(maxsize=1)
def get_redis() -> Redis:
    return Redis.from_url(
        REDIS_URL,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5
    )


def session_key(session_id: str) -> str:
    return f"chat:{session_id}"


def get_history(session_id: str) -> list[ChatMessage]:
    client = get_redis()

    messages = cast(
        list[str],
        client.lrange(session_key(session_id), 0, -1)
    )

    return [
        ChatMessage.model_validate_json(message)
        for message in messages
    ]


def save_turn(
    session_id: str,
    question: str,
    answer: str
) -> None:

    client = get_redis()
    key = session_key(session_id)

    user_message = ChatMessage(
        role="user",
        content=question
    )

    assistant_message = ChatMessage(
        role="assistant",
        content=answer
    )

    with client.pipeline(transaction=True) as pipe:

        pipe.rpush(
            key,
            user_message.model_dump_json(),
            assistant_message.model_dump_json()
        )

        pipe.ltrim(key, -MAX_MESSAGES, -1)
        pipe.expire(key, SESSION_TTL)

        pipe.execute()


def clear_history(session_id: str) -> None:
    client = get_redis()
    client.delete(session_key(session_id))
