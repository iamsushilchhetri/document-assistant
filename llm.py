
import os
from functools import lru_cache

from dotenv import load_dotenv
from groq import Groq, APIError


load_dotenv()

MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")


class LLMError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def get_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise LLMError("GROQ_API_KEY is not configured")

    return Groq(api_key=api_key, timeout=30.0)


def ask_llm(
    system_prompt: str,
    user_prompt: str,
    json_mode: bool = False
) -> str:

    client = get_client()

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            temperature=0.2,
            max_completion_tokens=2048,
            response_format={
                "type": "json_object" if json_mode else "text"
            }
        )

    except APIError as error:
        raise LLMError("LLM request failed") from error

    answer = response.choices[0].message.content

    if not answer or not answer.strip():
        raise LLMError("LLM returned an empty response")

    return answer.strip()
