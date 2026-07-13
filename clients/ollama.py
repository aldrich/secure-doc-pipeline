import ollama
from pydantic import BaseModel

from clients.base import LLMClient


class OllamaClient(LLMClient):
    def __init__(self, ollama_host: str, timeout=120):
        self._ollama_host = ollama_host
        self._timeout = timeout

    def get_name(self) -> str:
        return 'ollama'

    async def generate_structured(
        self,
        model: str,
        system_prompt: str,
        user_content: str,
        response_schema: type[BaseModel],
    ) -> BaseModel:

        async with ollama.AsyncClient(
            host=self._ollama_host, timeout=self._timeout
        ) as client:
            response = await client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                format=response_schema.model_json_schema(),
            )

        raw_content = response["message"]["content"]
        result = response_schema.model_validate_json(raw_content)
        return result
