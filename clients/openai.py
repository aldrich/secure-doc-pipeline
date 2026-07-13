from openai import AsyncOpenAI
from pydantic import BaseModel

from clients.base import LLMClient
from domain.error import ExtractionError


class OpenAIClient(LLMClient):
    def __init__(self, api_key: str, timeout: int = 120) -> None:
        self._api_key = api_key
        self._timeout = timeout
        self._client = AsyncOpenAI(api_key=self._api_key, timeout=timeout)

    def get_name(self) -> str:
        return "openai"

    async def generate_structured(
        self,
        model: str,
        system_prompt: str,
        user_content: str,
        response_schema: type[BaseModel],
    ) -> BaseModel:
        response = await self._client.responses.parse(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            text_format=response_schema,
        )

        if not isinstance(response.output_parsed, response_schema):
            raise ExtractionError(
                f"Unexpected response shape: {type(response.output_parsed)}"
            )

        return response.output_parsed
