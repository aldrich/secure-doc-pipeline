from google import genai
from google.genai import types as genai_types
from pydantic import BaseModel, ValidationError

from clients.base import LLMClient
from domain.error import ExtractionError


class GeminiClient(LLMClient):
    def __init__(self, api_key: str, timeout=120):
        self._api_key = api_key
        self._timeout = timeout
        self._client = genai.Client(
            api_key=self._api_key, http_options={"timeout": timeout * 1000}
        )

    def get_name(self) -> str:
        return 'gemini'

    async def generate_structured(
        self,
        model: str,
        system_prompt: str,
        user_content: str,
        response_schema: type[BaseModel],
    ) -> BaseModel:

        response = await self._client.aio.models.generate_content(
            model=model,
            contents=user_content,
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_schema,
                system_instruction=system_prompt,
                temperature=0,
            ),
        )

        try:
            clean_data = response_schema.model_validate(response.parsed)
        except ValidationError:
            raise ExtractionError(f"Unexpected response shape: {type(response.parsed)}")

        return clean_data
