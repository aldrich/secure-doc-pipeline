from openai import AsyncOpenAI
from pydantic import BaseModel

from clients.base import LLMClient
from domain.error import ExtractionError


class DeepSeekClient(LLMClient):
    def __init__(self, api_key: str, base_url: str, timeout=120):
        self._api_key = api_key
        self._timeout = timeout
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def get_name(self) -> str:
        return "deepseek"

    async def generate_structured(
        self,
        model: str,
        system_prompt: str,
        user_content: str,
        response_schema: type[BaseModel],
    ) -> BaseModel:

        response_shape_spec = f"respond in the following JSON format: {response_schema.model_json_schema()}"
        modified_system_prompt = "\n".join([system_prompt, response_shape_spec])

        response = await self._client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": modified_system_prompt},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )

        raw_content = response.choices[0].message.content
        if raw_content is None:
            raise ExtractionError("Empty response from DeepSeek")

        result = response_schema.model_validate_json(raw_content)
        return result
