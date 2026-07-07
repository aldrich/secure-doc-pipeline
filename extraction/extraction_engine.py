import logging

from abc import ABC, abstractmethod

from domain.error import ConfigurationError, ExtractionError
from schemas.clinical_summary import ClinicalSummary
from domain.settings import settings
from domain.extraction_prompt import system_prompt

import ollama
from openai import AsyncOpenAI
from google import genai
from google.genai import types as genai_types

logger = logging.getLogger(__name__)

class ExtractionEngine(ABC):
    """Abstract base class for extraction engines."""

    @abstractmethod
    async def extract(self, source_transcript: str, session_id: str) -> ClinicalSummary:
        pass
    
    @property
    @abstractmethod
    def model(self) -> str:
        pass
    
class OpenAIExtractor(ExtractionEngine):

    @property
    def model(self) -> str:
        return self._model

    def __init__(self, api_key: str, model: str):
        if not model:
            message = "OPENAI_MODEL_FOR_EXTRACTION not found from environment."
            logger.warning(message)
            raise ConfigurationError(message)

        if not api_key:
            message = "OPENAI_API_KEY not found from environment."
            logger.warning(message)
            raise ConfigurationError(message)

        self._model = model
        self.api_key = api_key
        self.client = AsyncOpenAI(api_key=self.api_key)

    async def extract(self, source_transcript: str, session_id: str) -> ClinicalSummary:

        logger.info("extraction_started", extra={"engine": "openai", "model": self.model, "session_id": session_id})

        response = await self.client.responses.parse(
            model=self.model,
            input=[
                { "role": "system", "content": system_prompt },
                { "role": "user", "content": f"<transcript>{source_transcript}</transcript>" }
            ],
            text_format=ClinicalSummary
        )

        if not isinstance(response.output_parsed, ClinicalSummary):
            raise ExtractionError(f"Unexpected response shape: {type(response.output_parsed)}")

        return response.output_parsed

class GeminiExtractor(ExtractionEngine):

    @property
    def model(self) -> str:
        return self._model
        
    def __init__(self, api_key: str, model: str):
        if not model:
            message = "GEMINI_MODEL_FOR_EXTRACTION not found from environment."
            logger.warning(message)
            raise ConfigurationError(message)

        if not api_key:
            message = "GEMINI_API_KEY not found from environment."
            logger.warning(message)
            raise ConfigurationError(message)

        self._model = model
        self.api_key = api_key
        self.client = genai.Client(api_key=self.api_key)

    async def extract(self, source_transcript: str, session_id: str) -> ClinicalSummary:

        logger.info("extraction_started", extra={"engine": "gemini", "model": self.model, "session_id": session_id})

        response = self.client.models.generate_content(
            model=self.model,
            contents=f"""<transcript>
{source_transcript}
</transcript>
""",
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ClinicalSummary,
                system_instruction=system_prompt,
                temperature=0,
            ),
        )

        clean_data = ClinicalSummary.model_validate(response.parsed)
        return clean_data

class LlamaExtractor(ExtractionEngine):

    @property
    def model(self) -> str:
        return self._model
        
    def __init__(self, model: str):
        if not model:
            message = "LLAMA_MODEL_FOR_EXTRACTION not found from environment."
            logger.warning(message)
            raise ConfigurationError(message)

        self._model = model

    async def extract(self, source_transcript: str, session_id: str) -> ClinicalSummary:

        logger.info("extraction_started", extra={"engine": "llama", "model": self.model, "session_id": session_id})

        async with ollama.AsyncClient(host=settings.ollama_host) as client:
            response = await client.chat(
                model=self.model,
                messages=[
                    { 'role': 'system', 'content': system_prompt },
                    { 'role': 'user', 'content': f"<transcript>{source_transcript}</transcript>" }
                ],
                format=ClinicalSummary.model_json_schema()
            )

        raw_content = response['message']['content']
        summary_evaluation = ClinicalSummary.model_validate_json(raw_content)
        return summary_evaluation

class DeepSeekExtractor(ExtractionEngine):

    @property
    def model(self) -> str:
        return self._model
    
    def __init__(self, api_key: str, model: str, base_url: str):
        if not api_key:
            message = "DEEPSEEK_API_KEY not found from environment."
            logger.warning(message)
            raise ConfigurationError(message)

        if not model:
            message = "DEEPSEEK_MODEL_FOR_EXTRACTION not found from environment."
            logger.warning(message)
            raise ConfigurationError(message)

        self._model = model
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def extract(self, source_transcript: str, session_id: str) -> ClinicalSummary:
        logger.info("extraction_started", extra={"engine": "deepseek", "model": self.model, "session_id": session_id})
        
        # source: https://api-docs.deepseek.com/guides/json_mode/
        response_shape_spec = f'respond in the following JSON format: {ClinicalSummary.model_json_schema()}'

        modified_system_prompt = "\n".join([system_prompt, response_shape_spec])

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                { "role": "system", "content": modified_system_prompt },
                { "role": "user", "content": f"<transcript>{source_transcript}</transcript>" }
            ],
            response_format={ "type": "json_object" },
            temperature=0,
        )

        raw_content = response.choices[0].message.content
        if raw_content is None:
            raise ExtractionError("Empty response from DeepSeek")
        
        logger.info("extraction_raw_response", extra={"engine": "deepseek", "model": self.model, "session_id": session_id, "raw_content_length": len(raw_content)})

        clinical_summary = ClinicalSummary.model_validate_json(raw_content)
        return clinical_summary