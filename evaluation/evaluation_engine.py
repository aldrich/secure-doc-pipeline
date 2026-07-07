import logging
from abc import ABC, abstractmethod

from domain.error import ConfigurationError, EvaluationError
from schemas.clinical_summary import ClinicalSummary
from schemas.evaluation_metrics import SummaryEvaluation
import ollama
from openai import AsyncOpenAI
from google import genai
from google.genai import types as genai_types
from domain.settings import settings
from domain.evaluation_prompt import get_prompt, system_prompt

logger = logging.getLogger(__name__)

class EvaluationEngine(ABC):
    """Abstract base class for evaluation engines."""

    @abstractmethod
    async def evaluate(self,
                       summary_data: ClinicalSummary,
                       source_transcript: str,
                       session_id: str) -> SummaryEvaluation:
        pass
    
    @property
    @abstractmethod
    def model(self) -> str:
        pass

class GeminiEvaluator(EvaluationEngine):
    
    @property
    def model(self) -> str:
        return self._model
    
    def __init__(self, api_key: str, model: str):

        if not api_key:
            message = "GEMINI_API_KEY not found from environment."
            logger.error(message)
            raise ConfigurationError(message)

        if not model:
            message = "GEMINI_MODEL_FOR_EVALUATION not found from environment."
            logger.error(message)
            raise ConfigurationError(message)

        self.client = genai.Client(api_key=api_key)
        self._model = model

    async def evaluate(self, summary_data: ClinicalSummary, source_transcript: str, session_id: str) -> SummaryEvaluation:

        logger.info("evaluation_started", extra={"engine": "gemini", "model": self.model, "session_id": session_id})

        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=get_prompt(source_transcript, summary_data),
            config=genai_types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                response_schema=SummaryEvaluation,
                temperature=0.0,
            )
        )

        raw_parsed = response.parsed
        if not isinstance(raw_parsed, SummaryEvaluation):
            raise EvaluationError(f"Unexpected response shape: {type(raw_parsed)}")

        metrics = raw_parsed
        return metrics

class OpenAIEvaluator(EvaluationEngine):

    @property
    def model(self) -> str:
        return self._model
    
    def __init__(self, api_key: str, model: str):

        if not api_key:
            message = "OPENAI_API_KEY not found from environment."
            logger.error(message)
            raise ConfigurationError(message)

        if not model:
            message = "OPENAI_MODEL_FOR_EVALUATION not found from environment."
            logger.error(message)
            raise ConfigurationError(message)

        self._model = model
        self.api_key = api_key
        self.client = AsyncOpenAI(api_key=self.api_key)

    async def evaluate(self, summary_data: ClinicalSummary, source_transcript: str, session_id: str) -> SummaryEvaluation:

        logger.info("evaluation_started", extra={"engine": "openai", "model": self.model, "session_id": session_id})

        response = await self.client.responses.parse(
            model=self.model,
            input=[
                { "role": "system", "content": system_prompt },
                { "role": "user", "content": get_prompt(source_transcript, summary_data) }
            ],
            text_format=SummaryEvaluation,
        )

        clinical_summary = response.output_parsed

        if not isinstance(clinical_summary, SummaryEvaluation):
            raise EvaluationError(f"Unexpected response shape: {type(clinical_summary)}")

        return clinical_summary

class LlamaEvaluator(EvaluationEngine):

    @property
    def model(self) -> str:
        return self._model
    def __init__(self, model: str):
        if not model:
            message = "LLAMA_MODEL_FOR_EVALUATION not found from environment."
            logger.warning(message)
            raise ConfigurationError(message)

        self._model = model
        
    async def evaluate(self, summary_data: ClinicalSummary, source_transcript: str, session_id: str) -> SummaryEvaluation:

        logger.info("evaluation_started", extra={"engine": "llama", "model": self.model, "session_id": session_id})

        async with ollama.AsyncClient(host=settings.ollama_host) as client:
            response = await client.chat(
                model=self.model,
                messages=[
                    { 'role': 'system', 'content': system_prompt },
                    { 'role': 'user', 'content': get_prompt(source_transcript, summary_data) }
                ],
                format=SummaryEvaluation.model_json_schema()
            )

        raw_content = response['message']['content']
        summary_evaluation = SummaryEvaluation.model_validate_json(raw_content)
        return summary_evaluation

class DeepSeekEvaluator(EvaluationEngine):

    @property
    def model(self) -> str:
        return self._model
        
    def __init__(self, api_key: str, model: str, base_url: str):
        if not api_key:
            message = "DEEPSEEK_API_KEY not found from environment."
            logger.error(message)
            raise ConfigurationError(message)

        if not model:
            message = "DEEPSEEK_MODEL_FOR_EVALUATION not found from environment."
            logger.error(message)
            raise ConfigurationError(message)

        self._model = model
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def evaluate(self, summary_data: ClinicalSummary, source_transcript: str, session_id: str) -> SummaryEvaluation:
        logger.info("evaluation_started", extra={"engine": "deepseek", "model": self.model, "session_id": session_id})

        # this is a DeepSeek-specific way to get the response in the correct format
        # source: https://api-docs.deepseek.com/guides/json_mode/
        response_shape_spec = f'respond in the following JSON format: {SummaryEvaluation.model_json_schema()}'

        modified_system_prompt = "\n".join([system_prompt, response_shape_spec])
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                { "role": "system", "content": modified_system_prompt },
                { "role": "user", "content": get_prompt(source_transcript, summary_data) }
            ],
            response_format={ "type": "json_object" },
            temperature=0,
        )

        raw_content = response.choices[0].message.content
        if raw_content is None:
            raise EvaluationError("Empty response from DeepSeek")

        summary_evaluation = SummaryEvaluation.model_validate_json(raw_content)
        return summary_evaluation