from sqlalchemy.ext.asyncio import async_sessionmaker

from clients.base import LLMClient
from clients.deepseek import DeepSeekClient
from clients.gemini import GeminiClient
from clients.ollama import OllamaClient
from clients.openai import OpenAIClient
from domain.database import create_session_factory
from domain.error import ConfigurationError
from domain.repository import EvaluationRepository
from domain.settings import settings
from evaluation.evaluation_engine import EvaluationEngine
from extraction.extraction_engine import ExtractionEngine
from services.pipeline import PipelineService


class DependencyContainer:
    def __init__(self, settings=settings):

        self._model_for_extraction = {
            "gemini": settings.gemini_model_for_extraction,
            "llama": settings.llama_model_for_extraction,
            "openai": settings.openai_model_for_extraction,
            "deepseek": settings.deepseek_model_for_extraction,
        }

        self._model_for_evaluation = {
            "gemini": settings.gemini_model_for_evaluation,
            "llama": settings.llama_model_for_evaluation,
            "openai": settings.openai_model_for_evaluation,
            "deepseek": settings.deepseek_model_for_evaluation,
        }

        self._settings = settings
        self._extract_engine: ExtractionEngine | None = None
        self._eval_engine: EvaluationEngine | None = None
        self._pipeline_service: PipelineService | None = None

        self._session_factory: async_sessionmaker | None = None
        self._evaluation_repo: EvaluationRepository | None = None

    def _retry_kwargs(self):
        return {
            "max_retries": self._settings.llm_max_retries,
            "retry_base_delay": self._settings.llm_retry_base_delay,
            "retry_max_delay": self._settings.llm_retry_max_delay,
        }

    def _create_client(self, engine_type: str) -> LLMClient:
        t = self._settings.llm_timeout
        if engine_type == "openai":
            return OpenAIClient(self._settings.openai_api_key, t)
        elif engine_type == "gemini":
            return GeminiClient(self._settings.gemini_api_key, t)
        elif engine_type == "llama":
            return OllamaClient(self._settings.ollama_host, t)
        elif engine_type == "deepseek":
            return DeepSeekClient(
                self._settings.deepseek_api_key, self._settings.deepseek_base_url, t
            )

        raise ConfigurationError(f"Unsupported engine: {engine_type}")

    def create_extract_engine(self) -> ExtractionEngine:
        client = self._create_client(self._settings.extract_engine)
        return ExtractionEngine(
            client,
            self._model_for_extraction[self._settings.extract_engine],
            **self._retry_kwargs(),
        )

    def create_eval_engine(self) -> EvaluationEngine:
        client = self._create_client(self._settings.eval_engine)
        return EvaluationEngine(
            client,
            self._model_for_evaluation[self._settings.eval_engine],
            **self._retry_kwargs(),
        )

    def make_session_factory(self):
        if self._session_factory is None:
            self._session_factory = create_session_factory()
        return self._session_factory

    def create_evaluation_repo(self) -> EvaluationRepository:
        if self._evaluation_repo is None:
            self._evaluation_repo = EvaluationRepository(self.make_session_factory())
        return self._evaluation_repo

    def create_pipeline_service(self) -> PipelineService:
        return PipelineService(self.extract_engine, self.eval_engine, self.evaluation_repo)

    @property
    def eval_engine(self) -> EvaluationEngine:
        if self._eval_engine is None:
            self._eval_engine = self.create_eval_engine()
        return self._eval_engine

    @property
    def evaluation_repo(self) -> EvaluationRepository:
        if self._evaluation_repo is None:
            self._evaluation_repo = self.create_evaluation_repo()
        return self._evaluation_repo

    @property
    def extract_engine(self) -> ExtractionEngine:
        if self._extract_engine is None:
            self._extract_engine = self.create_extract_engine()
        return self._extract_engine

    @property
    def pipeline_service(self) -> PipelineService:
        if self._pipeline_service is None:
            self._pipeline_service = self.create_pipeline_service()
        return self._pipeline_service
