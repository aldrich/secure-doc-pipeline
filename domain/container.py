from domain.settings import settings
from domain.error import ConfigurationError
from evaluation.evaluation_engine import EvaluationEngine
from extraction.extraction_engine import ExtractionEngine
from services.pipeline import PipelineService

class DependencyContainer:
    def __init__(self, settings=settings):

        self._settings = settings
        self._eval_engine: EvaluationEngine | None = None
        self._extract_engine: ExtractionEngine | None = None
        self._pipeline_service: PipelineService | None = None

    def _retry_kwargs(self):
        return {
            "max_retries": self._settings.llm_max_retries,
            "retry_base_delay": self._settings.llm_retry_base_delay,
            "retry_max_delay": self._settings.llm_retry_max_delay,
            "timeout": self._settings.llm_timeout,
        }

    def create_eval_engine(self) -> EvaluationEngine:
        engine = self._settings.eval_engine
        r = self._retry_kwargs()
        if engine == "openai":
            from evaluation.evaluation_engine import OpenAIEvaluator
            return OpenAIEvaluator(self._settings.openai_api_key, self._settings.openai_model_for_evaluation, **r)
        elif engine == "llama":
            from evaluation.evaluation_engine import LlamaEvaluator
            return LlamaEvaluator(self._settings.llama_model_for_evaluation, self._settings.ollama_host, **r)
        elif engine == "deepseek":
            from evaluation.evaluation_engine import DeepSeekEvaluator
            return DeepSeekEvaluator(self._settings.deepseek_api_key, self._settings.deepseek_model_for_evaluation, self._settings.deepseek_base_url, **r)
        elif engine == "gemini":
            from evaluation.evaluation_engine import GeminiEvaluator
            return GeminiEvaluator(self._settings.gemini_api_key, self._settings.gemini_model_for_evaluation, **r)
        else:
            raise ConfigurationError(f"Unsupported evaluation engine: {engine}")
        
    def create_extract_engine(self) -> ExtractionEngine:
        engine = self._settings.extract_engine
        r = self._retry_kwargs()
        if engine == "openai":
            from extraction.extraction_engine import OpenAIExtractor
            return OpenAIExtractor(self._settings.openai_api_key, self._settings.openai_model_for_extraction, **r)
        elif engine == "llama":
            from extraction.extraction_engine import LlamaExtractor
            return LlamaExtractor(self._settings.llama_model_for_extraction, self._settings.ollama_host, **r)
        elif engine == "deepseek":
            from extraction.extraction_engine import DeepSeekExtractor
            return DeepSeekExtractor(self._settings.deepseek_api_key, 
                                     self._settings.deepseek_model_for_extraction, 
                                     self._settings.deepseek_base_url, **r)
        elif engine == "gemini":
            from extraction.extraction_engine import GeminiExtractor
            return GeminiExtractor(self._settings.gemini_api_key, self._settings.gemini_model_for_extraction, **r)
        else:
            raise ConfigurationError(f"Unsupported extraction engine: {engine}")
        
    def create_pipeline_service(self) -> PipelineService:
        return PipelineService(self.extract_engine, self.eval_engine)

    @property
    def eval_engine(self) -> EvaluationEngine:
        if self._eval_engine is None:
            self._eval_engine = self.create_eval_engine()
        return self._eval_engine
    
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