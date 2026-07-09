from domain.settings import settings
from domain.error import ConfigurationError
from evaluation.evaluation_engine import EvaluationEngine, OpenAIEvaluator, LlamaEvaluator, DeepSeekEvaluator, GeminiEvaluator
from extraction.extraction_engine import ExtractionEngine, OpenAIExtractor, LlamaExtractor, DeepSeekExtractor, GeminiExtractor
from services.pipeline import PipelineService

class DependencyContainer:
    def __init__(self, settings=settings):

        self._settings = settings
        self._eval_engine: EvaluationEngine | None = None
        self._extract_engine: ExtractionEngine | None = None
        self._pipeline_service: PipelineService | None = None

    def create_eval_engine(self) -> EvaluationEngine:
        engine = self._settings.eval_engine
        if engine == "openai":
            return OpenAIEvaluator(self._settings.openai_api_key, self._settings.openai_model_for_evaluation)
        elif engine == "llama":
            return LlamaEvaluator(self._settings.llama_model_for_evaluation)
        elif engine == "deepseek":
            return DeepSeekEvaluator(self._settings.deepseek_api_key, self._settings.deepseek_model_for_evaluation, self._settings.deepseek_base_url)
        elif engine == "gemini":
            return GeminiEvaluator(self._settings.gemini_api_key, self._settings.gemini_model_for_evaluation)
        else:
            raise ConfigurationError(f"Unsupported evaluation engine: {engine}")
        
    def create_extract_engine(self) -> ExtractionEngine:
        engine = self._settings.extract_engine
        if engine == "openai":
            return OpenAIExtractor(self._settings.openai_api_key, self._settings.openai_model_for_extraction)
        elif engine == "llama":
            return LlamaExtractor(self._settings.llama_model_for_extraction)
        elif engine == "deepseek":
            return DeepSeekExtractor(self._settings.deepseek_api_key, self._settings.deepseek_model_for_extraction, self._settings.deepseek_base_url)
        elif engine == "gemini":
            return GeminiExtractor(self._settings.gemini_api_key, self._settings.gemini_model_for_extraction)
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