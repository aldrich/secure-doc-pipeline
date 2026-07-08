from unittest.mock import Mock

import pytest

from domain.container import DependencyContainer
from domain.error import ConfigurationError
from evaluation.evaluation_engine import (
    DeepSeekEvaluator,
    GeminiEvaluator,
    LlamaEvaluator,
    OpenAIEvaluator,
)
from extraction.extraction_engine import (
    DeepSeekExtractor,
    GeminiExtractor,
    LlamaExtractor,
    OpenAIExtractor,
)


class TestDependencyContainer:

    def make_settings(self, **overrides):
        settings = Mock()
        settings.extract_engine = "llama"
        settings.eval_engine = "llama"
        settings.openai_api_key = "sk-test"
        settings.openai_model_for_extraction = "gpt-4.1-nano"
        settings.openai_model_for_evaluation = "gpt-4.1-mini"
        settings.gemini_api_key = "gemini-test"
        settings.gemini_model_for_extraction = "gemini-2.5-flash-lite"
        settings.gemini_model_for_evaluation = "gemini-2.5-flash"
        settings.llama_model_for_extraction = "llama3.2:3b"
        settings.llama_model_for_evaluation = "llama3.1:8b"
        settings.deepseek_api_key = "ds-test"
        settings.deepseek_model_for_extraction = "deepseek-v4-flash"
        settings.deepseek_model_for_evaluation = "deepseek-v4-pro"
        settings.deepseek_base_url = "https://api.deepseek.com"
        for k, v in overrides.items():
            setattr(settings, k, v)
        return settings

    def test_create_openai_extractor(self):
        settings = self.make_settings(extract_engine="openai")
        container = DependencyContainer(settings)
        engine = container.create_extract_engine()
        assert isinstance(engine, OpenAIExtractor)

    def test_create_gemini_extractor(self):
        settings = self.make_settings(extract_engine="gemini")
        container = DependencyContainer(settings)
        engine = container.create_extract_engine()
        assert isinstance(engine, GeminiExtractor)

    def test_create_llama_extractor(self):
        settings = self.make_settings(extract_engine="llama")
        container = DependencyContainer(settings)
        engine = container.create_extract_engine()
        assert isinstance(engine, LlamaExtractor)

    def test_create_deepseek_extractor(self):
        settings = self.make_settings(extract_engine="deepseek")
        container = DependencyContainer(settings)
        engine = container.create_extract_engine()
        assert isinstance(engine, DeepSeekExtractor)

    def test_create_openai_evaluator(self):
        settings = self.make_settings(eval_engine="openai")
        container = DependencyContainer(settings)
        engine = container.create_eval_engine()
        assert isinstance(engine, OpenAIEvaluator)

    def test_create_gemini_evaluator(self):
        settings = self.make_settings(eval_engine="gemini")
        container = DependencyContainer(settings)
        engine = container.create_eval_engine()
        assert isinstance(engine, GeminiEvaluator)

    def test_create_llama_evaluator(self):
        settings = self.make_settings(eval_engine="llama")
        container = DependencyContainer(settings)
        engine = container.create_eval_engine()
        assert isinstance(engine, LlamaEvaluator)

    def test_create_deepseek_evaluator(self):
        settings = self.make_settings(eval_engine="deepseek")
        container = DependencyContainer(settings)
        engine = container.create_eval_engine()
        assert isinstance(engine, DeepSeekEvaluator)

    def test_unsupported_extraction_engine(self):
        settings = self.make_settings(extract_engine="invalid")
        container = DependencyContainer(settings)
        with pytest.raises(ConfigurationError, match="Unsupported extraction engine"):
            container.create_extract_engine()

    def test_unsupported_evaluation_engine(self):
        settings = self.make_settings(eval_engine="invalid")
        container = DependencyContainer(settings)
        with pytest.raises(ConfigurationError, match="Unsupported evaluation engine"):
            container.create_eval_engine()

    def test_eval_engine_property_caches(self):
        settings = self.make_settings(eval_engine="openai")
        container = DependencyContainer(settings)
        engine1 = container.eval_engine
        engine2 = container.eval_engine
        assert engine1 is engine2

    def test_extract_engine_property_caches(self):
        settings = self.make_settings(extract_engine="openai")
        container = DependencyContainer(settings)
        engine1 = container.extract_engine
        engine2 = container.extract_engine
        assert engine1 is engine2