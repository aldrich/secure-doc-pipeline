from unittest.mock import Mock

import pytest

from clients.deepseek import DeepSeekClient
from clients.gemini import GeminiClient
from clients.ollama import OllamaClient
from clients.openai import OpenAIClient
from domain.container import DependencyContainer
from domain.error import ConfigurationError


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
        settings.llm_max_retries = 3
        settings.llm_retry_base_delay = 1.0
        settings.llm_retry_max_delay = 30.0
        settings.llm_timeout = 120
        for k, v in overrides.items():
            setattr(settings, k, v)
        return settings

    def test_create_openai_extractor_client(self):
        settings = self.make_settings(extract_engine="openai")
        container = DependencyContainer(settings)
        engine = container.create_extract_engine()
        assert isinstance(engine.client, OpenAIClient)

    def test_create_gemini_extractor_client(self):
        settings = self.make_settings(extract_engine="gemini")
        container = DependencyContainer(settings)
        engine = container.create_extract_engine()
        assert isinstance(engine.client, GeminiClient)

    def test_create_llama_extractor_client(self):
        settings = self.make_settings(extract_engine="llama")
        container = DependencyContainer(settings)
        engine = container.create_extract_engine()
        assert isinstance(engine.client, OllamaClient)

    def test_create_deepseek_extractor_client(self):
        settings = self.make_settings(extract_engine="deepseek")
        container = DependencyContainer(settings)
        engine = container.create_extract_engine()
        assert isinstance(engine.client, DeepSeekClient)

    def test_create_openai_evaluator_client(self):
        settings = self.make_settings(eval_engine="openai")
        container = DependencyContainer(settings)
        engine = container.create_eval_engine()
        assert isinstance(engine.client, OpenAIClient)

    def test_create_gemini_evaluator_client(self):
        settings = self.make_settings(eval_engine="gemini")
        container = DependencyContainer(settings)
        engine = container.create_eval_engine()
        assert isinstance(engine.client, GeminiClient)

    def test_create_llama_evaluator_client(self):
        settings = self.make_settings(eval_engine="llama")
        container = DependencyContainer(settings)
        engine = container.create_eval_engine()
        assert isinstance(engine.client, OllamaClient)

    def test_create_deepseek_evaluator_client(self):
        settings = self.make_settings(eval_engine="deepseek")
        container = DependencyContainer(settings)
        engine = container.create_eval_engine()
        assert isinstance(engine.client, DeepSeekClient)

    def test_unsupported_extraction_engine(self):
        settings = self.make_settings(extract_engine="invalid")
        container = DependencyContainer(settings)
        with pytest.raises(ConfigurationError, match="Unsupported engine: invalid"):
            container.create_extract_engine()

    def test_unsupported_evaluation_engine(self):
        settings = self.make_settings(eval_engine="invalid")
        container = DependencyContainer(settings)
        with pytest.raises(ConfigurationError, match="Unsupported engine: invalid"):
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
