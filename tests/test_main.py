import pytest

from domain.error import ConfigurationError
from domain.settings import Settings, settings, validate_settings

class TestValidateSettings:

    def _make_settings(self, **overrides) -> Settings:
        defaults = {
            "api_key": "key-123",
            "extract_engine": "gemini",
            "eval_engine": "gemini",
            "gemini_api_key": "g-key",
            "gemini_model_for_extraction": "g-model-extract",
            "gemini_model_for_evaluation": "g-model-eval",
            "openai_api_key": "o-key",
            "openai_model_for_extraction": "o-model-extract",
            "openai_model_for_evaluation": "o-model-eval",
            "deepseek_api_key": "d-key",
            "deepseek_model_for_extraction": "d-model-extract",
            "deepseek_model_for_evaluation": "d-model-eval",
            "llama_model_for_extraction": "l-model-extract",
            "llama_model_for_evaluation": "l-model-eval",
        }
        merged = {**defaults, **overrides}
        return Settings(**merged)

    def test_missing_api_key(self):
        s = self._make_settings(api_key="")
        with pytest.raises(ConfigurationError, match="api_key"):
            validate_settings(s)

    def test_missing_extract_engine(self):
        s = self._make_settings(extract_engine="")
        with pytest.raises(ConfigurationError, match="extract_engine"):
            validate_settings(s)

    def test_missing_eval_engine(self):
        s = self._make_settings(eval_engine="")
        with pytest.raises(ConfigurationError, match="eval_engine"):
            validate_settings(s)

    def test_missing_multiple_required(self):
        s = self._make_settings(api_key="", extract_engine="", eval_engine="")
        with pytest.raises(ConfigurationError, match="api_key, extract_engine, eval_engine"):
            validate_settings(s)

    def test_gemini_extract_missing_api_key(self):
        s = self._make_settings(extract_engine="gemini", gemini_api_key="")
        with pytest.raises(ConfigurationError, match="Gemini API key is required for extraction engine"):
            validate_settings(s)

    def test_gemini_extract_missing_model(self):
        s = self._make_settings(extract_engine="gemini", gemini_model_for_extraction="")
        with pytest.raises(ConfigurationError, match="Gemini model is required for extraction engine"):
            validate_settings(s)

    def test_gemini_eval_missing_api_key(self):
        s = self._make_settings(extract_engine="llama", eval_engine="gemini", gemini_api_key="")
        with pytest.raises(ConfigurationError, match="Gemini API key is required for evaluation engine"):
            validate_settings(s)

    def test_gemini_eval_missing_model(self):
        s = self._make_settings(eval_engine="gemini", gemini_model_for_evaluation="")
        with pytest.raises(ConfigurationError, match="Gemini model is required for evaluation engine"):
            validate_settings(s)

    def test_openai_extract_missing_api_key(self):
        s = self._make_settings(extract_engine="openai", openai_api_key="")
        with pytest.raises(ConfigurationError, match="OpenAI API key is required for extraction engine"):
            validate_settings(s)

    def test_openai_extract_missing_model(self):
        s = self._make_settings(extract_engine="openai", openai_model_for_extraction="")
        with pytest.raises(ConfigurationError, match="OpenAI model is required for extraction engine"):
            validate_settings(s)

    def test_openai_eval_missing_api_key(self):
        s = self._make_settings(eval_engine="openai", openai_api_key="")
        with pytest.raises(ConfigurationError, match="OpenAI API key is required for evaluation engine"):
            validate_settings(s)

    def test_openai_eval_missing_model(self):
        s = self._make_settings(eval_engine="openai", openai_model_for_evaluation="")
        with pytest.raises(ConfigurationError, match="OpenAI model is required for evaluation engine"):
            validate_settings(s)

    def test_deepseek_extract_missing_api_key(self):
        s = self._make_settings(extract_engine="deepseek", deepseek_api_key="")
        with pytest.raises(ConfigurationError, match="DeepSeek API key is required for extraction engine"):
            validate_settings(s)

    def test_deepseek_extract_missing_model(self):
        s = self._make_settings(extract_engine="deepseek", deepseek_model_for_extraction="")
        with pytest.raises(ConfigurationError, match="DeepSeek model is required for extraction engine"):
            validate_settings(s)

    def test_deepseek_eval_missing_api_key(self):
        s = self._make_settings(eval_engine="deepseek", deepseek_api_key="")
        with pytest.raises(ConfigurationError, match="DeepSeek API key is required for evaluation engine"):
            validate_settings(s)

    def test_deepseek_eval_missing_model(self):
        s = self._make_settings(eval_engine="deepseek", deepseek_model_for_evaluation="")
        with pytest.raises(ConfigurationError, match="DeepSeek model is required for evaluation engine"):
            validate_settings(s)

    def test_llama_extract_succeeds_without_api_key(self):
        s = self._make_settings(extract_engine="llama", eval_engine="gemini", llama_model_for_extraction="l-model")
        validate_settings(s)

    def test_llama_extract_missing_model(self):
        s = self._make_settings(extract_engine="llama", llama_model_for_extraction="")
        with pytest.raises(ConfigurationError, match="Llama model is required for extraction engine"):
            validate_settings(s)

    def test_llama_eval_missing_model(self):
        s = self._make_settings(eval_engine="llama", llama_model_for_evaluation="")
        with pytest.raises(ConfigurationError, match="Llama model is required for evaluation engine"):
            validate_settings(s)

    def test_inactive_engine_missing_fields_ignored(self):
        s = self._make_settings(
            extract_engine="gemini",
            eval_engine="gemini",
            openai_api_key="",
            openai_model_for_extraction="",
            openai_model_for_evaluation="",
            deepseek_api_key="",
            deepseek_model_for_extraction="",
            deepseek_model_for_evaluation="",
            llama_model_for_extraction="",
            llama_model_for_evaluation="",
        )
        validate_settings(s)

    def test_all_settings_present_succeeds(self):
        s = self._make_settings()
        validate_settings(s)

    def test_different_engines_for_extract_and_eval(self):
        s = self._make_settings(
            extract_engine="openai",
            eval_engine="deepseek",
        )
        validate_settings(s)


class TestMainApp:
    def test_app_title(self):
        from main import app
        assert app.title == "Secure Clinical Documentation Pipeline"

    def test_routers_registered(self):
        from main import app
        paths = []
        for route in app.routes:
            if hasattr(route, "path"):
                paths.append(route.path)
            elif hasattr(route, "original_router"):
                prefix = route.include_context.prefix
                for r in route.original_router.routes:
                    if hasattr(r, "path"):
                        paths.append(prefix + r.path)
        assert any("/api/v1" in path for path in paths)

    @pytest.mark.asyncio
    async def test_lifespan_raises_on_missing_settings(self, monkeypatch):
        monkeypatch.setattr(settings, "gemini_api_key", "")
        monkeypatch.setattr(settings, "gemini_model_for_extraction", "")
        monkeypatch.setattr(settings, "openai_api_key", "")
        monkeypatch.setattr(settings, "openai_model_for_extraction", "")
        monkeypatch.setattr(settings, "api_key", "")
        monkeypatch.setattr(settings, "llama_model_for_extraction", "")
        monkeypatch.setattr(settings, "llama_model_for_evaluation", "")
        monkeypatch.setattr(settings, "ollama_host", "")
        monkeypatch.setattr(settings, "deepseek_api_key", "")
        monkeypatch.setattr(settings, "deepseek_model_for_extraction", "")
        monkeypatch.setattr(settings, "deepseek_model_for_evaluation", "")
        monkeypatch.setattr(settings, "deepseek_base_url", "")
        monkeypatch.setattr(settings, "extract_engine", "")
        monkeypatch.setattr(settings, "eval_engine", "")
        monkeypatch.setattr(settings, "gemini_model_for_evaluation", "")
        monkeypatch.setattr(settings, "openai_model_for_evaluation", "")

        from main import lifespan, app

        with pytest.raises(ConfigurationError, match="Required settings are missing"):
            async with lifespan(app):
                pass

    @pytest.mark.asyncio
    async def test_lifespan_succeeds_when_all_settings_present(self, monkeypatch):
        for attr in [
            "gemini_api_key", "gemini_model_for_extraction", "gemini_model_for_evaluation",
            "openai_api_key", "openai_model_for_extraction", "openai_model_for_evaluation",
            "api_key",
            "llama_model_for_extraction", "llama_model_for_evaluation",
            "ollama_host",
            "deepseek_api_key", "deepseek_model_for_extraction", "deepseek_model_for_evaluation",
            "deepseek_base_url",
            "extract_engine", "eval_engine",
        ]:
            monkeypatch.setattr(settings, attr, "non-empty")

        from main import lifespan, app

        async with lifespan(app):
            assert app.state.container is not None