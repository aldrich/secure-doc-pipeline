from fastapi.testclient import TestClient
import pytest

from domain.error import ConfigurationError
from domain.settings import settings


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