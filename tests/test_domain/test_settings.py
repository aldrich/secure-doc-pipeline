from domain.settings import Settings


class TestSettings:
    def test_default_values(self, monkeypatch):
        monkeypatch.delenv("EXTRACT_ENGINE", raising=False)
        monkeypatch.delenv("EVAL_ENGINE", raising=False)
        monkeypatch.delenv("API_KEY", raising=False)
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        settings = Settings(_env_file=None)
        assert settings.extract_engine == "llama"
        assert settings.eval_engine == "llama"
        assert settings.ollama_host == "http://localhost:11434"
        assert settings.api_key == ""

    def test_env_var_overrides(self, monkeypatch):
        monkeypatch.setenv("EXTRACT_ENGINE", "openai")
        monkeypatch.setenv("EVAL_ENGINE", "gemini")
        monkeypatch.setenv("API_KEY", "test-key-123")
        settings = Settings()
        assert settings.extract_engine == "openai"
        assert settings.eval_engine == "gemini"
        assert settings.api_key == "test-key-123"

    def test_api_key_defaults_to_empty(self, monkeypatch):
        monkeypatch.delenv("API_KEY", raising=False)
        settings = Settings(_env_file=None)
        assert settings.api_key == ""