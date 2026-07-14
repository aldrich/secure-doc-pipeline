from pydantic_settings import BaseSettings
from domain.error import ConfigurationError

ENGINE_REQUIREMENTS: dict[str, tuple[str, str | None, str, str]] = {
    "gemini":   ("Gemini",   "gemini_api_key",   "gemini_model_for_extraction",   "gemini_model_for_evaluation"),
    "openai":   ("OpenAI",   "openai_api_key",   "openai_model_for_extraction",   "openai_model_for_evaluation"),
    "deepseek": ("DeepSeek", "deepseek_api_key", "deepseek_model_for_extraction", "deepseek_model_for_evaluation"),
    "llama":    ("Llama",    None,               "llama_model_for_extraction",    "llama_model_for_evaluation"),
}

class Settings(BaseSettings):
    extract_engine: str = "llama"
    eval_engine: str = "llama"
    
    gemini_api_key: str = ""
    gemini_model_for_extraction: str = "gemini-2.5-flash-lite"
    gemini_model_for_evaluation: str = "gemini-2.5-flash"
    
    openai_api_key: str = ""
    openai_model_for_extraction: str = "gpt-4.1-nano"
    openai_model_for_evaluation: str = "gpt-4.1-mini"
    
    deepseek_api_key: str = ""
    deepseek_model_for_extraction: str = "deepseek-v4-flash"
    deepseek_model_for_evaluation: str = "deepseek-v4-pro"
    deepseek_base_url: str = "https://api.deepseek.com"

    ollama_host: str = "http://localhost:11434"
    llama_model_for_extraction: str = "llama3.2:3b"
    llama_model_for_evaluation: str = "llama3.1:8b"

    llm_max_retries: int = 3
    llm_retry_base_delay: float = 1.0
    llm_retry_max_delay: float = 30.0
    llm_timeout: int = 120
    
    api_key: str = ""   

    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5433/clinical_pipeline"
    
    model_config = { "env_file": ".env", "env_file_encoding": "utf-8" }
    
settings = Settings()

def _validate_engine(
    settings: "Settings",
    engine_name: str,
    display_name: str,
    role: str,
    api_key_field: str | None,
    model_field: str,
    active_engine: str,
):
    if active_engine != engine_name:
        return
    if api_key_field is not None and not getattr(settings, api_key_field):
        raise ConfigurationError(
            f"{display_name} API key is required for {role} engine"
        )
    if not getattr(settings, model_field):
        raise ConfigurationError(
            f"{display_name} model is required for {role} engine"
        )


def validate_settings(settings: "Settings"):
    missing = [
        name for name, value in [
            ("api_key", settings.api_key),
            ("extract_engine", settings.extract_engine),
            ("eval_engine", settings.eval_engine),
        ]
        if not value
    ]
    if missing:
        raise ConfigurationError(
            "Required settings are missing or empty: "
            + ", ".join(missing)
        )

    for engine_name, (display_name, api_key_field, extract_model_field, eval_model_field) in ENGINE_REQUIREMENTS.items():
        _validate_engine(settings, engine_name, display_name, "extraction", api_key_field, extract_model_field, settings.extract_engine)
        _validate_engine(settings, engine_name, display_name, "evaluation", api_key_field, eval_model_field, settings.eval_engine)