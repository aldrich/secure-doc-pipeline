from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    extract_engine: str = "llama"
    eval_engine: str = "llama"
    
    gemini_api_key: str = ""
    gemini_model_for_extraction: str = "gemini-2.5-flash-lite"
    gemini_model_for_evaluation: str = "gemini-2.5-flash"
    
    openai_api_key: str = ""
    openai_model_for_extraction: str = "gpt-4.1-nano"
    openai_model_for_evaluation: str = "gpt-4.1-mini"
    
    ollama_host: str = "http://localhost:11434"
    llama_model_for_extraction: str = "llama3.2:3b"
    llama_model_for_evaluation: str = "llama3.1:8b"
    
    api_key: str = ""
    
    
    model_config = { "env_file": ".env", "env_file_encoding": "utf-8" }
    
settings = Settings()