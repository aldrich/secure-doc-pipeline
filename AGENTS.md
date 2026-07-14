# Secure Clinical Documentation Pipeline

## Quick commands

| Action | Command |
|---|---|
| Install deps | `uv sync` |
| Dev server | `uv run poe server` (or `uv run uvicorn main:app --reload`) |
| Run all tests | `uv run pytest` |
| Run single test file | `uv run pytest tests/test_extraction/` |
| Run single test | `uv run pytest tests/test_extraction/test_extraction_engine.py::TestExtractionEngine::test_extract_success -v` |
| Run retry tests | `uv run pytest tests/test_domain/test_retry.py -v` |
| Run extractor retry tests | `uv run pytest tests/test_extraction/test_extractor_retries.py -v` |

## Key patterns

- **Package manager**: `uv` only (no pip/poetry). Python 3.14+.
- **Testing**: pytest with `asyncio_mode = "auto"` (async tests need `@pytest.mark.asyncio`). Use `mocker` fixture from `pytest-mock` for patching.
- **No lint/typecheck config**: no ruff, mypy, black etc. configured.
- **Config**: `domain/settings.py` — pydantic-settings loads `.env` automatically. Copy `.env.example` first. Validation in `validate_settings()` called on startup.
- **Auth**: `X-API-Key` header, validated in `domain/auth.py` via `os.getenv("API_KEY")` (not pydantic-settings).
- **Logging**: JSON structured via `StructuredFormatter`. Pass structured fields via `extra={}` dict.
- **Docker**: `docker compose up` starts FastAPI + an Ollama container.
- **PostgreSQL**: SQLAlchemy async with asyncpg. `domain/database.py` creates engine from `DATABASE_URL`. Schema auto-created via `Base.metadata.create_all` on startup. No Alembic migrations yet.
- **Retry**: `domain/retry.py` provides `with_llm_retry()` decorator — exponential backoff with jitter, configurable via env vars. Used by `LLMEngine.__init__`.

## Architecture

- Entrypoint: `main.py` (FastAPI app, routes defined inline — NOT in `api/routes/`)
- DI container: `domain/container.py` — lazy-init singletons per engine type
- Pipeline: `extraction` → `evaluation` (eval runs as `BackgroundTasks`)
- The `api/routes/` dir exists but is empty; the README reference to `transformers.py` is stale.
- Orchestration: `services/pipeline.py` — `PipelineService` ties extraction, evaluation, and persistence together
- Persistence: Evaluation results stored in PostgreSQL via `domain/repository.py` and `domain/models.py`

## Engine injection pattern

Two-layer architecture: **Client** handles API communication, **Engine** handles business logic.

### Client layer (`LLMClient` subclasses)

| Client | Constructor signature |
|---|---|
| `OpenAIClient` | `(api_key, timeout=120)` |
| `GeminiClient` | `(api_key, timeout=120)` |
| `DeepSeekClient` | `(api_key, base_url, timeout=120)` |
| `OllamaClient` | `(ollama_host, timeout=120)` |

### Engine layer (`LLMEngine` subclasses)

`LLMEngine.__init__(client, model, max_retries=3, retry_base_delay=1.0, retry_max_delay=30.0)` — inherited by `ExtractionEngine` and `EvaluationEngine`.

When adding a new engine or modifying constructor params, update `domain/container.py`.

## Engine quirks

- **OpenAI**: uses `client.responses.parse()` (beta Responses API), NOT `chat.completions.create`.
- **DeepSeek**: uses `AsyncOpenAI` client with custom `base_url`. Appends JSON schema to system prompt + uses `response_format={"type": "json_object"}`.
- **Gemini**: uses `client.aio.models.generate_content()` (both extraction and evaluation are async).
- **Llama**: uses `ollama.AsyncClient` — `host` is injected via constructor.