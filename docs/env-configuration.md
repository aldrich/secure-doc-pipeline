# Environment Configuration Reference

This document describes every variable in `.env.example`, its purpose, valid values, defaults, dependencies, and behavioral notes.

---

## Engine Selection

### `EVAL_ENGINE`
| Attribute | Details |
|---|---|
| **Purpose** | Selects the LLM backend for **evaluation** (scoring extracted data against ground truth). |
| **Possible values** | `gemini`, `llama`, `openai` |
| **Default** | `llama` |
| **Required?** | No — default is used if absent. |
| **Where validated** | `evaluation/evaluator.py:202` — unknown values raise `ValueError`. |
| **Docker** | Passed as `- EVAL_ENGINE=${EVAL_ENGINE}` in `docker-compose.yml:8`. |

### `EXTRACT_ENGINE`
| Attribute | Details |
|---|---|
| **Purpose** | Selects the LLM backend for **extraction** (parsing clinical text into structured data). |
| **Possible values** | `gemini`, `llama`, `openai` |
| **Default** | `llama` |
| **Required?** | No — default is used if absent. |
| **Where validated** | `extraction/extractor.py:156` — unknown values raise `ConfigError`. |
| **Docker** | Passed as `- EXTRACT_ENGINE=${EXTRACT_ENGINE}` in `docker-compose.yml:9`. |

---

## Gemini (Google AI)

### `GEMINI_API_KEY`
| Attribute | Details |
|---|---|
| **Purpose** | API key for the Google Gemini API. Used by both `GeminiExtractor` and `GeminiEvaluator`. |
| **Possible values** | A valid Google AI Studio API key string. |
| **Default** | `""` (empty) |
| **Required?** | **Yes** when `EVAL_ENGINE=gemini` or `EXTRACT_ENGINE=gemini`. Startup will raise `RuntimeError` if empty. |
| **Dependency** | `google-genai>=2.10.0` |
| **Where validated** | `extraction/extractor.py:80-82`, `evaluation/evaluator.py:84-87`, `main.py:38` |
| **Docker** | `- GEMINI_API_KEY=${GEMINI_API_KEY}` in `docker-compose.yml:10`. |

### `GEMINI_MODEL_FOR_EXTRACTION`
| Attribute | Details |
|---|---|
| **Purpose** | Model identifier for Gemini extraction requests. |
| **Possible values** | Any Gemini model ID (e.g., `gemini-2.5-flash-lite`, `gemini-2.5-flash`, `gemini-2.0-pro`). |
| **Default** | `gemini-2.5-flash-lite` |
| **Required?** | **Yes** when `EXTRACT_ENGINE=gemini`. Startup will raise `RuntimeError` if empty. |
| **Where validated** | `extraction/extractor.py:75-78`, `main.py:39` |
| **Consumed by** | `gemini_client.models.generate_content(model=self.model, ...)` in `extraction/extractor.py:95` |
| **Docker** | `- GEMINI_MODEL_FOR_EXTRACTION=${GEMINI_MODEL_FOR_EXTRACTION}` in `docker-compose.yml:11`. |

### `GEMINI_MODEL_FOR_EVALUATION`
| Attribute | Details |
|---|---|
| **Purpose** | Model identifier for Gemini evaluation requests. |
| **Possible values** | Any Gemini model ID (e.g., `gemini-2.5-flash`, `gemini-2.5-pro`). |
| **Default** | `gemini-2.5-flash` |
| **Required?** | **Yes** when `EVAL_ENGINE=gemini`. Startup will raise `RuntimeError` if empty. |
| **Where validated** | `evaluation/evaluator.py:89-92`, `main.py:40` |
| **Consumed by** | `client.aio.models.generate_content(model=self.model, ...)` in `evaluation/evaluator.py:101` |
| **Docker** | `- GEMINI_MODEL_FOR_EVALUATION=${GEMINI_MODEL_FOR_EVALUATION}` in `docker-compose.yml:12`. |

---

## OpenAI

### `OPENAI_API_KEY`
| Attribute | Details |
|---|---|
| **Purpose** | API key for the OpenAI API. Used by both `OpenAIExtractor` and `OpenAIEvaluator`. |
| **Possible values** | A valid OpenAI API key string. |
| **Default** | `""` (empty) |
| **Required?** | **Yes** when `EVAL_ENGINE=openai` or `EXTRACT_ENGINE=openai`. Startup will raise `RuntimeError` if empty. |
| **Dependency** | `openai>=2.44.0` |
| **Where validated** | `extraction/extractor.py:39-42` and `extraction/extractor.py:51-54` (double-checked), `evaluation/evaluator.py:123-126`, `main.py:41` |
| **Docker** | `- OPENAI_API_KEY=${OPENAI_API_KEY}` in `docker-compose.yml:13`. |

### `OPENAI_MODEL_FOR_EXTRACTION`
| Attribute | Details |
|---|---|
| **Purpose** | Model identifier for OpenAI extraction requests. |
| **Possible values** | Any OpenAI chat model ID (e.g., `gpt-4.1-nano`, `gpt-4.1-mini`, `gpt-4o`). |
| **Default** | `gpt-4.1-nano` |
| **Required?** | **Yes** when `EXTRACT_ENGINE=openai`. Startup will raise `RuntimeError` if empty. |
| **Where validated** | `extraction/extractor.py:34-37`, `main.py:42` |
| **Consumed by** | `client.responses.parse(model=self.model, ...)` in `extraction/extractor.py:57` |
| **Docker** | `- OPENAI_MODEL_FOR_EXTRACTION=${OPENAI_MODEL_FOR_EXTRACTION}` in `docker-compose.yml:14`. |

### `OPENAI_MODEL_FOR_EVALUATION`
| Attribute | Details |
|---|---|
| **Purpose** | Model identifier for OpenAI evaluation requests. |
| **Possible values** | Any OpenAI chat model ID (e.g., `gpt-4.1-mini`, `gpt-4o`, `o3-mini`). |
| **Default** | `gpt-4.1-mini` |
| **Required?** | **Yes** when `EVAL_ENGINE=openai`. Startup will raise `RuntimeError` if empty. |
| **Where validated** | `evaluation/evaluator.py:128-131`, `main.py:43` |
| **Consumed by** | `client.responses.parse(model=self.model, ...)` in `evaluation/evaluator.py:141` |
| **Docker** | `- OPENAI_MODEL_FOR_EVALUATION=${OPENAI_MODEL_FOR_EVALUATION}` in `docker-compose.yml:15`. |

---

## Ollama (Local LLMs)

### `OLLAMA_HOST`
| Attribute | Details |
|---|---|
| **Purpose** | Base URL of the Ollama service. |
| **Possible values** | Any valid HTTP URL pointing to an Ollama instance (e.g., `http://localhost:11434`, `http://ollama-service:11434`). |
| **Default** | `http://localhost:11434` |
| **Required?** | Validated at startup as non-empty, but **not actually consumed by the Ollama client** — see note below. |
| **Dependency** | `ollama>=0.6.2` |
| **Where checked** | `main.py:47` — startup validation only. |
| **Docker** | Overridden to `http://ollama-service:11434` in `docker-compose.yml:16`. |

> **Known issue**: `OLLAMA_HOST` is validated at startup but never passed to `ollama.AsyncClient()` or `ollama.chat()` in either `extractor.py` or `evaluator.py`. The Ollama Python client hardcodes `http://localhost:11434` as its default. This means changing `OLLAMA_HOST` has **no runtime effect** beyond passing the startup check. To use a non-default host, the `host=` parameter would need to be wired through to `ollama.AsyncClient()`.

### `LLAMA_MODEL_FOR_EXTRACTION`
| Attribute | Details |
|---|---|
| **Purpose** | Ollama model tag for extraction requests. |
| **Possible values** | Any model tag available in the Ollama instance (e.g., `llama3.2:3b`, `llama3.2:1b`, `mistral:7b`). |
| **Default** | `llama3.2:3b` |
| **Required?** | **Yes** when `EXTRACT_ENGINE=llama`. Startup will raise `RuntimeError` if empty. |
| **Where validated** | `extraction/extractor.py:114-117`, `main.py:45` |
| **Consumed by** | `client.chat(model=self.model, ...)` in `extraction/extractor.py:126` |
| **Docker** | `- LLAMA_MODEL_FOR_EXTRACTION=${LLAMA_MODEL_FOR_EXTRACTION}` in `docker-compose.yml:17`. |

### `LLAMA_MODEL_FOR_EVALUATION`
| Attribute | Details |
|---|---|
| **Purpose** | Ollama model tag for evaluation requests. |
| **Possible values** | Any model tag available in the Ollama instance (e.g., `llama3.1:8b`, `llama3.2:3b`, `qwen2.5:7b`). |
| **Default** | `llama3.1:8b` |
| **Required?** | **Yes** when `EVAL_ENGINE=llama`. Startup will raise `RuntimeError` if empty. |
| **Where validated** | `evaluation/evaluator.py:160-163`, `main.py:46` |
| **Consumed by** | `client.chat(model=self.model, ...)` in `evaluation/evaluator.py:172` |
| **Docker** | `- LLAMA_MODEL_FOR_EVALUATION=${LLAMA_MODEL_FOR_EVALUATION}` in `docker-compose.yml:18`. |

---

## API Authentication

### `API_KEY`
| Attribute | Details |
|---|---|
| **Purpose** | Static bearer token for authenticating incoming HTTP requests. Expected in the `X-API-Key` request header. |
| **Possible values** | Any non-empty string. Should be a strong, secret value. |
| **Default** | `""` (empty) |
| **Required?** | **Yes**. Startup will raise `RuntimeError` if empty. Requests with missing or mismatched keys receive a `401` response. |
| **Where validated** | `main.py:44` (startup), `domain/auth.py:14-19` (per-request via `verify_api_key` dependency). |
| **Mechanism** | `fastapi.security.APIKeyHeader(name="X-API-Key")` extracts the header. `verify_api_key()` compares it against the env var. |
| **Routes protected** | `/process-session` via `Depends(verify_api_key)` in `api/routes/transformers.py:27`. |
| **Docker** | `- API_KEY=${API_KEY}` in `docker-compose.yml:19`. |

> **Note on loading inconsistencies**: This project uses two patterns to read environment variables:
> 1. **pydantic-settings** (`domain/settings.py`) — loads `.env` into a validated `Settings` object. Used by `main.py`, `extractor.py`, and `evaluator.py`.
> 2. **`os.getenv`** — `domain/auth.py:14` reads `API_KEY` directly via `os.getenv("API_KEY")`. This works because `pydantic-settings` calls `python-dotenv` under the hood, which populates `os.environ` from `.env`. But it is a fragile coupling — if the loading mechanism changes, `auth.py` could silently lose access.

---

## Dependency Summary

All runtime dependencies are managed via `pyproject.toml`:

| Dependency | Version | Used For |
|---|---|---|
| `pydantic-settings` | >=2.14.2 | Loading `.env` into `Settings` class |
| `google-genai` | >=2.10.0 | Gemini client (`genai.Client`) |
| `openai` | >=2.44.0 | OpenAI client (`AsyncOpenAI`) |
| `ollama` | >=0.6.2 | Ollama client (`ollama.AsyncClient`) |
| `python-dotenv` | >=1.2.2 | Pulled in by `pydantic-settings`; populates `os.environ` |

---

## Quick-Start Example

```bash
# Minimal local setup using Ollama (default engine)
OLLAMA_HOST=http://localhost:11434
LLAMA_MODEL_FOR_EXTRACTION=llama3.2:3b
LLAMA_MODEL_FOR_EVALUATION=llama3.1:8b
API_KEY=your-secret-api-key

# Or switch to OpenAI
EXTRACT_ENGINE=openai
EVAL_ENGINE=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL_FOR_EXTRACTION=gpt-4.1-nano
OPENAI_MODEL_FOR_EVALUATION=gpt-4.1-mini
API_KEY=your-secret-api-key
```

Unused provider variables can be left empty — they are only validated when their corresponding engine is selected.