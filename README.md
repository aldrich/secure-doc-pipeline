# Secure Clinical Documentation Pipeline

A FastAPI-based pipeline for extracting structured clinical summaries from therapy session transcripts and evaluating their factual faithfulness.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        API Layer                                 │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │         POST /api/v1/process-session                       │  │
│  │                    (202 Accepted)                          │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────────┐
│                    Processing Pipeline                            │
│                                                                   │
│  ┌─────────────────────┐         ┌──────────────────────┐         │
│  │   Extraction        │         │   Evaluation         │         │
│  │   (Async)           │────────▶│   (Background)       │         │
│  │                     │         │                      │         │
│  │ • Extract           │         │ • Score faithfulness │         │
│  │ • Structured        │         │ • Detect             │         │
│  │   data              │         │   hallucinations     │         │
│  │   (ClinicalSummary) │         │ • Find               │         │
│  │                     │         │   omissions          │         │
│  │                     │         │ • Identify           │         │
│  │                     │         │   contradictions     │         │
│  └─────────────────────┘         └──────────────────────┘         │
└───────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Supported Engines                             │
│                                                                  │
│  OpenAI    │ Gemini    │ DeepSeek │ Ollama (Llama)               │
│  gpt-4.1   │ gemini-2  │ deepseek │ llama3.x                     │
│  nano/mini │ .5-flash  │ v4-flash │                              │
└──────────────────────────────────────────────────────────────────┘
```

## Setup

### Prerequisites

- Python 3.14+ (managed via `uv`)
- One of the following:
  - OpenAI API key
  - Google Gemini API key
  - DeepSeek API key
  - Ollama running locally (for Llama models)

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd secure-doc-pipeline

# Install dependencies using uv
uv sync
```

### Environment Configuration

Copy the example environment file and configure your settings:

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

```bash
# Engine selection (default: llama)
EXTRACT_ENGINE=llama
EVAL_ENGINE=llama

# Ollama configuration (for local LLMs)
OLLAMA_HOST=http://localhost:11434
LLAMA_MODEL_FOR_EXTRACTION=llama3.2:3b
LLAMA_MODEL_FOR_EVALUATION=llama3.1:8b

# API Authentication
API_KEY=your-secret-api-key
```

## Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `EXTRACT_ENGINE` | Backend for extraction: `gemini`, `llama`, `openai`, `deepseek` | `llama` | Yes (has default) |
| `EVAL_ENGINE` | Backend for evaluation: `gemini`, `llama`, `openai`, `deepseek` | `llama` | Yes (has default) |
| `API_KEY` | Bearer token for request authentication | - | Yes |
| `OLLAMA_HOST` | Ollama server URL | `http://localhost:11434` | When using llama engine |
| `LLAMA_MODEL_FOR_EXTRACTION` | Ollama model for extraction | `llama3.2:3b` | When using llama engine |
| `LLAMA_MODEL_FOR_EVALUATION` | Ollama model for evaluation | `llama3.1:8b` | When using llama engine |
| `GEMINI_API_KEY` | Google Gemini API key | - | When using gemini engine |
| `GEMINI_MODEL_FOR_EXTRACTION` | Gemini model for extraction | `gemini-2.5-flash-lite` | When using gemini engine |
| `GEMINI_MODEL_FOR_EVALUATION` | Gemini model for evaluation | `gemini-2.5-flash` | When using gemini engine |
| `OPENAI_API_KEY` | OpenAI API key | - | When using openai engine |
| `OPENAI_MODEL_FOR_EXTRACTION` | OpenAI model for extraction | `gpt-4.1-nano` | When using openai engine |
| `OPENAI_MODEL_FOR_EVALUATION` | OpenAI model for evaluation | `gpt-4.1-mini` | When using openai engine |
| `DEEPSEEK_API_KEY` | DeepSeek API key | - | When using deepseek engine |
| `DEEPSEEK_MODEL_FOR_EXTRACTION` | DeepSeek model for extraction | `deepseek-v4-flash` | When using deepseek engine |
| `DEEPSEEK_MODEL_FOR_EVALUATION` | DeepSeek model for evaluation | `deepseek-v4-pro` | When using deepseek engine |
| `DEEPSEEK_BASE_URL` | DeepSeek API base URL | `https://api.deepseek.com` | When using deepseek engine |

## Running

### Development Server

```bash
uv run poe server
```

Or directly with uvicorn:

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Docker

```bash
docker compose up
```

## API Usage

### POST /api/v1/process-session

Process a session transcript and get structured output.

**Request:**

```bash
curl -X POST http://localhost:8000/api/v1/process-session \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-api-key" \
  -d '{
    "transcript": "Patient reported feeling anxious during today session. Completed 15 minutes of balance exercises. Reports mild knee pain. Next steps: schedule follow-up in 2 weeks."
  }'
```

**Response (202 Accepted):**

```json
{
  "status": "processing_verification",
  "session_id": "abc12345",
  "data": {
    "patient_mood": "anxious",
    "exercises_completed": ["balance exercises"],
    "symptoms_mentioned": ["mild knee pain"],
    "next_steps": "schedule follow-up in 2 weeks"
  }
}
```

### Request Schema

```json
{
  "transcript": "string (1-5000 characters)"
}
```

### Response Schema

```json
{
  "status": "processing_verification",
  "session_id": "string (8 hex characters)",
  "data": {
    "patient_mood": "string",
    "exercises_completed": ["string"],
    "symptoms_mentioned": ["string"],
    "next_steps": "string"
  }
}
```

## Output Schema: ClinicalSummary

| Field | Type | Description |
|-------|------|-------------|
| `patient_mood` | string | The emotional state of the patient during the session |
| `exercises_completed` | string[] | List of specific physical or cognitive exercises performed |
| `symptoms_mentioned` | string[] | Any symptoms, pains, or cognitive difficulties complained about |
| `next_steps` | string | The plan or homework for the next session |

## Evaluation Metrics

The evaluation engine returns a `SummaryEvaluation` with:

| Field | Type | Description |
|-------|------|-------------|
| `faithful` | boolean | Whether the summary is factually grounded |
| `score` | float (0-1) | Faithfulness score (1.0 = perfect) |
| `unsupported_claims` | array | Claims not supported by transcript |
| `omitted_information` | array | Important facts omitted from summary |
| `contradictions` | array | Statements contradicting the transcript |
| `reasoning` | string | Detailed evaluation rationale |

## Testing

```bash
uv run pytest
```

## Project Structure

```
├── main.py                    # Application entry point
├── pyproject.toml             # Project configuration
├── Dockerfile                 # Container build
├── docker-compose.yml         # Docker orchestration
├── .env.example               # Environment template
├── docs/
│   └── env-configuration.md   # Detailed env var reference
├── domain/
│   ├── settings.py            # Configuration management
│   ├── container.py           # Dependency injection
│   ├── auth.py                # API key authentication
│   ├── error.py               # Custom exceptions
│   └── structured_logger.py   # JSON logging formatter
├── api/
│   └── routes/                    # Empty; routes defined inline in main.py
├── extraction/
│   ├── extraction_engine.py   # LLM extraction backends
│   └── extractor.py           # Unified extraction interface
├── evaluation/
│   ├── evaluation_engine.py   # LLM evaluation backends
│   └── evaluator.py           # Unified evaluation interface
├── schemas/
│   ├── clinical_summary.py    # Output schema
│   ├── session_request.py     # Input schema
│   └── session_response.py    # Response schema
└── prompts/
    ├── extraction.py          # Extraction system prompt
    └── evaluation.py          # Evaluation system prompt
```

## Supported Engines

| Engine | Extraction | Evaluation | Dependencies |
|--------|------------|------------|--------------|
| OpenAI | ✓ | ✓ | `openai>=2.44.0` |
| Gemini | ✓ | ✓ | `google-genai>=2.10.0` |
| DeepSeek | ✓ | ✓ | `openai>=2.44.0` (via compatible API) |
| Ollama/Llama | ✓ | ✓ | `ollama>=0.6.2` |

## License

MIT