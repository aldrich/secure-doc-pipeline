"""
Application entry point.

This module is responsible for:

- Loading environment variables from the project's `.env` file.
- Configuring application-wide logging.
- Creating the FastAPI application instance.
- Registering all API routers.

The application exposes versioned endpoints under `/api/v1`.
"""

import sys
import logging

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from domain.auth import app as auth_app
from api.routes.transformers import app as transformers_app
from domain.error import AuthenticationError, ConfigurationError, EvaluationError, ExtractionError, ProviderError
from domain.settings import settings

# Configure the application's root logger to emit INFO-level
# messages to standard output using a concise format.
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="[%(levelname)s] %(message)s",
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    
    # all these settings are required to be non-null.
    missing = [
        name for name, value in [
            ("gemini_api_key", settings.gemini_api_key),
            ("gemini_model_for_extraction", settings.gemini_model_for_extraction),
            ("gemini_model_for_evaluation", settings.gemini_model_for_evaluation),
            ("openai_api_key", settings.openai_api_key),
            ("openai_model_for_extraction", settings.openai_model_for_extraction),
            ("openai_model_for_evaluation", settings.openai_model_for_evaluation),
            ("api_key", settings.api_key),
            ("llama_model_for_extraction", settings.llama_model_for_extraction),
            ("llama_model_for_evaluation", settings.llama_model_for_evaluation),
            ("ollama_host", settings.ollama_host),
        ]
        if not value
    ]
    if missing:
        raise ConfigurationError(
            "Required settings are missing or empty: "
            + ", ".join(missing)
        )
    yield

app = FastAPI(
    title="Secure Clinical Documentation Pipeline", lifespan=lifespan
)

# Register authentication endpoints.
app.include_router(auth_app.router, prefix="/api/v1")

# Register document transformation endpoints.
app.include_router(transformers_app.router, prefix="/api/v1")


@app.exception_handler(ConfigurationError)
async def config_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )
    
@app.exception_handler(AuthenticationError)
async def auth_handler(request, exc):
    return JSONResponse(
        status_code=401,
        content={"detail": str(exc)},
    )
    
@app.exception_handler(ProviderError)
async def provider_handler(request, exc):
    return JSONResponse(
        status_code=502,
        content={"detail": str(exc)},
    )
    
@app.exception_handler(ExtractionError)
async def extraction_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc)},
    )
    
@app.exception_handler(EvaluationError)
async def evaluation_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc)},
    )

@app.exception_handler(Exception)
async def generic_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )