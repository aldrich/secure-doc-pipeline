"""
Application entry point.

This module is responsible for:

- Loading environment variables from the project's `.env` file.
- Configuring application-wide logging.
- Creating the FastAPI application instance.
- Registering all API routers.

The application exposes versioned endpoints under `/api/v1`.
"""

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, Depends, FastAPI
from fastapi.responses import JSONResponse

from domain.auth import router as auth_router
from domain.auth import verify_api_key
from domain.container import DependencyContainer
from domain.dependencies import get_container
from domain.error import (
    AuthenticationError,
    ConfigurationError,
    EvaluationError,
    ExtractionError,
    ProviderError,
)
from domain.settings import settings, validate_settings
from domain.structured_logger import StructuredFormatter
from schemas.session_request import SessionRequest
from schemas.session_response import SessionResponse

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(StructuredFormatter())
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):

    container = DependencyContainer()
    app.state.container = container

    validate_settings(settings)

    from domain.database import init_db

    await init_db()

    yield

    from domain.database import close_db

    await close_db()


app = FastAPI(title="Secure Clinical Documentation Pipeline", lifespan=lifespan)

# Register authentication endpoints.
app.include_router(auth_router, prefix="/api/v1")


@app.post(
    "/api/v1/process-session",
    status_code=202,
    response_model=SessionResponse,
    dependencies=[Depends(verify_api_key)],
)
async def process_session(
    payload: SessionRequest,
    background_tasks: BackgroundTasks,
    container: DependencyContainer = Depends(get_container),
):
    return await container.pipeline_service.process_session(
        payload.transcript, background_tasks
    )

@app.exception_handler(ConfigurationError)
async def config_handler(request, exc):
    logger.exception("configuration_error", extra={"detail": str(exc)})
    return JSONResponse(
        status_code=500,
        content={"detail": "Configuration error"},
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
    logger.exception("generic_exception", extra={"detail": str(exc)})
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )
