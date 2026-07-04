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

from dotenv import load_dotenv
from fastapi import FastAPI

from api.routes.transformers import app as transformers_app
from domain.auth import app as auth_app


# Load environment variables before any application components
# attempt to access configuration values.
load_dotenv()


# Configure the application's root logger to emit INFO-level
# messages to standard output using a concise format.
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="[%(levelname)s] %(message)s",
)


#: Root FastAPI application instance.
app = FastAPI(title="Secure Clinical Documentation Pipeline")


# Register authentication endpoints.
app.include_router(auth_app.router, prefix="/api/v1")


# Register document transformation endpoints.
app.include_router(transformers_app.router, prefix="/api/v1")
