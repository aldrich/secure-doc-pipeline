import secrets

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import APIKeyHeader

from domain.settings import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

router = APIRouter(tags=["auth"])


async def verify_api_key(api_key: str = Depends(api_key_header)):
    valid_api_key = settings.api_key
    if not valid_api_key or not secrets.compare_digest(api_key, valid_api_key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
