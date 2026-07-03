import os

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import APIKeyHeader

from domain.config_error import ConfigError

load_dotenv()

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

app = FastAPI(title='Auth API')

async def verify_api_key(api_key: str = Depends(api_key_header)):

    VALID_API_KEY=os.getenv("API_KEY")
    if VALID_API_KEY is None:
        raise ConfigError("VALID_API_KEY not configured!")
    
    if api_key != VALID_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")