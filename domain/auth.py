import os

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

app = FastAPI(title='Auth API')

async def verify_api_key(api_key: str = Depends(api_key_header)):

    valid_api_key=os.getenv("API_KEY")
    if valid_api_key is None or api_key != valid_api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")