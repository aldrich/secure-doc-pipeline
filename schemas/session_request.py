# Input contract for incoming requests to our API
from pydantic import BaseModel

class SessionRequest(BaseModel):
    transcript: str