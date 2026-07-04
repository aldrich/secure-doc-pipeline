# Input contract for incoming requests to our API
from pydantic import BaseModel, Field

class SessionRequest(BaseModel):
    transcript: str = Field(
        description='transcript recorded from a session',
        min_length=1,
        max_length=5000,
    )