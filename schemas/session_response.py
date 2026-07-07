from pydantic import BaseModel

from schemas.clinical_summary import ClinicalSummary


class SessionResponse(BaseModel):
    status: str
    session_id: str
    data: ClinicalSummary