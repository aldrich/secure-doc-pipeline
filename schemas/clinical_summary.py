import logging
from typing import List

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class ClinicalSummary(BaseModel):
    
    patient_mood: str = Field(
        description="The emotional state of the patient during the session."
    )
    
    exercises_completed: List[str] = Field(
        description="List of specific physical or cognitive exercises performed."
    )
    
    symptoms_mentioned: List[str] = Field(
        description="Any symptoms, pains, or cognitive difficulties the patient complained about."
    )
    
    next_steps: str = Field(
        description="The plan or homework for the next session."
    )