import logging
import os
from pprint import pprint
from typing import List, Self

import ollama
from pydantic import BaseModel, Field

from domain.config_error import ConfigError

logger = logging.getLogger(__name__)
llama_model_for_extraction = os.environ.get("LLAMA_MODEL_FOR_TRANSCRIPT_EXTRACTION")

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
    
    @classmethod
    def extract_structured_data(cls, transcript: str) -> Self:
        
        if llama_model_for_extraction is None:
            message = 'LLAMA_MODEL_FOR_TRANSCRIPT_EXTRACTION not found in .env'
            logger.error(message)
            raise ConfigError(message)
        
        response = ollama.chat(
            model=llama_model_for_extraction,
            messages=[
                {
                    'role': 'system',
                    'content': (
                        "You are a clinical automation assistant. Extract the relevant information "
                        "from the transcript into the requested JSON schema. Be concise and factual. "
                        "Only extract metrics, tasks, and symptoms experienced directly by the patient. "
                        "Do not extract or attribute actions or complaints made by spouses, family "
                        "members, or third parties."
                    )
                },
                {
                    'role': 'user',
                    'content': f"Transcript to parse:\n{transcript}"
                }
            ],
            format=ClinicalSummary.model_json_schema()
        )
        
        raw_content = response['message']['content']        
        clean_data = ClinicalSummary.model_validate_json(raw_content)
        
        return cls(
            patient_mood=clean_data.patient_mood,
            exercises_completed=clean_data.exercises_completed,
            symptoms_mentioned=clean_data.symptoms_mentioned,
            next_steps=clean_data.next_steps,
        )
        
def main():
    
    transcript='''The patient was cooperative but noticeably fatigued today, requiring frequent verbal prompts
    to remain engaged. We spent the bulk of the session conducting auditory numeric-recall sequences and
    short-term semantic association drills. She expressed deep frustration with remembering short lists of
    chores around the house, which she claims is causing tension with her family. I advised her to utilize the
    digital voice-recorder tool for her daily tasks at least three times a week and requested that her spouse
    attend the next session to discuss collaborative strategies.
    '''
    structured_output = ClinicalSummary.extract_structured_data(transcript)
    
    pprint(structured_output)

if __name__ == "__main__":
    main()