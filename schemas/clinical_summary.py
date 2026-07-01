from typing import List, Self

import ollama
from pydantic import BaseModel, Field

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
        response = ollama.chat(
            model='llama3.2:3b',
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