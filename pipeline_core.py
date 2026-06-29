import json
import logging
import sys
from typing import List
from pydantic import BaseModel, Field
import ollama

logging.basicConfig(level=logging.INFO,
                    stream=sys.stdout,
                    format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# 1. Define the strict data structure you want the AI to return
# class ClinicalSummary1(BaseModel):
#     patient_mood: str = Field(description="The emotional state of the patient during the session.")
#     exercises_completed: List[str] = Field(description="List of specific physical or cognitive exercises performed.")
#     symptoms_mentioned: List[str] = Field(description="Any symptoms, pains, or cognitive difficulties the patient complained about.")
#     next_steps: str = Field(description="The plan or homework for the next session.")

class ClinicalSummary(BaseModel):
    patient_mood: str = Field(description="The emotional state of the patient during the session.")
    exercises_completed: List[str] = Field(description="List of specific physical or cognitive exercises performed.")
    symptoms_mentioned: List[str] = Field(description="Any symptoms, pains, or cognitive difficulties the patient complained about.")
    next_steps: str = Field(description="The plan or homework for the next session.")


# 2. Mock a messy, real-world conversational transcript
raw_transcript = """
The patient arrived a bit anxious today but warmed up during the cognitive tasks. 
We spent the first twenty minutes working on the spatial memory grid exercises and then 
moved on to vocal articulation drills. She mentioned experiencing some mild headaches 
over the weekend and a bit of frustration with word-finding when fatigued. 
For next time, I told her to practice the articulation sheet twice a day and keep track 
of when the headaches occur.
"""

def extract_structured_data(transcript: str) -> ClinicalSummary:
    # We pass the schema directly to Ollama to force a structured JSON output
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
        # This parameter forces the model to output valid JSON matching our Pydantic schema
        format=ClinicalSummary.model_json_schema()
    )
    
    # Parse the raw string response back into our Pydantic object
    raw_content = response['message']['content']
    structured_data = ClinicalSummary.model_validate_json(raw_content)
    return structured_data

# 3. Run the extraction pipeline
if __name__ == "__main__":
    logger.info("Parsing transcript using local LLM...")
    try:
        result = extract_structured_data(raw_transcript)
        logger.info("Successfully structured the messy data!")
        
        # Print it as a pretty JSON string to see the structural formatting
        logger.info(json.dumps(result.model_dump(), indent=2))
        
    except Exception as e:
        logger.info(f"\n❌ Pipeline failed: {e}")