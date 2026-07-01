import json
import logging, sys
from typing import List
from pydantic import BaseModel, Field
import ollama

logger = logging.getLogger(__name__)
class ClinicalSummary(BaseModel):
    patient_mood: str = Field(description="The emotional state of the patient during the session.")
    exercises_completed: List[str] = Field(description="List of specific physical or cognitive exercises performed.")
    symptoms_mentioned: List[str] = Field(description="Any symptoms, pains, or cognitive difficulties the patient complained about.")
    next_steps: str = Field(description="The plan or homework for the next session.")

raw_transcript = """
The patient presented as highly energetic but struggled significantly with focused attention early in the session. We dedicated fifteen minutes to the abstract pattern-matching blocks, where he demonstrated a tendency to rush through errors, followed by a series of trail-making sequencing tasks. He reported sleeping poorly over the past three days and noted that a persistent ringing in his left ear has made concentrating difficult in noisy environments. For our next appointment, I instructed him to complete the level-two sequence sheets in a quiet room and log his daily sleep duration.
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

def main():
    
    logging.basicConfig(level=logging.INFO,
                    stream=sys.stdout,
                    format='[%(levelname)s] %(message)s')
    
    logger.info("Parsing transcript using local LLM...")
    try:
        result = extract_structured_data(raw_transcript)
        logger.info("Successfully structured the messy data!")
        
        # Print it as a pretty JSON string to see the structural formatting
        logger.info(json.dumps(result.model_dump(), indent=2))
        
    except Exception as e:
        logger.info(f"\n❌ Pipeline failed: {e}")

if __name__ == "__main__":
    main()