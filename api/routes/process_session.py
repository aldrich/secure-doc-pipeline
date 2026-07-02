import sys, logging
import uuid
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from evaluation.evaluator import run_evaluation
from schemas.clinical_summary import ClinicalSummary

load_dotenv()

logging.basicConfig(level=logging.INFO,
                    stream=sys.stdout,
                    format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Secure Clinical Documentation Pipeline")

# Input contract for incoming requests to our API
class SessionRequest(BaseModel):
    transcript: str

@app.post("/api/v1/process-session", status_code=202)
async def process_session(payload: SessionRequest, background_tasks: BackgroundTasks):

    session_id = uuid.uuid4().hex[:8]
    logger.info(f"Received session processing request. Assigned Job ID: {session_id}.")

    try:
        structured_output = ClinicalSummary.extract_structured_data_openai(payload.transcript)

        background_tasks.add_task(
            run_evaluation,
            structured_output,
            payload.transcript,
            session_id,            
        )

        # TODO: Write to database / trigger webhook back to n8n here...
        
        return {
            "status": "processing_verification",
            "session_id": session_id,
            "data": structured_output
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline processing failed: {str(e)}")

async def main():
    
    source_transcript="""
    Patient initially stated, "I didn't do any physical therapy or movement work over the weekend at all." 
    However, later in the review, when prompted about specific logs, they recalled and corrected themselves: 
    "Oh, wait, I actually spent about 20 minutes doing my balance and gait exercises on Saturday afternoon 
    with the home nurse." They reported feeling stable throughout.
    """
    
    sample_summary = ClinicalSummary(
        patient_mood="obvious",
        exercises_completed=["balance", "gait"],
        symptoms_mentioned=[],
        next_steps="schedule for 60 mins of strength exercises"
    )

    logger.info(f"Triggering evaluation pipeline...")

    await run_evaluation(sample_summary, source_transcript)

if __name__ == "__main__":
    asyncio.run(main())