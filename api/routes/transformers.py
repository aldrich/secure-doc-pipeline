import logging
import uuid

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException

from domain.auth import verify_api_key
from evaluation.evaluator import run_evaluation
from extraction.extractor import run_extraction
from schemas.session_request import SessionRequest

logger = logging.getLogger(__name__)

app = FastAPI(title="Transformers API")

@app.post("/process-session", status_code=202)
async def process_session(payload: SessionRequest, background_tasks: BackgroundTasks, _: None = Depends(verify_api_key)):

    session_id = uuid.uuid4().hex[:8]
    logger.info(f"Received session processing request. Assigned Job ID: {session_id}.")

    try:
        structured_output = run_extraction(payload.transcript)
        
        background_tasks.add_task(
            run_evaluation,
            structured_output,
            payload.transcript,
            session_id,            
        )
        
        return {
            "status": "processing_verification",
            "session_id": session_id,
            "data": structured_output
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline processing failed: {str(e)}")