"""
Transformers API endpoint file.

This module provides a FastAPI endpoint for processing session transcripts
using a transformer-based pipeline. It handles the extraction of structured
data and triggers an asynchronous evaluation process.
"""

import logging
import uuid

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request

from domain.auth import verify_api_key
from domain.error import EvaluationError
from evaluation.evaluator import run_evaluation
from extraction.extractor import run_extraction
from schemas.session_request import SessionRequest
from schemas.session_response import SessionResponse

logger = logging.getLogger(__name__)

app = FastAPI(title="Transformers API")

@app.post("/process-session", status_code=202, response_model=SessionResponse)
async def process_session(
    payload: SessionRequest,
    background_tasks: BackgroundTasks,
    request: Request,
    _: None = Depends(verify_api_key)
):
    """
    Process a session request using the transformers pipeline.

    This endpoint accepts a session transcript, performs an immediate
    extraction of structured data, and queues a background task to
    evaluate the quality of the extraction.

    Args:
        payload (SessionRequest): The session data containing the transcript to be processed.
        background_tasks (BackgroundTasks): FastAPI utility for running the evaluation
            process in the background after the response is sent.
        _ (None): Dependency injection for API key verification.

    Returns:
        dict: A dictionary containing the status, a unique session_id,
            and the extracted data.
    """

    container = request.app.state.container

    session_id = uuid.uuid4().hex[:8]
    logger.info("session_processing_request", extra={"session_id": session_id})

    structured_output = await run_extraction(payload.transcript, session_id, container)

    background_tasks.add_task(
        run_evaluation,
        structured_output,
        payload.transcript,
        session_id,
        container
    )

    return {
        "status": "processing_verification",
        "session_id": session_id,
        "data": structured_output
    }