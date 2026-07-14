import logging
import uuid

from fastapi import BackgroundTasks

from domain.repository import EvaluationRepository
from evaluation.evaluation_engine import EvaluationEngine
from evaluation.evaluator import run_evaluation
from extraction.extraction_engine import ExtractionEngine
from extraction.extractor import run_extraction

logger = logging.getLogger(__name__)


class PipelineService:
    def __init__(
        self,
        extractor: ExtractionEngine,
        evaluator: EvaluationEngine,
        evaluation_repo: EvaluationRepository,
    ):

        self.extractor = extractor
        self.evaluator = evaluator
        self.repo = evaluation_repo

    async def process_session(self, transcript: str, background_tasks: BackgroundTasks):

        session_id = uuid.uuid4().hex[:8]
        logger.info("session_processing_request", extra={"session_id": session_id})

        structured_output = await run_extraction(transcript, session_id, self.extractor)

        background_tasks.add_task(
            run_evaluation,
            structured_output,
            transcript,
            session_id,
            self.evaluator,
            self.repo,
        )

        return {
            "status": "processing_verification",
            "session_id": session_id,
            "data": structured_output,
        }
