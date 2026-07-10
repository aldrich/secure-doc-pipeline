import asyncio
import sys
import logging
import time
import uuid

from domain.error import ProviderError
from domain.structured_logger import StructuredFormatter
from evaluation.evaluation_engine import EvaluationEngine
from schemas.clinical_summary import ClinicalSummary

logger = logging.getLogger(__name__)

async def run_evaluation(summary_data: ClinicalSummary, source_transcript: str, session_id: str, evaluator: EvaluationEngine):
    """Unified evaluation function using the configured engine."""

    start_time = time.perf_counter()
    
    try:
        metrics = await evaluator.evaluate(summary_data, source_transcript, session_id)
    except ProviderError:
        logger.error("evaluation_provider_failed", extra={"session_id": session_id, "model": evaluator.model})
        return

    elapsed = round(time.perf_counter() - start_time, 2)
    
    if metrics is not None:
        logger.debug(f"evaluation_result: {metrics.model_dump_json(indent=2)}")

        logger.info("evaluation_complete", extra={
            "session_id": session_id,
            "score": metrics.score,
            "faithful": metrics.faithful,
            "model": evaluator.model,
            "latency": elapsed,
        })


async def main():

    source_transcript="""\
Patient initially stated, "I didn't do any physical therapy or movement work over the weekend at all."
However, later in the review, when prompted about specific logs, they recalled and corrected themselves:
"Oh, wait, I actually spent about 20 minutes doing my balance and gait exercises on Saturday afternoon
with the home nurse." They reported feeling stable throughout.
"""

    sample_summary = ClinicalSummary(
        patient_mood="",
        exercises_completed=["balance", "gait"],
        symptoms_mentioned=[],
        next_steps="schedule for 60 mins of strength exercises"
    )
    
    session_id = uuid.uuid4().hex[:8]
    
    from domain.container import DependencyContainer
    container = DependencyContainer()
    
    await run_evaluation(sample_summary, source_transcript, session_id, container.eval_engine)

if __name__ == "__main__":

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    
    asyncio.run(main())