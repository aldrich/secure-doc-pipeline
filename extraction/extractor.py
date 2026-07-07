import asyncio
import logging, sys, time
import uuid

from domain.container import DependencyContainer
from domain.structured_logger import StructuredFormatter
from schemas.clinical_summary import ClinicalSummary

logger = logging.getLogger(__name__)

async def run_extraction(source_transcript: str, session_id: str, container: DependencyContainer) -> ClinicalSummary:
    """Unified extractor function using the configured engine."""
    
    extractor = container.extract_engine

    start_time = time.perf_counter()
    
    clinical_summary = await extractor.extract(source_transcript, session_id)

    elapsed = round(time.perf_counter() - start_time, 2)
    
    logger.debug(f"extraction_result: {clinical_summary.model_dump_json(indent=2)}")

    logger.info("extraction_complete", extra={
        "session_id": session_id,
        "model": extractor.model,
        "latency": elapsed,
    })
    
    return clinical_summary

def main():

    source_transcript="""\
Patient initially stated, "I didn't do any physical therapy or movement work over the weekend at all."
However, later in the review, when prompted about specific logs, they recalled and corrected themselves:
"Oh, wait, I actually spent about 20 minutes doing my balance and gait exercises on Saturday afternoon
with the home nurse." They reported feeling stable throughout.
"""

    session_id = uuid.uuid4().hex[:8]
    container = DependencyContainer()
    asyncio.run(run_extraction(source_transcript, session_id, container))

if __name__ == "__main__":

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredFormatter())
    logging.basicConfig(level=logging.DEBUG, handlers=[handler])

    main()