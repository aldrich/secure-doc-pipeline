import asyncio
import logging
import sys
import time
import uuid

from domain.error import ProviderError
from domain.structured_logger import StructuredFormatter
from extraction.extraction_engine import ExtractionEngine
from schemas.clinical_summary import ClinicalSummary

logger = logging.getLogger(__name__)

async def run_extraction(source_transcript: str, session_id: str, extractor: ExtractionEngine) -> ClinicalSummary:
    """Unified extractor function using the configured engine."""
    
    start_time = time.perf_counter()
    
    try:
        clinical_summary = await extractor.extract(source_transcript, session_id)
    except ProviderError:
        logger.error("extraction_provider_failed", extra={"session_id": session_id, "model": extractor.model})
        raise

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
    
    from domain.container import DependencyContainer
    container = DependencyContainer()
    
    asyncio.run(run_extraction(source_transcript, session_id, container.extract_engine))

if __name__ == "__main__":

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler])

    main()