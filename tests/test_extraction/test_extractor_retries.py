# Put this in a scratch file or a new test
from unittest.mock import AsyncMock

import httpx
import pytest

from extraction.extraction_engine import OpenAIExtractor
from schemas.clinical_summary import ClinicalSummary


@pytest.mark.asyncio
async def test_openai_extractor_retries_on_timeout():
    """Test that the OpenAI extractor retries on timeout.

    Note: Add the following variables to the .env file to make retries happen sooner
        LLM_MAX_RETRIES=2
        LLM_RETRY_BASE_DELAY=0.5
        LLM_RETRY_MAX_DELAY=2.0
        LLM_TIMEOUT=5
    """
    
    extractor = OpenAIExtractor(
        api_key="sk_test",
        model="gpt-4.1-nano",
        max_retries=2,
        retry_base_delay=0.01,
        retry_max_delay=0.1,
    )
    extractor.client = AsyncMock()

    # First call: timeout. Second call: success.
    mock_response = AsyncMock()
    mock_response.output_parsed = ClinicalSummary(
        patient_mood="calm",
        exercises_completed=[],
        symptoms_mentioned=[],
        next_steps="",
    )
    extractor.client.responses.parse.side_effect = [
        httpx.TimeoutException("timed out"),
        httpx.TimeoutException("timed out 2"),
        mock_response,
    ]

    result = await extractor.extract("Patient is calm.", "sess_001")
    assert isinstance(result, ClinicalSummary)
    assert extractor.client.responses.parse.call_count == 3
