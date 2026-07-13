from unittest.mock import AsyncMock

import httpx
import pytest

from clients.base import LLMClient
from extraction.extraction_engine import ExtractionEngine
from schemas.clinical_summary import ClinicalSummary


@pytest.mark.asyncio
async def test_extraction_engine_retries_on_timeout():
    """Test that ExtractionEngine retries on httpx timeout errors."""

    mock_client = AsyncMock(spec=LLMClient)

    mock_summary = ClinicalSummary(
        patient_mood="calm",
        exercises_completed=[],
        symptoms_mentioned=[],
        next_steps="",
    )
    mock_client.generate_structured.side_effect = [
        httpx.TimeoutException("timed out"),
        httpx.TimeoutException("timed out 2"),
        mock_summary,
    ]

    engine = ExtractionEngine(
        mock_client,
        "gpt-4.1-nano",
        max_retries=2,
        retry_base_delay=0.01,
        retry_max_delay=0.1,
    )

    result = await engine.extract("Patient is calm.", "sess_001")
    assert isinstance(result, ClinicalSummary)
    assert mock_client.generate_structured.call_count == 3