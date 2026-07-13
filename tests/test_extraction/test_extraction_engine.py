from unittest.mock import AsyncMock

import pytest

from clients.base import LLMClient
from domain.error import ExtractionError
from extraction.extraction_engine import ExtractionEngine
from prompts.extraction import system_prompt
from schemas.clinical_summary import ClinicalSummary


class TestExtractionEngine:
    @pytest.mark.asyncio
    async def test_extract_success(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.generate_structured = AsyncMock(
            return_value=ClinicalSummary(
                patient_mood="anxious",
                exercises_completed=["balance exercises"],
                symptoms_mentioned=["anxiety"],
                next_steps="Continue daily exercises",
            )
        )

        engine = ExtractionEngine(mock_client, "gpt-4.1-nano")
        result = await engine.extract(
            "Patient reported feeling anxious.", "sess_001"
        )

        assert isinstance(result, ClinicalSummary)
        assert result.patient_mood == "anxious"

        mock_client.generate_structured.assert_awaited_once_with(
            model="gpt-4.1-nano",
            system_prompt=system_prompt,
            user_content="<transcript>Patient reported feeling anxious.</transcript>",
            response_schema=ClinicalSummary,
        )

    @pytest.mark.asyncio
    async def test_extract_wrong_type(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.generate_structured = AsyncMock(
            return_value="Not a ClinicalSummary object"
        )

        engine = ExtractionEngine(mock_client, "gpt-4.1-nano")

        with pytest.raises(ExtractionError, match="Unexpected response shape"):
            await engine.extract("Patient reported feeling anxious!", "sess_001")

    @pytest.mark.asyncio
    async def test_extract_client_error_propagates(self):
        mock_client = AsyncMock(spec=LLMClient)
        mock_client.generate_structured = AsyncMock(
            side_effect=ExtractionError("Empty response from DeepSeek")
        )

        engine = ExtractionEngine(mock_client, "deepseek-chat")

        with pytest.raises(ExtractionError, match="Empty response from DeepSeek"):
            await engine.extract("Patient reported feeling anxious!", "sess_001")