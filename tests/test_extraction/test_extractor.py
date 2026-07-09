import pytest

from extraction.extractor import run_extraction
from schemas.clinical_summary import ClinicalSummary


class TestRunExtraction:
    @pytest.mark.asyncio
    async def test_returns_clinical_summary(self, mock_extract_engine):
        mock_extract_engine.extract.return_value = ClinicalSummary(
            patient_mood="anxious",
            exercises_completed=["balance exercises"],
            symptoms_mentioned=["anxiety"],
            next_steps="Continue daily exercises",
        )

        result = await run_extraction("test transcript", "sess_001", mock_extract_engine)

        assert isinstance(result, ClinicalSummary)

    @pytest.mark.asyncio
    async def test_passes_correct_args_to_engine(self, mock_extract_engine):
        mock_extract_engine.extract.return_value = ClinicalSummary(
            patient_mood="calm",
            exercises_completed=[],
            symptoms_mentioned=[],
            next_steps="Rest",
        )

        await run_extraction("Hello patient", "sess_002", mock_extract_engine)

        mock_extract_engine.extract.assert_awaited_once_with("Hello patient", "sess_002")

    @pytest.mark.asyncio
    async def test_propagates_engine_errors(self, mock_extract_engine):
        from domain.error import ExtractionError
        mock_extract_engine.extract.side_effect = ExtractionError("Engine failed")

        with pytest.raises(ExtractionError, match="Engine failed"):
            await run_extraction("test", "sess_003", mock_extract_engine)