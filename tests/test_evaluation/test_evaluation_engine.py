from unittest.mock import AsyncMock

import pytest

from clients.base import LLMClient
from domain.error import EvaluationError
from evaluation.evaluation_engine import EvaluationEngine
from prompts.evaluation import get_prompt, system_prompt
from schemas.clinical_summary import ClinicalSummary
from schemas.evaluation_metrics import SummaryEvaluation


class TestEvaluationEngine:
    @pytest.mark.asyncio
    async def test_evaluate_success(self):
        summary_data = ClinicalSummary(
            patient_mood="anxious",
            exercises_completed=["balance exercises"],
            symptoms_mentioned=["anxiety"],
            next_steps="Continue daily exercises",
        )
        expected_metrics = SummaryEvaluation(
            faithful=True,
            score=0.95,
            unsupported_claims=[],
            omitted_information=[],
            contradictions=[],
            reasoning="All claims supported by transcript.",
        )

        mock_client = AsyncMock(spec=LLMClient)
        mock_client.generate_structured = AsyncMock(return_value=expected_metrics)

        engine = EvaluationEngine(mock_client, "gemini-2.5-flash")
        result = await engine.evaluate(
            summary_data, "Patient reported feeling anxious.", "sess_001"
        )

        assert isinstance(result, SummaryEvaluation)
        assert result.faithful is True
        assert result.score == 0.95

        mock_client.generate_structured.assert_awaited_once_with(
            model="gemini-2.5-flash",
            system_prompt=system_prompt,
            user_content=get_prompt("Patient reported feeling anxious.", summary_data),
            response_schema=SummaryEvaluation,
        )

    @pytest.mark.asyncio
    async def test_evaluate_wrong_type(self):
        summary_data = ClinicalSummary(
            patient_mood="anxious",
            exercises_completed=[],
            symptoms_mentioned=[],
            next_steps="Rest",
        )

        mock_client = AsyncMock(spec=LLMClient)
        mock_client.generate_structured = AsyncMock(
            return_value="Not a SummaryEvaluation object"
        )

        engine = EvaluationEngine(mock_client, "gemini-2.5-flash")

        with pytest.raises(EvaluationError, match="Unexpected response shape"):
            await engine.evaluate(summary_data, "test transcript", "sess_001")

    @pytest.mark.asyncio
    async def test_evaluate_client_error_propagates(self):
        summary_data = ClinicalSummary(
            patient_mood="anxious",
            exercises_completed=[],
            symptoms_mentioned=[],
            next_steps="Rest",
        )

        mock_client = AsyncMock(spec=LLMClient)
        mock_client.generate_structured = AsyncMock(
            side_effect=EvaluationError("Empty response from DeepSeek")
        )

        engine = EvaluationEngine(mock_client, "deepseek-v4-pro")

        with pytest.raises(EvaluationError, match="Empty response from DeepSeek"):
            await engine.evaluate(summary_data, "test transcript", "sess_001")