from unittest.mock import AsyncMock

import pytest

from evaluation.evaluator import run_evaluation
from schemas.clinical_summary import ClinicalSummary
from schemas.evaluation_metrics import SummaryEvaluation


class TestRunEvaluation:
    @pytest.mark.asyncio
    async def test_returns_none(self, mock_container, mock_eval_engine, sample_clinical_summary):
        mock_container.eval_engine = mock_eval_engine

        result = await run_evaluation(
            sample_clinical_summary, "test transcript", "sess_001", mock_container
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_passes_correct_args_to_engine(self, mock_container, mock_eval_engine, sample_clinical_summary):
        expected = SummaryEvaluation(
            faithful=True,
            score=1.0,
            unsupported_claims=[],
            omitted_information=[],
            contradictions=[],
            reasoning="Perfect.",
        )
        mock_eval_engine.evaluate.return_value = expected
        mock_container.eval_engine = mock_eval_engine

        await run_evaluation(sample_clinical_summary, "Hello transcript", "sess_002", mock_container)

        mock_eval_engine.evaluate.assert_awaited_once_with(
            sample_clinical_summary, "Hello transcript", "sess_002"
        )

    @pytest.mark.asyncio
    async def test_propagates_engine_errors(self, mock_container, mock_eval_engine, sample_clinical_summary):
        from domain.error import EvaluationError
        mock_eval_engine.evaluate.side_effect = EvaluationError("Eval failed")
        mock_container.eval_engine = mock_eval_engine

        with pytest.raises(EvaluationError, match="Eval failed"):
            await run_evaluation(sample_clinical_summary, "test", "sess_003", mock_container)