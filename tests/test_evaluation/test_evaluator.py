import pytest

from evaluation.evaluator import run_evaluation
from schemas.evaluation_metrics import SummaryEvaluation

class TestRunEvaluation:
    @pytest.mark.asyncio
    async def test_returns_none(self, mock_eval_engine, sample_clinical_summary):
        mock_eval_engine.evaluate.return_value = None

        result = await run_evaluation(
            sample_clinical_summary, "test transcript", "sess_001", mock_eval_engine
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_passes_correct_args_to_engine(self, mock_eval_engine, sample_clinical_summary):
        expected = SummaryEvaluation(
            faithful=True,
            score=1.0,
            unsupported_claims=[],
            omitted_information=[],
            contradictions=[],
            reasoning="Perfect.",
        )
        mock_eval_engine.evaluate.return_value = expected

        await run_evaluation(sample_clinical_summary, "Hello transcript", "sess_002", mock_eval_engine)

        mock_eval_engine.evaluate.assert_awaited_once_with(
            sample_clinical_summary, "Hello transcript", "sess_002"
        )

    @pytest.mark.asyncio
    async def test_propagates_engine_errors(self, mock_eval_engine, sample_clinical_summary):
        from domain.error import EvaluationError
        mock_eval_engine.evaluate.side_effect = EvaluationError("Eval failed")

        with pytest.raises(EvaluationError, match="Eval failed"):
            await run_evaluation(sample_clinical_summary, "test", "sess_003", mock_eval_engine)