from pydantic import ValidationError

import pytest

from schemas.evaluation_metrics import (
    Contradiction,
    EvaluationMetrics,
    OmittedInformation,
    SummaryEvaluation,
    UnsupportedClaim,
)


class TestSummaryEvaluation:
    def test_valid_summary_evaluation(self):
        evaluation = SummaryEvaluation(
            faithful=True,
            score=0.85,
            unsupported_claims=[],
            omitted_information=[],
            contradictions=[],
            reasoning="All claims supported.",
        )
        assert evaluation.faithful is True
        assert evaluation.score == 0.85

    def test_score_boundary_zero(self):
        evaluation = SummaryEvaluation(
            faithful=False,
            score=0.0,
            unsupported_claims=[],
            omitted_information=[],
            contradictions=[],
            reasoning="Fabricated.",
        )
        assert evaluation.score == 0.0

    def test_score_boundary_one(self):
        evaluation = SummaryEvaluation(
            faithful=True,
            score=1.0,
            unsupported_claims=[],
            omitted_information=[],
            contradictions=[],
            reasoning="Perfect.",
        )
        assert evaluation.score == 1.0

    def test_score_below_zero(self):
        with pytest.raises(ValidationError):
            SummaryEvaluation(
                faithful=True,
                score=-0.1,
                unsupported_claims=[],
                omitted_information=[],
                contradictions=[],
                reasoning="Bad.",
            )

    def test_score_above_one(self):
        with pytest.raises(ValidationError):
            SummaryEvaluation(
                faithful=True,
                score=1.1,
                unsupported_claims=[],
                omitted_information=[],
                contradictions=[],
                reasoning="Bad.",
            )


class TestUnsupportedClaim:
    def test_with_evidence(self):
        claim = UnsupportedClaim(claim="Patient ran 5k", evidence="No mention of running.")
        assert claim.claim == "Patient ran 5k"
        assert claim.evidence == "No mention of running."

    def test_without_evidence(self):
        claim = UnsupportedClaim(claim="Patient ran 5k", evidence=None)
        assert claim.evidence is None


class TestOmittedInformation:
    @pytest.mark.parametrize("importance", ["low", "medium", "high"])
    def test_valid_importance(self, importance):
        info = OmittedInformation(information="Patient mentioned headache", importance=importance)
        assert info.importance == importance

    def test_invalid_importance(self):
        with pytest.raises(ValidationError):
            OmittedInformation(information="Headache", importance="critical")


class TestContradiction:
    def test_valid_contradiction(self):
        c = Contradiction(
            summary_statement="Patient was happy",
            transcript_evidence="Patient reported feeling sad",
        )
        assert c.summary_statement == "Patient was happy"
        assert c.transcript_evidence == "Patient reported feeling sad"


class TestEvaluationMetrics:
    def test_valid_evaluation_metrics(self):
        metrics = EvaluationMetrics(
            passed=True,
            faithfulness_score=0.9,
            unsupported_exercises=["yoga"],
            omitted_symptoms=["dizziness"],
            latency_seconds=1.5,
        )
        assert metrics.passed is True
        assert metrics.faithfulness_score == 0.9
        assert metrics.latency_seconds == 1.5

    def test_latency_defaults_to_zero(self):
        metrics = EvaluationMetrics(
            passed=True,
            faithfulness_score=1.0,
            unsupported_exercises=[],
            omitted_symptoms=[],
        )
        assert metrics.latency_seconds == 0.0