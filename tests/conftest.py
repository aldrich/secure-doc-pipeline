from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, Mock

import pytest

from schemas.clinical_summary import ClinicalSummary
from schemas.evaluation_metrics import SummaryEvaluation


@pytest.fixture
def sample_transcript() -> str:
    return "Patient reported feeling anxious. Completed balance exercises for 20 minutes."


@pytest.fixture
def sample_clinical_summary() -> ClinicalSummary:
    return ClinicalSummary(
        patient_mood="anxious",
        exercises_completed=["balance exercises"],
        symptoms_mentioned=["anxiety"],
        next_steps="Continue balance exercises daily",
    )


@pytest.fixture
def sample_summary_evaluation() -> SummaryEvaluation:
    return SummaryEvaluation(
        faithful=True,
        score=0.95,
        unsupported_claims=[],
        omitted_information=[],
        contradictions=[],
        reasoning="All claims supported by transcript.",
    )


@pytest.fixture
def sample_session_id() -> str:
    return "abc12345"


@pytest.fixture
def mock_container() -> Mock:
    return Mock()


@pytest.fixture
def mock_extract_engine() -> AsyncMock:
    engine = AsyncMock()
    engine.model = "test-model"
    return engine


@pytest.fixture
def mock_eval_engine() -> AsyncMock:
    engine = AsyncMock()
    engine.model = "test-model"
    return engine

@pytest.fixture
def mock_evaluation_repo() -> AsyncMock:
    return AsyncMock()