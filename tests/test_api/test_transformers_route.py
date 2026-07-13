from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from domain.auth import verify_api_key
from domain.error import ExtractionError
from domain.settings import settings
from main import extraction_handler, process_session
from schemas.clinical_summary import ClinicalSummary
from schemas.session_response import SessionResponse
from services.pipeline import PipelineService


class TestProcessSession:
    @pytest.fixture(autouse=True)
    def setup(self, mocker):
        mocker.patch.object(settings, "api_key", "test-api-key")
        mocker.patch("services.pipeline.uuid.uuid4", return_value=Mock(hex="abc12345"))
        self.test_app = FastAPI()
        self.test_app.post(
            "/api/v1/process-session",
            status_code=202,
            response_model=SessionResponse,
            dependencies=[Depends(verify_api_key)],
        )(process_session)
        self.test_app.exception_handler(ExtractionError)(extraction_handler)
        self.client = TestClient(self.test_app, raise_server_exceptions=False)

    @pytest.fixture
    def mock_extraction(self):
        return ClinicalSummary(
            patient_mood="anxious",
            exercises_completed=["balance exercises"],
            symptoms_mentioned=["anxiety"],
            next_steps="Continue daily exercises",
        )

    @pytest.fixture
    def container_with_async_eval(self):
        container = Mock()
        eval_engine = AsyncMock()
        eval_engine.evaluate.return_value = None
        container.pipeline_service = PipelineService(
            extractor=Mock(),
            evaluator=eval_engine,
        )
        return container

    def test_successful_processing(self, mock_extraction, container_with_async_eval):
        self.test_app.state.container = container_with_async_eval
        with patch(
            "services.pipeline.run_extraction", new_callable=AsyncMock
        ) as mock_run:
            mock_run.return_value = mock_extraction
            response = self.client.post(
                "/api/v1/process-session",
                json={"transcript": "Patient reported feeling anxious."},
                headers={"X-API-Key": "test-api-key"},
            )
        assert response.status_code == 202
        data = response.json()
        assert data["status"] == "processing_verification"
        assert data["session_id"] == "abc12345"
        assert data["data"]["patient_mood"] == "anxious"

    def test_missing_api_key(self, mock_extraction, container_with_async_eval):
        self.test_app.state.container = container_with_async_eval
        response = self.client.post(
            "/api/v1/process-session",
            json={"transcript": "Patient reported feeling anxious."},
        )
        assert response.status_code == 401

    def test_wrong_api_key(self, mock_extraction, container_with_async_eval):
        self.test_app.state.container = container_with_async_eval
        response = self.client.post(
            "/api/v1/process-session",
            json={"transcript": "Patient reported feeling anxious."},
            headers={"X-API-Key": "wrong-key"},
        )
        assert response.status_code == 401

    def test_empty_body(self, container_with_async_eval):
        self.test_app.state.container = container_with_async_eval
        response = self.client.post(
            "/api/v1/process-session",
            json={},
            headers={"X-API-Key": "test-api-key"},
        )
        assert response.status_code == 422

    def test_missing_transcript_field(self, container_with_async_eval):
        self.test_app.state.container = container_with_async_eval
        response = self.client.post(
            "/api/v1/process-session",
            json={},
            headers={"X-API-Key": "test-api-key"},
        )
        assert response.status_code == 422

    def test_extraction_failure(self, container_with_async_eval):
        self.test_app.state.container = container_with_async_eval
        with patch(
            "services.pipeline.run_extraction", new_callable=AsyncMock
        ) as mock_run:
            mock_run.side_effect = ExtractionError("Extraction failed")
            response = self.client.post(
                "/api/v1/process-session",
                json={"transcript": "Patient reported feeling anxious."},
                headers={"X-API-Key": "test-api-key"},
            )
        assert response.status_code == 422

    def test_background_task_scheduled(
        self, mock_extraction, container_with_async_eval
    ):
        self.test_app.state.container = container_with_async_eval
        with (
            patch(
                "services.pipeline.run_extraction", new_callable=AsyncMock
            ) as mock_run,
            patch("services.pipeline.run_evaluation") as mock_eval,
        ):
            mock_run.return_value = mock_extraction
            mock_eval.return_value = None
            self.client.post(
                "/api/v1/process-session",
                json={"transcript": "Patient reported feeling anxious."},
                headers={"X-API-Key": "test-api-key"},
            )
            mock_eval.assert_called_once()
