import pytest
from unittest.mock import AsyncMock, Mock, patch

from fastapi.testclient import TestClient
from api.routes.transformers import router
from fastapi import FastAPI
from domain.error import ExtractionError
from schemas.clinical_summary import ClinicalSummary

class TestProcessSession:
    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        monkeypatch.setenv("API_KEY", "test-api-key")
        monkeypatch.setattr("api.routes.transformers.uuid.uuid4", lambda: Mock(hex="abc12345"))
        self.test_app = FastAPI()
        self.test_app.include_router(router)
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
        container.eval_engine = AsyncMock()
        return container

    def test_successful_processing(self, mock_extraction, container_with_async_eval):
        self.test_app.state.container = container_with_async_eval
        with patch("api.routes.transformers.run_extraction", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_extraction
            response = self.client.post(
                "/process-session",
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
            "/process-session",
            json={"transcript": "Patient reported feeling anxious."},
        )
        assert response.status_code == 401

    def test_wrong_api_key(self, mock_extraction, container_with_async_eval):
        self.test_app.state.container = container_with_async_eval
        response = self.client.post(
            "/process-session",
            json={"transcript": "Patient reported feeling anxious."},
            headers={"X-API-Key": "wrong-key"},
        )
        assert response.status_code == 401

    def test_empty_body(self, container_with_async_eval):
        self.test_app.state.container = container_with_async_eval
        response = self.client.post(
            "/process-session",
            json={},
            headers={"X-API-Key": "test-api-key"},
        )
        assert response.status_code == 422

    def test_missing_transcript_field(self, container_with_async_eval):
        self.test_app.state.container = container_with_async_eval
        response = self.client.post(
            "/process-session",
            json={},
            headers={"X-API-Key": "test-api-key"},
        )
        assert response.status_code == 422

    def test_extraction_failure(self, container_with_async_eval):
        self.test_app.state.container = container_with_async_eval
        with patch("api.routes.transformers.run_extraction", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = ExtractionError("Extraction failed")
            response = self.client.post(
                "/process-session",
                json={"transcript": "Patient reported feeling anxious."},
                headers={"X-API-Key": "test-api-key"},
            )
        assert response.status_code == 500

    def test_background_task_scheduled(self, mock_extraction, container_with_async_eval):
        self.test_app.state.container = container_with_async_eval
        with (
            patch("api.routes.transformers.run_extraction", new_callable=AsyncMock) as mock_run,
            patch("api.routes.transformers.run_evaluation") as mock_eval,
        ):
            mock_run.return_value = mock_extraction
            self.client.post(
                "/process-session",
                json={"transcript": "Patient reported feeling anxious."},
                headers={"X-API-Key": "test-api-key"},
            )
            mock_eval.assert_called_once()