from schemas.clinical_summary import ClinicalSummary
from schemas.session_response import SessionResponse


class TestSessionResponse:
    def test_valid_session_response(self):
        summary = ClinicalSummary(
            patient_mood="anxious",
            exercises_completed=["balance exercises"],
            symptoms_mentioned=["anxiety"],
            next_steps="Continue daily exercises",
        )
        response = SessionResponse(
            status="processing_verification",
            session_id="abc12345",
            data=summary,
        )
        assert response.status == "processing_verification"
        assert response.session_id == "abc12345"
        assert response.data == summary

    def test_serialization_to_dict(self):
        summary = ClinicalSummary(
            patient_mood="calm",
            exercises_completed=[],
            symptoms_mentioned=[],
            next_steps="Rest",
        )
        response = SessionResponse(
            status="done", session_id="xyz789", data=summary
        )
        dumped = response.model_dump()
        assert dumped["status"] == "done"
        assert dumped["session_id"] == "xyz789"
        assert dumped["data"]["patient_mood"] == "calm"