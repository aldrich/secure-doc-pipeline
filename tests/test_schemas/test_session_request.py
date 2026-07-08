from pydantic import ValidationError

import pytest

from schemas.session_request import SessionRequest


class TestSessionRequest:
    def test_valid_transcript(self):
        req = SessionRequest(transcript="Patient reported feeling anxious.")
        assert req.transcript == "Patient reported feeling anxious."

    def test_empty_transcript(self):
        with pytest.raises(ValidationError):
            SessionRequest(transcript="")

    def test_transcript_exceeds_max_length(self):
        with pytest.raises(ValidationError):
            SessionRequest(transcript="x" * 5001)

    def test_transcript_at_max_length_boundary(self):
        req = SessionRequest(transcript="x" * 5000)
        assert len(req.transcript) == 5000