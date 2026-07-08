from schemas.clinical_summary import ClinicalSummary


class TestClinicalSummary:
    def test_valid_full_creation(self) -> None:
        summary = ClinicalSummary(
            patient_mood="anxious",
            exercises_completed=["balance exercises", "stretching"],
            symptoms_mentioned=["headache", "fatigue"],
            next_steps="Continue daily exercises",
        )
        assert summary.patient_mood == "anxious"
        assert summary.exercises_completed == ["balance exercises", "stretching"]
        assert summary.symptoms_mentioned == ["headache", "fatigue"]
        assert summary.next_steps == "Continue daily exercises"

    def test_empty_strings_allowed(self) -> None:
        summary = ClinicalSummary(
            patient_mood="",
            exercises_completed=["walking"],
            symptoms_mentioned=[],
            next_steps="",
        )
        assert summary.patient_mood == ""
        assert summary.next_steps == ""

    def test_empty_lists_allowed(self) -> None:
        summary = ClinicalSummary(
            patient_mood="calm",
            exercises_completed=[],
            symptoms_mentioned=[],
            next_steps="Rest",
        )
        assert summary.exercises_completed == []
        assert summary.symptoms_mentioned == []

    def test_serialization_round_trip(self) -> None:
        original = ClinicalSummary(
            patient_mood="happy",
            exercises_completed=["yoga"],
            symptoms_mentioned=["stiffness"],
            next_steps="Stretch daily",
        )
        data = original.model_dump()
        restored = ClinicalSummary.model_validate(data)
        assert restored == original

    def test_field_descriptions_set(self) -> None:
        fields = ClinicalSummary.model_fields
        assert fields["patient_mood"].description is not None
        assert fields["exercises_completed"].description is not None
        assert fields["symptoms_mentioned"].description is not None
        assert fields["next_steps"].description is not None

    def test_non_string_fields_rejected(self) -> None:
        from pydantic import ValidationError
        import pytest

        with pytest.raises(ValidationError):
            ClinicalSummary(
                patient_mood=123,  # type: ignore
                exercises_completed=["walking"],
                symptoms_mentioned=[],
                next_steps="Rest",
            )