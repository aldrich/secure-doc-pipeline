from schemas.clinical_summary import ClinicalSummary
from prompts.evaluation import get_prompt, system_prompt


class TestEvaluationPrompt:
    def test_system_prompt_is_non_empty(self):
        assert system_prompt is not None
        assert len(system_prompt.strip()) > 0

    def test_get_prompt_includes_transcript(self):
        transcript = "Patient reported feeling anxious."
        summary = ClinicalSummary(
            patient_mood="anxious",
            exercises_completed=[],
            symptoms_mentioned=["anxiety"],
            next_steps="Rest",
        )
        result = get_prompt(transcript, summary)
        assert transcript in result

    def test_get_prompt_includes_summary_json(self):
        transcript = "Patient reported feeling anxious."
        summary = ClinicalSummary(
            patient_mood="anxious",
            exercises_completed=[],
            symptoms_mentioned=["anxiety"],
            next_steps="Rest",
        )
        result = get_prompt(transcript, summary)
        assert summary.model_dump_json(indent=2) in result