from unittest.mock import AsyncMock, Mock

import pytest
from google.genai import types as genai_types

from domain.error import ConfigurationError, EvaluationError
from prompts.evaluation import get_prompt, system_prompt

from evaluation.evaluation_engine import (
    DeepSeekEvaluator,
    EvaluationEngine,
    GeminiEvaluator,
    LlamaEvaluator,
    OpenAIEvaluator,
)
from schemas.clinical_summary import ClinicalSummary
from schemas.evaluation_metrics import SummaryEvaluation


class TestEvaluationEngine:

    @pytest.mark.asyncio
    async def test_gemini_evaluator_evaluate_success(self):
        summary_data = ClinicalSummary(
            patient_mood="anxious",
            exercises_completed=["balance exercises"],
            symptoms_mentioned=["anxiety"],
            next_steps="Continue daily exercises",
        )
        expected_metrics = SummaryEvaluation(
            faithful=True,
            score=0.95,
            unsupported_claims=[],
            omitted_information=[],
            contradictions=[],
            reasoning="All claims supported by transcript.",
        )

        mock_response = Mock(parsed=expected_metrics)

        evaluator = GeminiEvaluator(api_key="sk_test", model="gemini-2.5-flash")
        evaluator.client = AsyncMock()
        evaluator.client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        result = await evaluator.evaluate(summary_data, "Patient reported feeling anxious.", "sess_001")

        assert isinstance(result, SummaryEvaluation)
        assert result.faithful is True
        assert result.score == 0.95
        evaluator.client.aio.models.generate_content.assert_awaited_once_with(
            model="gemini-2.5-flash",
            contents=get_prompt("Patient reported feeling anxious.", summary_data),
            config=genai_types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                response_schema=SummaryEvaluation,
                temperature=0.0,
            ),
        )

    @pytest.mark.asyncio
    async def test_gemini_evaluator_evaluate_wrong_type(self):
        summary_data = ClinicalSummary(
            patient_mood="anxious",
            exercises_completed=[],
            symptoms_mentioned=[],
            next_steps="Rest",
        )

        bad_response = Mock(parsed="Not a SummaryEvaluation object")

        evaluator = GeminiEvaluator(api_key="sk_test", model="gemini-2.5-flash")
        evaluator.client = AsyncMock()
        evaluator.client.aio.models.generate_content = AsyncMock(return_value=bad_response)

        with pytest.raises(EvaluationError, match="Unexpected response shape"):
            await evaluator.evaluate(summary_data, "test transcript", "sess_001")

    def test_gemini_evaluator_init_missing_api_key(self):
        with pytest.raises(ConfigurationError, match="GEMINI_API_KEY not found"):
            GeminiEvaluator(api_key="", model="gemini-2.5-flash")

    def test_gemini_evaluator_init_missing_model(self):
        with pytest.raises(ConfigurationError, match="GEMINI_MODEL_FOR_EVALUATION not found"):
            GeminiEvaluator(api_key="sk_test", model="")

    @pytest.mark.asyncio
    async def test_openai_evaluator_evaluate_success(self):
        summary_data = ClinicalSummary(
            patient_mood="anxious",
            exercises_completed=["balance exercises"],
            symptoms_mentioned=["anxiety"],
            next_steps="Continue daily exercises",
        )
        expected_metrics = SummaryEvaluation(
            faithful=True,
            score=0.95,
            unsupported_claims=[],
            omitted_information=[],
            contradictions=[],
            reasoning="All claims supported by transcript.",
        )

        mock_response = Mock(output_parsed=expected_metrics)

        evaluator = OpenAIEvaluator(api_key="sk_test", model="gpt-4.1-mini")
        evaluator.client = AsyncMock()
        evaluator.client.responses.parse = AsyncMock(return_value=mock_response)

        result = await evaluator.evaluate(summary_data, "Patient reported feeling anxious.", "sess_001")

        assert isinstance(result, SummaryEvaluation)
        assert result.faithful is True
        evaluator.client.responses.parse.assert_awaited_once_with(
            model="gpt-4.1-mini",
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": get_prompt("Patient reported feeling anxious.", summary_data)},
            ],
            text_format=SummaryEvaluation,
        )

    @pytest.mark.asyncio
    async def test_openai_evaluator_evaluate_wrong_type(self):
        summary_data = ClinicalSummary(
            patient_mood="anxious",
            exercises_completed=[],
            symptoms_mentioned=[],
            next_steps="Rest",
        )

        bad_response = Mock()
        bad_response.output_parsed = "Not a SummaryEvaluation object"

        evaluator = OpenAIEvaluator(api_key="sk_test", model="gpt-4.1-mini")
        evaluator.client = AsyncMock()
        evaluator.client.responses.parse = AsyncMock(return_value=bad_response)

        with pytest.raises(EvaluationError, match="Unexpected response shape"):
            await evaluator.evaluate(summary_data, "test transcript", "sess_001")

    def test_openai_evaluator_init_missing_api_key(self):
        with pytest.raises(ConfigurationError, match="OPENAI_API_KEY not found"):
            OpenAIEvaluator(api_key="", model="gpt-4.1-mini")

    def test_openai_evaluator_init_missing_model(self):
        with pytest.raises(ConfigurationError, match="OPENAI_MODEL_FOR_EVALUATION not found"):
            OpenAIEvaluator(api_key="sk_test", model="")

    @pytest.mark.asyncio
    async def test_llama_evaluator_evaluate_success(self, mocker):
        summary_data = ClinicalSummary(
            patient_mood="anxious",
            exercises_completed=["balance exercises"],
            symptoms_mentioned=["anxiety"],
            next_steps="Continue daily exercises",
        )
        expected_metrics = SummaryEvaluation(
            faithful=True,
            score=0.95,
            unsupported_claims=[],
            omitted_information=[],
            contradictions=[],
            reasoning="All claims supported by transcript.",
        )

        mock_response = {"message": {"content": expected_metrics.model_dump_json()}}

        mock_ollama_client = AsyncMock()
        mock_ollama_client.__aenter__.return_value = mock_ollama_client
        mock_ollama_client.chat = AsyncMock(return_value=mock_response)
        mocker.patch("ollama.AsyncClient", return_value=mock_ollama_client)

        evaluator = LlamaEvaluator(model="llama3.1:8b", ollama_host="http://localhost:11434")
        result = await evaluator.evaluate(summary_data, "Patient reported feeling anxious.", "sess_001")

        assert isinstance(result, SummaryEvaluation)
        assert result.faithful is True
        mock_ollama_client.chat.assert_awaited_once_with(
            model="llama3.1:8b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": get_prompt("Patient reported feeling anxious.", summary_data)},
            ],
            format=SummaryEvaluation.model_json_schema(),
        )

    def test_llama_evaluator_init_missing_model(self):
        with pytest.raises(ConfigurationError, match="LLAMA_MODEL_FOR_EVALUATION not found"):
            LlamaEvaluator(model="", ollama_host="http://localhost:11434")

    @pytest.mark.asyncio
    async def test_deepseek_evaluator_evaluate_success(self):
        summary_data = ClinicalSummary(
            patient_mood="anxious",
            exercises_completed=["balance exercises"],
            symptoms_mentioned=["anxiety"],
            next_steps="Continue daily exercises",
        )
        expected_metrics = SummaryEvaluation(
            faithful=True,
            score=0.95,
            unsupported_claims=[],
            omitted_information=[],
            contradictions=[],
            reasoning="All claims supported by transcript.",
        )

        mock_response = Mock(choices=[
            Mock(message=Mock(content=expected_metrics.model_dump_json()))
        ])

        evaluator = DeepSeekEvaluator(
            api_key="sk_test", model="deepseek-v4-pro", base_url="https://api.deepseek.com"
        )
        evaluator.client = AsyncMock()
        evaluator.client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await evaluator.evaluate(summary_data, "Patient reported feeling anxious.", "sess_001")

        assert isinstance(result, SummaryEvaluation)
        assert result.faithful is True

        response_shape_spec = f"respond in the following JSON format: {SummaryEvaluation.model_json_schema()}"
        modified_system_prompt = "\n".join([system_prompt, response_shape_spec])

        evaluator.client.chat.completions.create.assert_awaited_once_with(
            model="deepseek-v4-pro",
            messages=[
                {"role": "system", "content": modified_system_prompt},
                {"role": "user", "content": get_prompt("Patient reported feeling anxious.", summary_data)},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )

    @pytest.mark.asyncio
    async def test_deepseek_evaluator_evaluate_empty_response(self):
        summary_data = ClinicalSummary(
            patient_mood="anxious",
            exercises_completed=[],
            symptoms_mentioned=[],
            next_steps="Rest",
        )

        bad_response = Mock()
        bad_response.choices = [Mock(message=Mock(content=None))]

        evaluator = DeepSeekEvaluator(
            api_key="sk_test", model="deepseek-v4-pro", base_url="https://api.deepseek.com"
        )
        evaluator.client = AsyncMock()
        evaluator.client.chat.completions.create = AsyncMock(return_value=bad_response)

        with pytest.raises(EvaluationError, match="Empty response from DeepSeek"):
            await evaluator.evaluate(summary_data, "test transcript", "sess_001")

    def test_deepseek_evaluator_init_missing_api_key(self):
        with pytest.raises(ConfigurationError, match="DEEPSEEK_API_KEY not found"):
            DeepSeekEvaluator(api_key="", model="deepseek-v4-pro", base_url="https://api.deepseek.com")

    def test_deepseek_evaluator_init_missing_model(self):
        with pytest.raises(ConfigurationError, match="DEEPSEEK_MODEL_FOR_EVALUATION not found"):
            DeepSeekEvaluator(api_key="sk_test", model="", base_url="https://api.deepseek.com")

    def test_abstract_base_class_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            EvaluationEngine()