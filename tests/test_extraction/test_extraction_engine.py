from unittest.mock import AsyncMock, Mock

import pytest
from google.genai import types as genai_types

from domain.error import ConfigurationError, ExtractionError
from extraction.extraction_engine import (
    DeepSeekExtractor,
    ExtractionEngine,
    GeminiExtractor,
    LlamaExtractor,
    OpenAIExtractor,
)
from prompts.extraction import system_prompt
from schemas.clinical_summary import ClinicalSummary


class TestExtractionEngine:
    @pytest.mark.asyncio
    async def test_openai_extractor_extract_success(self):

        mock_response = Mock(
            output_parsed=ClinicalSummary(
                patient_mood="anxious",
                exercises_completed=["balance exercises"],
                symptoms_mentioned=["anxiety"],
                next_steps="Continue daily exercises",
            )
        )

        extractor = OpenAIExtractor(api_key="sk_test", model="gpt-4.1-nano")
        extractor.client = AsyncMock()
        extractor.client.responses.parse = AsyncMock(return_value=mock_response)

        result = await extractor.extract(
            "Patient reported feeling anxious.", "sess_001"
        )

        assert isinstance(result, ClinicalSummary)
        assert result.patient_mood == "anxious"

        extractor.client.responses.parse.assert_called_once_with(
            model="gpt-4.1-nano",
            input=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": "<transcript>Patient reported feeling anxious.</transcript>",
                },
            ],
            text_format=ClinicalSummary,
        )

    @pytest.mark.asyncio
    async def test_openai_extractor_extract_wrong_type(self):
        extractor = OpenAIExtractor(api_key="sk_test", model="gpt-4.1-nano")
        extractor.client = AsyncMock()

        bad_response = Mock()
        bad_response.output_parsed = "Not a ClinicalSummary object"

        extractor.client.responses.parse = AsyncMock(return_value=bad_response)

        with pytest.raises(ExtractionError, match="Unexpected response shape"):
            await extractor.extract("Patient reported feeling anxious!", "sess_001")

    def test_openai_extractor_init_missing_api_key(self):
        with pytest.raises(ConfigurationError, match="OPENAI_API_KEY not found"):
            OpenAIExtractor(api_key="", model="gpt-4.1-nano")

    def test_openai_extractor_init_missing_model(self):
        with pytest.raises(
            ConfigurationError, match="OPENAI_MODEL_FOR_EXTRACTION not found"
        ):
            OpenAIExtractor(api_key="sk_test", model="")

    @pytest.mark.asyncio
    async def test_gemini_extractor_extract_success(self):

        mock_response = Mock(
            parsed=ClinicalSummary(
                patient_mood="anxious",
                exercises_completed=["balance exercises"],
                symptoms_mentioned=["anxiety"],
                next_steps="Continue daily exercises",
            )
        )

        extractor = GeminiExtractor(api_key="sk_test", model="gemini-1.5-flash")
        extractor.client = Mock()

        extractor.client.aio.models.generate_content = AsyncMock(
            return_value=mock_response
        )

        result = await extractor.extract(
            "Patient reported feeling anxious.", "sess_001"
        )

        assert isinstance(result, ClinicalSummary)
        assert result.patient_mood == "anxious"

        extractor.client.aio.models.generate_content.assert_called_once_with(
            model="gemini-1.5-flash",
            contents="<transcript>\nPatient reported feeling anxious.\n</transcript>\n",
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ClinicalSummary,
                system_instruction=system_prompt,
                temperature=0,
            ),
        )

    @pytest.mark.asyncio
    async def test_gemini_extractor_extract_wrong_type(self):
        extractor = GeminiExtractor(api_key="sk_test", model="gemini-1.5-flash")
        extractor.client = Mock()

        bad_response = Mock(parsed="Not a ClinicalSummary object")

        extractor.client.aio.models.generate_content = AsyncMock(return_value=bad_response)

        with pytest.raises(ExtractionError, match="Unexpected response shape"):
            await extractor.extract("Patient reported feeling anxious!", "sess_001")

    def test_gemini_extractor_init_missing_api_key(self):
        with pytest.raises(ConfigurationError, match="GEMINI_API_KEY not found"):
            GeminiExtractor(api_key="", model="gemini-1.5-flash")

    def test_gemini_extractor_init_missing_model(self):
        with pytest.raises(
            ConfigurationError, match="GEMINI_MODEL_FOR_EXTRACTION not found"
        ):
            GeminiExtractor(api_key="sk_test", model="")

    @pytest.mark.asyncio
    async def test_deepseek_extractor_extract_success(self):

        mock_summary = ClinicalSummary(
            patient_mood="anxious",
            exercises_completed=["balance exercises"],
            symptoms_mentioned=["anxiety"],
            next_steps="Continue daily exercises",
        )
        # choices.message.content
        mock_response = Mock(
            choices=[Mock(message=Mock(content=mock_summary.model_dump_json()))]
        )

        extractor = DeepSeekExtractor(
            api_key="sk_test",
            model="deepseek-chat",
            base_url="https://api.deepseek.com",
        )
        extractor.client = AsyncMock()
        extractor.client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await extractor.extract(
            "Patient reported feeling anxious.", "sess_001"
        )

        assert isinstance(result, ClinicalSummary)
        assert result.patient_mood == "anxious"

        response_shape_spec = f"respond in the following JSON format: {ClinicalSummary.model_json_schema()}"
        modified_system_prompt = "\n".join([system_prompt, response_shape_spec])

        extractor.client.chat.completions.create.assert_called_once_with(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": modified_system_prompt},
                {
                    "role": "user",
                    "content": "<transcript>Patient reported feeling anxious.</transcript>",
                },
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )

    @pytest.mark.asyncio
    async def test_deepseek_extractor_extract_wrong_type(self):
        extractor = DeepSeekExtractor(
            api_key="sk_test",
            model="deepseek-v4-flash",
            base_url="https://api.deepseek.com",
        )
        extractor.client = AsyncMock()

        bad_response = Mock()
        bad_response.choices = [Mock(message=Mock(content="Not a valid json"))]

        extractor.client.chat.completions.create = AsyncMock(return_value=bad_response)

        with pytest.raises(ExtractionError, match="Unexpected response shape"):
            await extractor.extract("Patient reported feeling anxious!", "sess_001")

    @pytest.mark.asyncio
    async def test_deepseek_extractor_extract_empty_response(self):
        extractor = DeepSeekExtractor(
            api_key="sk_test",
            model="deepseek-v4-flash",
            base_url="https://api.deepseek.com",
        )
        extractor.client = AsyncMock()

        bad_response = Mock()
        bad_response.choices = [Mock(message=Mock(content=None))]

        extractor.client.chat.completions.create = AsyncMock(return_value=bad_response)

        with pytest.raises(ExtractionError, match="Empty response"):
            await extractor.extract("Patient reported feeling anxious!", "sess_001")

    @pytest.mark.asyncio
    async def test_llama_extractor_extract_success(self, mocker):
        mock_summary = ClinicalSummary(
            patient_mood="anxious",
            exercises_completed=["balance exercises"],
            symptoms_mentioned=["anxiety"],
            next_steps="Continue daily exercises",
        )
        mock_response = {"message": {"content": mock_summary.model_dump_json()}}

        mock_ollama_client = AsyncMock()
        mock_ollama_client.__aenter__.return_value = mock_ollama_client
        mock_ollama_client.chat = AsyncMock(return_value=mock_response)
        mocker.patch("ollama.AsyncClient", return_value=mock_ollama_client)

        extractor = LlamaExtractor(
            model="llama3.2:3b", ollama_host="http://localhost:11434"
        )
        result = await extractor.extract(
            "Patient reported feeling anxious.", "sess_001"
        )

        assert isinstance(result, ClinicalSummary)
        assert result.patient_mood == "anxious"
        mock_ollama_client.chat.assert_awaited_once_with(
            model="llama3.2:3b",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": "<transcript>Patient reported feeling anxious.</transcript>",
                },
            ],
            format=ClinicalSummary.model_json_schema(),
        )

    def test_llama_extractor_init_missing_model(self):
        with pytest.raises(
            ConfigurationError, match="LLAMA_MODEL_FOR_EXTRACTION not found"
        ):
            LlamaExtractor(model="", ollama_host="http://localhost:11434")

    def test_abstract_base_class_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            ExtractionEngine()
