from clients.base import LLMEngine
from domain.error import ExtractionError
from prompts.extraction import system_prompt
from schemas.clinical_summary import ClinicalSummary


class ExtractionEngine(LLMEngine):
    async def extract(self, source_transcript: str, session_id: str) -> ClinicalSummary:
        # logger.info("extraction_started", extra={"engine": "openai", "model": self.model, "session_id": session_id})
        return await self._retry_decorator(self._do_extract)(
            source_transcript, session_id
        )

    async def _do_extract(
        self, source_transcript: str, session_id: str
    ) -> ClinicalSummary:
        # logger.info("extraction_started", ...)
        result = await self._client.generate_structured(
            model=self._model,
            system_prompt=system_prompt,
            user_content=f"<transcript>{source_transcript}</transcript>",
            response_schema=ClinicalSummary,
        )
        if not isinstance(result, ClinicalSummary):
            raise ExtractionError(f"Unexpected response shape: {type(result)}")

        return result
