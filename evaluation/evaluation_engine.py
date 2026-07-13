import logging

from clients.base import LLMEngine
from domain.error import EvaluationError
from prompts.evaluation import get_prompt, system_prompt
from schemas.clinical_summary import ClinicalSummary
from schemas.evaluation_metrics import SummaryEvaluation

logger = logging.getLogger(__name__)


class EvaluationEngine(LLMEngine):
    async def evaluate(
        self, summary_data: ClinicalSummary, source_transcript: str, session_id: str
    ):
        logger.info("evaluation_started")  # ...
        return await self._retry_decorator(self._do_evaluate)(
            summary_data, source_transcript, session_id
        )

    async def _do_evaluate(
        self, summary_data: ClinicalSummary, source_transcript: str, session_id: str
    ):
        prompt = get_prompt(source_transcript, summary_data)
        result = await self._client.generate_structured(
            model=self._model,
            system_prompt=system_prompt,
            user_content=prompt,
            response_schema=SummaryEvaluation,
        )
        if not isinstance(result, SummaryEvaluation):
            raise EvaluationError(f"Unexpected response shape: {type(result)}")
        return result
