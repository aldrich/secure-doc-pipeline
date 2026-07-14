from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from domain.models import Evaluation
from schemas.evaluation_metrics import SummaryEvaluation


class EvaluationRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self._session_factory = session_factory

    async def store(
        self, session_id: str, metrics: SummaryEvaluation, model: str, latency: float
    ) -> None:

        evaluation = Evaluation(
            session_id=session_id,
            score=metrics.score,
            faithful=metrics.faithful,
            reasoning=metrics.reasoning,
            model=model,
            latency_seconds=latency,
            unsupported_claims=[c.model_dump() for c in metrics.unsupported_claims],
            omitted_information=[c.model_dump() for c in metrics.omitted_information],
            contradictions=[c.model_dump() for c in metrics.contradictions],
        )

        async with self._session_factory() as session:
            session.add(evaluation)
            await session.commit()
