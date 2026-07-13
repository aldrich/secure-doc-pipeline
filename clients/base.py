import logging
from abc import ABC, abstractmethod

from pydantic import BaseModel

from domain.retry import with_llm_retry

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    @abstractmethod
    async def generate_structured(
        self,
        model: str,
        system_prompt: str,
        user_content: str,
        response_schema: type[BaseModel],
    ) -> BaseModel:

        pass

    @abstractmethod
    def get_name(self) -> str:

        pass


class LLMEngine:
    def __init__(
        self,
        client: LLMClient,
        model: str,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
        retry_max_delay: float = 30.0,
    ):
        self._client = client
        self._model = model
        self._retry_decorator = with_llm_retry(
            max_retries=max_retries,
            base_delay=retry_base_delay,
            max_delay=retry_max_delay,
        )

    @property
    def model(self) -> str:
        return self._model

    @property
    def client(self) -> LLMClient:
        return self._client
