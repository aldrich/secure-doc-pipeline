import asyncio
import logging
import random
from functools import wraps
from typing import Awaitable, Callable, ParamSpec, TypeVar

import httpx

from domain.error import ProviderError

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")

RetryableCheck = Callable[[Exception], bool]


def _default_is_retryable(exc: Exception) -> bool:
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, httpx.ConnectError):
        return True
    if isinstance(exc, httpx.RemoteProtocolError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in {429, 500, 502, 503, 504}
    return False


def with_llm_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    is_retryable: RetryableCheck = _default_is_retryable,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    def decorator(fn: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @wraps(fn)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exc: Exception | None = None
            for attempt in range(max_retries + 1):
                try:
                    return await fn(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    if attempt < max_retries and is_retryable(exc):
                        delay = min(base_delay * (2 ** attempt) + random.uniform(0, 0.5 * base_delay), max_delay)
                        logger.warning(
                            "llm_retry",
                            extra={
                                "attempt": attempt + 1,
                                "max_retries": max_retries,
                                "delay": round(delay, 2),
                                "error": str(exc),
                                "fn": fn.__qualname__,
                            },
                        )
                        await asyncio.sleep(delay)
                    elif not is_retryable(exc):
                        raise
            raise ProviderError(f"LLM call failed after {max_retries + 1} attempts") from last_exc
        return wrapper
    return decorator
