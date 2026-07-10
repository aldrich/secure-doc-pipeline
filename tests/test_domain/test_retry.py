import httpx
import pytest

from domain.error import ProviderError
from domain.retry import with_llm_retry


class TestWithLLMRetry:

    @pytest.mark.asyncio
    async def test_success_no_retry(self):
        call_count = 0

        @with_llm_retry(max_retries=2)
        async def fn():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = await fn()
        assert result == "ok"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_then_succeed(self):
        call_count = 0

        @with_llm_retry(max_retries=3, base_delay=0.01, max_delay=0.1)
        async def fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise httpx.TimeoutException("timed out")
            return "recovered"

        result = await fn()
        assert result == "recovered"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_exhaust_retries_raises_provider_error(self):
        call_count = 0

        @with_llm_retry(max_retries=2, base_delay=0.01, max_delay=0.1)
        async def fn():
            nonlocal call_count
            call_count += 1
            raise httpx.TimeoutException("always fails")

        with pytest.raises(ProviderError, match="LLM call failed after 3 attempts"):
            await fn()

        assert call_count == 3

    @pytest.mark.asyncio
    async def test_non_retryable_exception_passes_through(self):
        @with_llm_retry(max_retries=3, base_delay=0.01, max_delay=0.1)
        async def fn():
            raise ValueError("not retryable")

        with pytest.raises(ValueError, match="not retryable"):
            await fn()

    @pytest.mark.asyncio
    async def test_custom_is_retryable(self):
        call_count = 0

        def my_is_retryable(exc):
            return isinstance(exc, RuntimeError)

        @with_llm_retry(max_retries=2, base_delay=0.01, max_delay=0.1, is_retryable=my_is_retryable)
        async def fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("retry me")
            return "done"

        result = await fn()
        assert result == "done"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_httpx_connect_error_is_retryable(self):
        call_count = 0

        @with_llm_retry(max_retries=1, base_delay=0.01, max_delay=0.1)
        async def fn():
            nonlocal call_count
            call_count += 1
            raise httpx.ConnectError("connection refused")

        with pytest.raises(ProviderError):
            await fn()

        assert call_count == 2

    @pytest.mark.asyncio
    async def test_httpx_http_status_429_is_retryable(self):
        call_count = 0

        @with_llm_retry(max_retries=1, base_delay=0.01, max_delay=0.1)
        async def fn():
            nonlocal call_count
            call_count += 1
            request = httpx.Request("GET", "http://example.com")
            response = httpx.Response(429, request=request)
            raise httpx.HTTPStatusError("rate limit", request=request, response=response)

        with pytest.raises(ProviderError):
            await fn()

        assert call_count == 2

    @pytest.mark.asyncio
    async def test_httpx_http_status_400_not_retryable(self):
        call_count = 0

        @with_llm_retry(max_retries=1, base_delay=0.01, max_delay=0.1)
        async def fn():
            nonlocal call_count
            call_count += 1
            request = httpx.Request("GET", "http://example.com")
            response = httpx.Response(400, request=request)
            raise httpx.HTTPStatusError("bad request", request=request, response=response)

        with pytest.raises(httpx.HTTPStatusError):
            await fn()

        assert call_count == 1