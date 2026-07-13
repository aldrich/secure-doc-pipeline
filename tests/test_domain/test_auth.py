import pytest
from fastapi import HTTPException

from domain.auth import verify_api_key
from domain.settings import settings


class TestVerifyApiKey:
    async def test_valid_api_key(self, mocker):
        mocker.patch.object(settings, "api_key", "valid-key")
        result = await verify_api_key(api_key="valid-key")
        assert result is None

    async def test_missing_api_key_header(self, mocker):
        mocker.patch.object(settings, "api_key", "valid-key")
        with pytest.raises(HTTPException) as exc:
            await verify_api_key(api_key="")
        assert exc.value.status_code == 401

    async def test_invalid_api_key(self, mocker):
        mocker.patch.object(settings, "api_key", "valid-key")
        with pytest.raises(HTTPException) as exc:
            await verify_api_key(api_key="wrong-key")
        assert exc.value.status_code == 401

    async def test_missing_env_var(self, mocker):
        mocker.patch.object(settings, "api_key", "")
        with pytest.raises(HTTPException) as exc:
            await verify_api_key(api_key="some-key")
        assert exc.value.status_code == 401
        assert "Invalid or missing" in exc.value.detail
