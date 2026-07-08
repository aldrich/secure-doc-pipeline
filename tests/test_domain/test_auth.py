import os

from fastapi import HTTPException
import pytest

from domain.auth import verify_api_key


class TestVerifyApiKey:
    async def test_valid_api_key(self, monkeypatch):
        monkeypatch.setenv("API_KEY", "valid-key")
        result = await verify_api_key(api_key="valid-key")
        assert result is None

    async def test_missing_api_key_header(self, monkeypatch):
        monkeypatch.setenv("API_KEY", "valid-key")
        with pytest.raises(HTTPException) as exc:
            await verify_api_key(api_key=None)
        assert exc.value.status_code == 401

    async def test_invalid_api_key(self, monkeypatch):
        monkeypatch.setenv("API_KEY", "valid-key")
        with pytest.raises(HTTPException) as exc:
            await verify_api_key(api_key="wrong-key")
        assert exc.value.status_code == 401

    async def test_missing_env_var(self, monkeypatch):
        monkeypatch.delenv("API_KEY", raising=False)
        with pytest.raises(HTTPException) as exc:
            await verify_api_key(api_key="some-key")
        assert exc.value.status_code == 401
        assert "Invalid or missing" in exc.value.detail