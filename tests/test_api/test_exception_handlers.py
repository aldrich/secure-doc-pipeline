from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from domain.error import AuthenticationError, ConfigurationError, EvaluationError, ExtractionError, ProviderError


def make_app_with_handler(exc_cls, status_code):
    from fastapi.responses import JSONResponse

    app = FastAPI()

    @app.exception_handler(exc_cls)
    async def handler(request, exc):
        return JSONResponse(status_code=status_code, content={"detail": str(exc)})

    @app.get("/raise")
    async def raise_error():
        raise exc_cls("test error")

    return app


class TestExceptionHandlers:
    @pytest.mark.parametrize("exc_cls,expected_status", [
        (ConfigurationError, 500),
        (AuthenticationError, 401),
        (ProviderError, 502),
        (ExtractionError, 422),
        (EvaluationError, 422),
    ])
    def test_custom_exception_returns_correct_status(self, exc_cls, expected_status):
        app = make_app_with_handler(exc_cls, expected_status)
        client = TestClient(app)
        response = client.get("/raise")
        assert response.status_code == expected_status
        assert response.json()["detail"] == "test error"

    def test_generic_exception_returns_500(self):
        app = FastAPI()

        @app.exception_handler(Exception)
        async def generic_handler(request, exc):
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=500, content={"detail": str(exc)})

        @app.get("/raise")
        async def raise_error():
            raise ValueError("unexpected")

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/raise")
        assert response.status_code == 500
        assert response.json()["detail"] == "unexpected"