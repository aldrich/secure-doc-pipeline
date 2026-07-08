import pytest

from domain.error import (
    AuthenticationError,
    ConfigurationError,
    EvaluationError,
    ExtractionError,
    ProviderError,
)


class TestCustomExceptions:
    @pytest.mark.parametrize("exc_cls,expected_msg", [
        (ConfigurationError, "Server misconfigured"),
        (AuthenticationError, "Invalid API key"),
        (ProviderError, "Upstream failed"),
        (ExtractionError, "Could not extract"),
        (EvaluationError, "Evaluation failed"),
    ])
    def test_message_propagation(self, exc_cls, expected_msg):
        exc = exc_cls(expected_msg)
        assert str(exc) == expected_msg

    def test_all_are_exception_subclasses(self):
        assert issubclass(ConfigurationError, Exception)
        assert issubclass(AuthenticationError, Exception)
        assert issubclass(ProviderError, Exception)
        assert issubclass(ExtractionError, Exception)
        assert issubclass(EvaluationError, Exception)

    def test_can_be_raised_and_caught(self):
        for exc_cls in [ConfigurationError, AuthenticationError, ProviderError, ExtractionError, EvaluationError]:
            try:
                raise exc_cls("test")
            except exc_cls as e:
                assert str(e) == "test"