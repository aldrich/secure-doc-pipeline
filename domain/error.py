# Server misconfigured
class ConfigurationError(Exception):
    pass

# Invalid or missing API key
class AuthenticationError(Exception):
    pass

# Upstream LLM provider failed
class ProviderError(Exception):
    pass

# Transcript could not be extracted into the required schema
class ExtractionError(Exception):
    pass

# Evaluation pipeline failed
class EvaluationError(Exception):
    pass