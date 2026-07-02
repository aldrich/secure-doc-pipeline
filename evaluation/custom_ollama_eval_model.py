import ollama

from deepeval.models import DeepEvalBaseLLM
class CustomOllamaEvalModel(DeepEvalBaseLLM):
    def __init__(self, model_name="llama3.1:8b"):
        self.__class__.__abstractmethods__ = frozenset()
        self.model_name = model_name

    # Added *args and **kwargs to perfectly satisfy any base class abstract signature
    def load_model(self):
        return self

    def get_model_name(self) -> str:
        return self.model_name

    def generate(self, prompt: str) -> str:
        res = ollama.generate(model=self.model_name, prompt=prompt, format="json")
        if isinstance(res, dict):
            return res.get("response", "")
        return getattr(res, "response", "")

    async def a_generate(self, prompt: str) -> str:
        res = await ollama.AsyncClient().generate(model=self.model_name, prompt=prompt, format="json")
        if isinstance(res, dict):
            return res.get("response", "")
        return getattr(res, "response", "")