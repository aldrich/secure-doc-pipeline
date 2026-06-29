import os
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase
import ollama
from deepeval.models import DeepEvalBaseLLM, OllamaModel

from pipeline_core import ClinicalSummary

OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Initialize our explicit judge model globally to save memory
JUDGE_MODEL = OllamaModel(
    model="llama3.2:3b",
    base_url=OLLAMA_URL,
    temperature=0.0
)

class CustomOllamaEvalModel(DeepEvalBaseLLM):
    def __init__(self, model_name="llama3:8b"):
        self.__class__.__abstractmethods__ = frozenset()
        self.model_name = model_name

    # Added *args and **kwargs to perfectly satisfy any base class abstract signature
    def load_model(self, *args, **kwargs):
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
    
def async_eval_and_store(session_id: str, transcript: str, structured_data: ClinicalSummary):
    """
    Background worker function. Runs the evaluation and logs to a data store
    without keeping the HTTP connection hanging.
    """
    print(f"🛠️ [Job {session_id}] Starting evaluation in the background...")

    # 1. Map data to DeepEval
    test_case = LLMTestCase(
        input=transcript,
        # Flatten our validated pydantic fields back into text for comparison
        actual_output=str(structured_data.model_dump()),
        context=[transcript]
    )

    # Create your safe evaluation model instance
    eval_model = CustomOllamaEvalModel(model_name="llama3:8b")

    # 2. Measure using our local judge
    metric = HallucinationMetric(threshold=0.5, model=eval_model)
    metric.measure(test_case)

    # 3. Simulate Database Storage/Flagging Logic
    print(f"📊 [Job {session_id}] Eval complete. Score: {metric.score}")
    print(f"📊 [Job Score: {metric.score}] Reason for score: {metric.reason}")

    if metric.is_successful():
        print(f"💾 [Job {session_id}] STATUS: PASSED. Commit entries cleanly to database.")
    else:
        print(f"🚨 [Job {session_id}] STATUS: FAILED. Flagging summary for manual clinician review!")
        print(f"⚠️ Reason: {metric.reason}")