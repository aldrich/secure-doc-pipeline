import asyncio
import logging
import sys
import time
import ollama

from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import DeepEvalBaseLLM
from evaluation_metrics import EvaluationMetrics
from pipeline_core import ClinicalSummary

logging.basicConfig(level=logging.INFO,
                    stream=sys.stdout,
                    format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# # Initialize our explicit judge model globally to save memory
# JUDGE_MODEL = OllamaModel(
#     model="llama3.2:3b",
#     base_url=OLLAMA_URL,
#     temperature=0.0
# )
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
    
async def run_evaluation_llama(summary_data: ClinicalSummary, source_transcript: str, session_id: str = "") -> EvaluationMetrics:
    """
    Background worker function. Runs the evaluation and logs to a data store
    without keeping the HTTP connection hanging.
    """
    logger.info(f"[Job {session_id}] Starting evaluation in the background...")

    test_case = LLMTestCase(
        input=source_transcript,
        actual_output=str(summary_data.model_dump()),
        context=[source_transcript]
    )
    
    start_time = time.perf_counter()

    # Create your safe evaluation model instance
    eval_model = CustomOllamaEvalModel(model_name="llama3:8b")

    # Measure using our local judge
    metric = HallucinationMetric(threshold=0.5, model=eval_model)
    metric.measure(test_case)

    # Simulate Database Storage/Flagging Logic
    logger.info(f"[Job {session_id}] Eval complete. Score: {metric.score}")
    logger.info(f"[Job Score: {metric.score}] Reason for score: {metric.reason}")

    eval_metric = EvaluationMetrics(
        passed=metric.is_successful(),
        faithfulness_score=metric.score if metric.score is not None else 0.0,
        latency_seconds=0.0,
        unsupported_exercises=[], # TODO: support this instead of metric.reason
        omitted_symptoms=[],
    )
    
    elapsed = round(time.perf_counter() - start_time, 2)
    eval_metric.latency_seconds = elapsed
    
    return eval_metric
    
async def main():
    
    source_transcript="""
    Patient initially stated, "I didn't do any physical therapy or movement work over the weekend at all." 
    However, later in the review, when prompted about specific logs, they recalled and corrected themselves: 
    "Oh, wait, I actually spent about 20 minutes doing my balance and gait exercises on Saturday afternoon 
    with the home nurse." They reported feeling stable throughout.
    """
    
    sample_summary = ClinicalSummary(
        patient_mood="apologetic",
        exercises_completed=["balance", "gait"],
        symptoms_mentioned=["nausea"],
        next_steps=""
    )

    logger.info(f"Triggering evaluation pipeline...")

    metrics = await run_evaluation_llama(sample_summary, source_transcript)

    logger.info("Pipeline Execution Metrics:")
    logger.info(f"Latency:     {metrics.latency_seconds}s")
    logger.info(f"Score:       {metrics.faithfulness_score}")
    logger.info(f"Status:      {'Passed' if metrics.passed else 'Flagged for Review'}")

if __name__ == "__main__":
    asyncio.run(main())