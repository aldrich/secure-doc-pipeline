import asyncio
import logging
import os
import sys
import time
from typing import Literal
from abc import ABC, abstractmethod

from google import genai
from google.genai import types
import ollama

from evaluation_metrics import EvaluationMetrics, EvaluationMetrics2
from pipeline_core import ClinicalSummary
from deepeval.metrics import GEval, HallucinationMetric

from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.models import DeepEvalBaseLLM

logging.basicConfig(level=logging.INFO,
                    stream=sys.stdout,
                    format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class EvaluationEngine(ABC):
    """Abstract base class for evaluation engines."""

    @abstractmethod
    async def evaluate(self, summary_data: ClinicalSummary, source_transcript: str, session_id: str = "") -> EvaluationMetrics:
        pass

class CustomEvaluator:
    def __init__(self, model_name: str):
        self.model_name = model_name

        # 1. Metric dedicated strictly to hunting hallucinated exercises
        self.exercise_metric = GEval(
            name="Exercise Hallucination Detector",
            criteria=(
                "Determine if the 'Exercises Logged' in the actual output contains ANY "
                "physical or cognitive exercises that were NOT explicitly mentioned or performed "
                "in the source transcript."
            ),
            evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.RETRIEVAL_CONTEXT],
            evaluation_steps=[
                "Isolate the list of exercises provided in the actual output summary.",
                "Cross-examine each exercise against the original source transcript timeline.",
                "If an exercise is listed in the output but was never performed or mentioned in the transcript, classify it as a hallucination.",
                "Deduct points heavily for every hallucinated exercise found. Score 1.0 only for perfect factual alignment."
            ],
            # Pass your custom Ollama model instance here if using self-hosted
            model=CustomOllamaEvalModel(model_name=model_name)
        )

        # 2. Metric dedicated strictly to finding omitted symptoms
        self.symptom_metric = GEval(
            name="Symptom Omission Detector",
            criteria=(
                "Determine if the 'Symptoms Logged' section failed to capture critical symptoms, "
                "pains, or cognitive difficulties explicitly vocalized by the patient in the source transcript."
            ),
            evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.RETRIEVAL_CONTEXT],
            evaluation_steps=[
                "Scan the source transcript and list all physical/cognitive complaints or symptoms vocalized by the patient.",
                "Check if every identified symptom is accurately reflected inside the actual output summary.",
                "If a major symptom or critical pain complaint was omitted, deduct points aggressively.",
                "Score 1.0 if all symptoms from the source context are safely accounted for."
            ],
            model=CustomOllamaEvalModel(model_name=model_name)
        )

    async def evaluate(self, summary_data: ClinicalSummary, source_transcript: str, session_id: str = "") -> EvaluationMetrics2:
        logger.info(f"Spawning dual G-Eval pipeline for session: {session_id}")
        start_time = time.perf_counter()

        # Build your corrected test case mapping using retrieval_context
        test_case = LLMTestCase(
            input=f"Verify documentation integrity for session {session_id}",
            actual_output=summary_data.model_dump_json(indent=2),
            retrieval_context=[source_transcript]
        )

        # Run both evaluations concurrently to optimize pipeline background speeds
        # (Assuming your CustomOllamaEvalModel handles concurrent a_measure calls smoothly)
        import asyncio
        await asyncio.gather(
            self.exercise_metric.a_measure(test_case),
            self.symptom_metric.a_measure(test_case)
        )

        elapsed = round(time.perf_counter() - start_time, 2)

        # A pipeline passes only if both sub-dimensions meet your safety threshold
        
        if self.exercise_metric.score is None or self.symptom_metric.score is None:
            logger.error("One or more evaluation metrics failed to compute a score.")
            pipeline_passed = False
        else:
            threshold = 0.67
            pipeline_passed = (
                self.exercise_metric.score >= threshold and
                self.symptom_metric.score >= threshold
            )

        return EvaluationMetrics2(
            passed=pipeline_passed,
            exercises_score=self.exercise_metric.score or 0.0,
            symptoms_score=self.symptom_metric.score or 0.0,
            exercises_reason=self.exercise_metric.reason or "No reason provided.",
            symptoms_reason=self.symptom_metric.reason or "No reason provided.",
            latency_seconds=elapsed
        )


class GeminiEvaluator(EvaluationEngine):
    def __init__(self, api_key: str, model: str):
        if not api_key:
            logger.warning("GEMINI_API_KEY not found from environment.")
            raise ValueError("GEMINI_API_KEY is missing from the container environment!")
        if not model:
            logger.warning("GEMINI_MODEL not found from environment.")
            raise ValueError("GEMINI_MODEL is missing from the container environment!")

        self.client = genai.Client(api_key=api_key)
        self.model = model

    async def evaluate(self, summary_data: ClinicalSummary, source_transcript: str, session_id: str = "") -> EvaluationMetrics:
        start_time = time.perf_counter()

        logger.info(f"Running evaluation in background for session id {session_id}")
        logger.info(f"using model={self.model}")

        system_instruction = """
        You are an AI Medical Auditor specialized in validating clinical notes. Your sole task is to compare
        a 'Raw Session Transcript' against an extracted 'Clinical Summary structure' to verify factual accuracy.

        Pay close attention to:
        1. Exercises: Did the summary list exercises that weren't actually executed?
        2. Symptoms: Did the summary forget to document physical or cognitive complaints explicit in the transcript?

        Strictly output your response matching the requested JSON Schema configuration.
        """

        prompt = f"""
        Please perform an audit validation for Session ID: {session_id}

        --- ORIGINAL SOURCE TRANSCRIPT ---
        {source_transcript}

        --- EXTRACTED SUMMARY DATA TO EVALUATE ---
        Patient Mood: {summary_data.patient_mood}
        Exercises Logged: {summary_data.exercises_completed}
        Symptoms Logged: {summary_data.symptoms_mentioned}
        Next Steps Plan: {summary_data.next_steps}
        """

        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=EvaluationMetrics,
                temperature=0.0,
            )
        )

        raw_parsed = response.parsed
        if not isinstance(raw_parsed, EvaluationMetrics):
            raise ValueError(f"Unexpected response shape: {type(raw_parsed)}")
        metrics: EvaluationMetrics = raw_parsed

        elapsed = round(time.perf_counter() - start_time, 2)
        metrics.latency_seconds = elapsed
        return metrics


class CustomOllamaEvalModel(DeepEvalBaseLLM):
    """Custom Ollama model for deepeval HallucinationMetric."""
    def __init__(self, model_name: str):
        self.__class__.__abstractmethods__ = frozenset()
        self.model_name = model_name

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
class LlamaEvaluator(EvaluationEngine):
    def __init__(self, model: str):
        if not model:
            logger.warning("LLAMA_MODEL not found from environment.")
            raise ValueError("LLAMA_MODEL is missing from environment!")
        self.model = model

    async def evaluate(self, summary_data: ClinicalSummary, source_transcript: str, session_id: str = "") -> EvaluationMetrics:
        logger.info(f"Running evaluation in background for session id {session_id}")
        logger.info(f"using model={self.model}")

        test_case = LLMTestCase(
            input=source_transcript,
            actual_output=summary_data.model_dump_json(indent=2),
            context=[source_transcript]
        )

        logger.debug(f"{test_case.model_dump_json=}")

        start_time = time.perf_counter()

        eval_model = CustomOllamaEvalModel(model_name=self.model)
        metric = HallucinationMetric(threshold=0.5, model=eval_model)
        await metric.a_measure(test_case)

        logger.info(f"Eval complete. Score: {metric.score}")
        logger.info(f"Reason for score: {metric.reason}")

        elapsed = round(time.perf_counter() - start_time, 2)

        return EvaluationMetrics(
            passed=metric.is_successful(),
            faithfulness_score=metric.score if metric.score is not None else 0.0,
            latency_seconds=elapsed,
            unsupported_exercises=[],
            omitted_symptoms=[],
        )

def get_evaluator(engine: Literal["gemini", "llama"]) -> EvaluationEngine:
    """Factory function to get the appropriate evaluator based on configuration."""
    if engine == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY")
        model = os.environ.get("GEMINI_MODEL")
        return GeminiEvaluator(api_key, model)
    elif engine == "llama":
        model = os.environ.get("LLAMA_MODEL")
        return LlamaEvaluator(model)
    else:
        raise ValueError(f"Unknown evaluation engine: {engine}. Choose 'gemini' or 'llama'.")


async def run_evaluation(summary_data: ClinicalSummary, source_transcript: str, session_id: str = ""):
    """Unified evaluation function using the configured engine."""
    engine_type = os.environ.get("EVAL_ENGINE", "llama").lower()

    if engine_type == "custom":
        logger.info("Using custom metrics")
        # model = os.environ.get("LLAMA_MODEL")
        model = "llama3.2:3b"
        evaluator = CustomEvaluator(model_name=model or "")
        metrics = await evaluator.evaluate(summary_data=summary_data, source_transcript=source_transcript, session_id=session_id)

        logger.info(f"[Background] Evaluation succeeded for session: {session_id}")
        logger.info(f"exercises score: {metrics.exercises_score}. Reason: {metrics.exercises_reason}")
        logger.info(f"symptoms score: {metrics.symptoms_score}. Reason: {metrics.symptoms_reason}")
        logger.info(f"Latency recorded: {metrics.latency_seconds}s")
        logger.info(f"Factual alignment check: {metrics.passed}")
        return

    evaluator = get_evaluator(engine_type)

    logger.debug(f"{summary_data=}")
    logger.debug(f"{engine_type=}")
    logger.debug(f"{source_transcript=}")

    metrics = await evaluator.evaluate(summary_data, source_transcript, session_id)

    logger.info(f"[Background] Evaluation succeeded for session: {session_id}")
    logger.info(f"Latency recorded: {metrics.latency_seconds}s")
    logger.info(f"Factual alignment check: {metrics.passed}")
