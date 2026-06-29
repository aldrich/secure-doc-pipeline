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

from evaluation_metrics import EvaluationMetrics
from pipeline_core import ClinicalSummary
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase
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
            actual_output=str(summary_data.model_dump()),
            context=[source_transcript]
        )
        
        logger.info(f"{test_case.model_dump_json=}")

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
    evaluator = get_evaluator(engine_type)
    
    logger.info(f"{summary_data=}")
    logger.info(f"{engine_type=}")
    logger.info(f"{source_transcript=}")
    
    metrics = await evaluator.evaluate(summary_data, source_transcript, session_id)
    
    logger.info(f"[Background] Evaluation succeeded for session: {session_id}")
    logger.info(f"Latency recorded: {metrics.latency_seconds}s")
    logger.info(f"Factual alignment check: {metrics.passed}")
