from typing import List
from pydantic import BaseModel, Field

class EvaluationMetrics(BaseModel):
    passed: bool = Field(description="True if the extracted metrics are completely grounded in the transcript without any fabricated details.")
    faithfulness_score: float = Field(description="Score between 0.0 and 1.0. A 1.0 means perfect alignment; deduction occurs for hallucinations or missed items.")
    unsupported_exercises: List[str] = Field(description="Exercises listed in the summary that the patient never actually performed or mentioned in the transcript.")
    omitted_symptoms: List[str] = Field(description="Critical symptoms or difficulties vocalized by the patient in the transcript that failed to make it into the summary.")
    latency_seconds: float = 0.0

class EvaluationMetrics2(BaseModel):
    passed: bool
    exercises_score: float
    symptoms_score: float
    exercises_reason: str
    symptoms_reason: str
    latency_seconds: float