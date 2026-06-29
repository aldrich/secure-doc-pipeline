import uuid
import os
import time
import logging
import asyncio

from typing import List
from google import genai
from google.genai import types
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from pipeline_core import ClinicalSummary
from pipeline_core import extract_structured_data

class EvaluationMetrics(BaseModel):
    passed: bool = Field(description="True if the extracted metrics are completely grounded in the transcript without any fabricated details.")
    faithfulness_score: float = Field(description="Score between 0.0 and 1.0. A 1.0 means perfect alignment; deduction occurs for hallucinations or missed items.")
    unsupported_exercises: List[str] = Field(description="Exercises listed in the summary that the patient never actually performed or mentioned in the transcript.")
    omitted_symptoms: List[str] = Field(description="Critical symptoms or difficulties vocalized by the patient in the transcript that failed to make it into the summary.")
    latency_seconds: float = Field(description="Total duration of the evaluation execution step.")

api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is missing from the container environment!")

client = genai.Client(api_key=api_key)

async def run_evaluation_async(summary_data: ClinicalSummary, source_transcript: str, session_id: str = "") -> EvaluationMetrics:
    """
    Evaluates a generated ClinicalSummary against its original source text transcript
    to guarantee clinical auditing data integrity.
    """
    start_time = time.perf_counter()

    # Normally, you'd pull the source transcript using the session_id from your database.
    # For this illustration, we assume it is accessible or passed alongside your context.
    # source_transcript = " [Retrieve original raw transcript text via session_id or state here] "

    print(f"Running evaluation in background for session id {session_id}")

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

    response = await client.aio.models.generate_content(
        model='gemini-2.5-flash',
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

app = FastAPI(title="Secure Clinical Documentation Pipeline")

logger = logging.getLogger("pipeline")

# Input contract for incoming requests to our API
class SessionRequest(BaseModel):
    transcript: str

@app.post("/api/v1/process-session", status_code=202)
async def process_session(payload: SessionRequest, background_tasks: BackgroundTasks):

    session_id = str(uuid.uuid4())[:8]
    print(f"\n📥 Received session processing request. Assigned Job ID: {session_id}.")

    try:
        structured_output = extract_structured_data(payload.transcript)

        metrics = await run_evaluation_async(
            summary_data=structured_output,
            source_transcript=payload.transcript,
            session_id=session_id
        )

        print(f"✅ [Background] Evaluation succeeded for session: {session_id}")
        print(f"⏱️ Latency recorded: {metrics.latency_seconds}s")
        print(f"🎯 Factual alignment check: {metrics.passed}")

        # TODO: Write to database / trigger webhook back to n8n here...

        return {
            "status": "processing_verification",
            "session_id": session_id,
            "data": structured_output
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline processing failed: {str(e)}")


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

    print(f"Triggering evaluation pipeline...")

    metrics = await run_evaluation_async(sample_summary, source_transcript)

    print("\n📊 Pipeline Execution Metrics:")
    print(f"⏱️ Latency:     {metrics.latency_seconds}s")
    print(f"🎯 Score:       {metrics.faithfulness_score}")
    print(f"✅ Status:      {'Passed' if metrics.passed else 'Flagged for Review'}")

if __name__ == "__main__":
    asyncio.run(main())