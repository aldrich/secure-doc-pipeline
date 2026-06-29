from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase
# Import the explicit Ollama model wrapper from deepeval
from deepeval.models import OllamaModel

OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# 1. Prepare the inputs for the automated judge
source_transcript = """
The patient arrived a bit anxious today but warmed up during the cognitive tasks. 
We spent the first twenty minutes working on the spatial memory grid exercises and then 
moved on to vocal articulation drills. She mentioned experiencing some mild headaches 
over the weekend and a bit of frustration with word-finding when fatigued. 
For next time, I told her to practice the articulation sheet twice a day and keep track 
of when the headaches occur.
"""

good_summary = """
Patient was anxious initially but improved. Completed spatial memory grid and vocal articulation exercises. 
Complained of mild headaches and word-finding issues when tired. Ordered to practice articulation sheet 
twice a day and log headaches.
"""

# bad_summary = """
# Patient completed advanced spinal mobility physical therapy. Doctor prescribed 400mg Ibuprofen 
# for severe back spasms and told them to return in two weeks.
# """

bad_summary = """
Anxious at first but improved. Completed spatial memory grid and vocal articulation exercises. 
Complained of headaches when tired. Ordered to practice articulation sheet twice a day and log headaches.
"""

def run_evaluation(summary_to_test: str):
    
    # 2. Define the Test Case
    test_case = LLMTestCase(
        input=source_transcript,
        actual_output=summary_to_test,
        context=[source_transcript] 
    )

    # 3. Explicitly define our local judge model
    # Use 'llama3.2:3b' if your system required the tag earlier
    judge_model = OllamaModel(
        model="llama3.2:3b",
        base_url=OLLAMA_URL,
        temperature=0.0
    )

    # 4. Initialize the Hallucination Metric passing our explicit model
    metric = HallucinationMetric(threshold=0.5, model=judge_model)

    # 5. Measure the test case
    metric.measure(test_case)
    
    print(f"\n📊 Evaluation Score (Lower is better for hallucination): {metric.score}")
    print(f"Reasoning: {metric.reason}")
    print(f"Status: {'✅ PASSED' if metric.is_successful() else '❌ FAILED'}")

if __name__ == "__main__":
    # print("🧠 Running Eval on ACCURATE Summary...")
    # run_evaluation(good_summary)
    
    print("\n" + "="*40 + "\n")
    
    print("🚨 Running Eval on HALLUCINATED Summary...")
    run_evaluation(bad_summary)