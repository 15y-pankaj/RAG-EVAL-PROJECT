from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

import os

load_dotenv()

# --- Test case 1: a good answer (should PASS) ---
case_1 = LLMTestCase(
    input="What is the capital of France?",
    actual_output="The capital of France is Paris.",
)

# --- Test case 2: an off-topic answer (should FAIL) ---
case_2 = LLMTestCase(
    input=" France is famous for ?",
    actual_output="France is a beautiful country famous for its food and wine.",
)


from deepeval.models import GeminiModel
model = GeminiModel(
    model="gemini-3.5-flash-lite",
    api_key=os.environ.get("GOOGLE_API_KEY"),
    temperature=0,
     )

# from deepeval.models import GrokModel
# model = GrokModel(
#     model="grok-4.1",
#     api_key=os.environ.get("GROK_API_KEY"),
#     temperature=0
# )
# --- One metric, judged by an LLM (pinned for reproducibility) ---
metric = AnswerRelevancyMetric(threshold=0.7, model=model, include_reason=True)
#metric = AnswerRelevancyMetric(threshold=0.7, model="gpt-4.1", include_reason=True)

# --- Run BOTH cases through the metric, with a printed report ---
evaluate(test_cases=[case_1, case_2], metrics=[metric])