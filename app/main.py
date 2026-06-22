import logging
from fastapi import FastAPI
from app.config import LOG_LEVEL
from app.models.schemas import (
    EvaluateRequest, EvaluateResponse,
    BatchEvaluateRequest, BatchEvaluateResponse,
    CompareRequest, CompareResponse,
)
from app.harness.runner import TestRunner, DIMENSIONS
from app.harness.loader import TestCase

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("ai_eval")

app = FastAPI(title="AI 评测框架", version="0.2.0")
runner = TestRunner()


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate(request: EvaluateRequest):
    return runner.run_single(
        request.question, request.context, request.answer,
        request.expected_key_points,
    )


@app.post("/evaluate/batch", response_model=BatchEvaluateResponse)
def evaluate_batch(request: BatchEvaluateRequest):
    cases = [
        TestCase(
            question=c.question, context=c.context, answer=c.answer,
            expected_key_points=c.expected_key_points, thresholds={},
        )
        for c in request.cases
    ]
    return runner.run_batch(cases)


@app.post("/compare", response_model=CompareResponse)
def compare(request: CompareRequest):
    results = {}
    for model, answer in request.answers.items():
        results[model] = runner.run_single(
            request.question, request.context, answer
        )
    ranking = sorted(
        results, key=lambda m: results[m].overall_score, reverse=True
    )
    return CompareResponse(
        question=request.question, results=results, ranking=ranking
    )


@app.get("/health")
def health():
    return {"status": "ok", "mode": runner.mode, "dimensions": DIMENSIONS}
