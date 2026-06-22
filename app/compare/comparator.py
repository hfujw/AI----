from app.harness.runner import TestRunner
from app.models.schemas import CompareRequest, CompareResponse


class ModelComparator:
    def __init__(self, mode="fast"):
        self.runner = TestRunner(mode=mode)

    def compare(self, request: CompareRequest) -> CompareResponse:
        results = {}
        for model, answer in request.answers.items():
            results[model] = self.runner.run_single(
                request.question, request.context, answer
            )
        ranking = sorted(
            results, key=lambda m: results[m].overall_score, reverse=True
        )
        return CompareResponse(
            question=request.question, results=results, ranking=ranking
        )
