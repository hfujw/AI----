from abc import ABC, abstractmethod
from app.evaluator.base import BaseMetric
from app.models.schemas import MetricResult


class LLMJudge(BaseMetric):
    """一次 API 调用评估四个维度"""

    @abstractmethod
    def judge(self, question: str, context: str, answer: str,
              expected_key_points: list[str] | None = None) -> dict[str, MetricResult]:
        pass

    def evaluate(self, question, context, answer, expected_key_points=None):
        results = self.judge(question, context, answer, expected_key_points)
        overall = sum(r.score for r in results.values()) / max(len(results), 1)
        return MetricResult(score=round(overall, 4), details={"per_dimension": results})
