"""
相关性评估——回答是否切题

算法：用 embedding 直接算 问题 和 回答 的语义相似度。
不再需要手动分词、去停用词、算关键词覆盖率。
"""

from app.evaluator.base import BaseMetric
from app.models.schemas import MetricResult
from app.embedding import semantic_similarity


class RelevanceEvaluator(BaseMetric):

    def evaluate(self, question, context, answer, expected_key_points=None):
        if not question or not question.strip():
            return MetricResult(score=0.0, details={"error": "问题为空"})

        sim = semantic_similarity(question, answer)

        return MetricResult(
            score=round(sim, 4),
            details={"semantic_similarity": round(sim, 4)},
        )
