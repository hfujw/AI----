"""
完整性评估——回答是否覆盖了预期关键点

算法（两阶段）：
1. 如果没定义关键点 → 默认满分
2. 先子串匹配：关键点是否直接出现在回答中
3. 子串没命中 → embedding 语义匹配（"感到恶心想吐" ≈ "恶心"）
4. 得分 = 覆盖数 / 总数
"""

from app.evaluator.base import BaseMetric
from app.models.schemas import MetricResult
from app.config import COMPLETENESS_THRESHOLD
from app.embedding import semantic_similarity


class CompletenessEvaluator(BaseMetric):

    def evaluate(self, question, context, answer, expected_key_points=None):
        if not expected_key_points:
            return MetricResult(score=1.0, details={"note": "未定义关键点，默认满分"})

        hits = []
        misses = []
        for kp in expected_key_points:
            # 阶段 1：子串匹配
            if kp.lower() in answer.lower():
                hits.append({"key_point": kp, "method": "substring"})
            else:
                # 阶段 2：embedding 语义匹配
                sim = semantic_similarity(kp, answer)
                if sim >= COMPLETENESS_THRESHOLD:
                    hits.append({"key_point": kp, "similarity": round(sim, 4), "method": "semantic"})
                else:
                    misses.append({"key_point": kp, "similarity": round(sim, 4)})

        score = len(hits) / len(expected_key_points)

        return MetricResult(
            score=round(score, 4),
            details={
                "hits": hits,
                "misses": misses,
                "covered": len(hits),
                "total": len(expected_key_points),
            },
        )
