"""
忠实度评估——回答是否忠于上下文

算法：
1. 把回答拆成独立声明（中英文标点分割）
2. 每条声明用 embedding 算与上下文的语义相似度
3. 相似度 > FAITHFULNESS_THRESHOLD → 有支撑
4. 得分 = 有支撑声明数 / 总声明数
"""

import re
from app.evaluator.base import BaseMetric
from app.models.schemas import MetricResult
from app.config import FAITHFULNESS_THRESHOLD
from app.embedding import semantic_similarity


class FaithfulnessEvaluator(BaseMetric):

    def evaluate(self, question, context, answer, expected_key_points=None):
        # 1. 拆声明
        claims = self._split_into_claims(answer)
        if not claims:
            return MetricResult(score=0.0, details={"error": "回答为空"})

        # 2. 逐条验证
        checked = []
        for claim in claims:
            sim = semantic_similarity(claim, context)       # ← embedding！
            checked.append({
                "text": claim,
                "supported": sim >= FAITHFULNESS_THRESHOLD,
                "similarity": round(sim, 4),
            })

        # 3. 算分
        supported = sum(1 for c in checked if c["supported"])
        score = supported / len(checked)

        return MetricResult(
            score=round(score, 4),
            details={
                "claims": checked,
                "supported": supported,
                "total": len(checked),
            },
        )

    def _split_into_claims(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        sentences = re.split(r'[。！？\n\.!\?;；]+', text)
        return [s.strip() for s in sentences if len(s.strip()) >= 2]
