"""
幻觉检测——回答是否编造了上下文中不存在的事实

算法（两阶段）：
1. 正则提取实体：数字、日期、英文专名、中文书名号/引号内容
2. embedding 交叉验证：实体在上下文中语义存在吗？

embedding 的作用：
"Guido" vs "Guido van Rossum" → 正则匹配不上（子串不全等）
但 embedding 能识别它们指向同一个人 → 减少误报
"""

import re
from app.evaluator.base import BaseMetric
from app.models.schemas import MetricResult
from app.embedding import semantic_similarity


class HallucinationEvaluator(BaseMetric):

    def evaluate(self, question, context, answer, expected_key_points=None):
        entities = self._extract_entities(answer)

        if not entities:
            return MetricResult(score=1.0, details={"note": "无实体可提取"})

        # 去重
        seen = set()
        unique = []
        for e in entities:
            if e["value"] not in seen:
                seen.add(e["value"])
                unique.append(e)

        # 两阶段验证：先子串 → embedding
        hallucinated = []
        verified = []
        for e in unique:
            if self._entity_exists(e["value"], context):
                verified.append(e)
            else:
                hallucinated.append(e)

        total = len(unique)
        score = (total - len(hallucinated)) / total

        return MetricResult(
            score=round(score, 4),
            details={
                "total_entities": total,
                "verified": verified,
                "hallucinated": hallucinated,
            },
        )

    def _extract_entities(self, text: str) -> list[dict]:
        """提取可验证的实体"""
        entities = []

        # 数字（整数、小数、百分比）
        for n in re.findall(r'\b\d+\.?\d*%?\b', text):
            entities.append({"type": "number", "value": n})

        # 日期
        for d in re.findall(r'\d{4}年\d{1,2}月\d{1,2}日|\d{4}-\d{2}-\d{2}', text):
            entities.append({"type": "date", "value": d})

        # 英文专有名词（连续大写开头词）
        for p in re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text):
            if len(p) > 3:
                entities.append({"type": "entity", "value": p})

        # 中文书名号/引号
        for match in re.findall(r'《([^》]+)》|"([^"]+)"', text):
            val = match[0] or match[1]
            if len(val) >= 2:
                entities.append({"type": "entity", "value": val})

        return entities

    def _entity_exists(self, entity_value: str, context: str) -> bool:
        """两阶段检查实体是否在上下文中存在"""
        # 阶段 1：直接子串匹配
        if entity_value.lower() in context.lower():
            return True

        # 阶段 2：embedding 语义交叉验证
        sim = semantic_similarity(
            f"提到 {entity_value}",
            context[:500]  # 取前 500 字符，长上下文截断
        )
        return sim > 0.5
