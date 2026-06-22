"""Hallucination（幻觉检测）测试"""
from app.evaluator.hallucination import HallucinationEvaluator


class TestHallucination:

    def test_no_hallucination(self, sample_faithful):
        """所有实体都在上下文中 → 满分"""
        e = HallucinationEvaluator()
        r = e.evaluate(sample_faithful["question"],
                       sample_faithful["context"],
                       sample_faithful["answer"])
        assert r.score == 1.0

    def test_number_made_up(self):
        """编造数字 → 扣分"""
        e = HallucinationEvaluator()
        r = e.evaluate(
            question="Python 什么时候创建的？",
            context="Python 由 Guido van Rossum 于 1991 年创建。",
            answer="Python 于 2005 年由 Guido van Rossum 创建。",  # 2005 不在上下文中
        )
        assert r.score < 1.0
        assert len(r.details["hallucinated"]) > 0

    def test_entity_made_up(self, sample_hallucinated):
        """编造人名 → 扣分"""
        e = HallucinationEvaluator()
        r = e.evaluate(sample_hallucinated["question"],
                       sample_hallucinated["context"],
                       sample_hallucinated["answer"])
        assert r.score < 1.0

    def test_empty_answer(self):
        """空回答 → 满分（无实体可提取，没有幻觉）"""
        e = HallucinationEvaluator()
        r = e.evaluate(question="Python？", context="Python 是编程语言。", answer="")
        assert r.score == 1.0
