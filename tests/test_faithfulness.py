"""Faithfulness（忠实度）测试"""
from app.evaluator.faithfulness import FaithfulnessEvaluator


class TestFaithfulness:

    def test_fully_faithful(self, sample_faithful):
        """回答基于上下文 → 高分"""
        e = FaithfulnessEvaluator()
        r = e.evaluate(sample_faithful["question"],
                       sample_faithful["context"],
                       sample_faithful["answer"])
        assert r.score > 0.7
        assert r.details["supported"] > 0
        for claim in r.details["claims"]:
            assert "text" in claim
            assert "supported" in claim
            assert "similarity" in claim

    def test_partially_faithful(self):
        """一半编造 → 中等分数"""
        e = FaithfulnessEvaluator()
        r = e.evaluate(
            question="Python 是什么？",
            context="Python 是 Guido van Rossum 于 1991 年创建的高级编程语言，语法简洁。",
            answer="Python 是 Guido van Rossum 创建的编程语言。可以飞上天。能治癌症。",
        )
        assert 0.2 < r.score < 0.7

    def test_paraphrase_detection(self):
        """同义改写也能识别为有支撑"""
        e = FaithfulnessEvaluator()
        r = e.evaluate(
            question="Python 是什么？",
            context="Python 是一门语法简洁的高级编程语言。",
            answer="Python 的代码写起来很简单明了。",  # "语法简洁"的同义改写
        )
        assert r.score > 0.4  # embedding 能抓到同义关系

    def test_empty_context(self):
        """空上下文 → 0"""
        e = FaithfulnessEvaluator()
        r = e.evaluate(question="Python？", context="", answer="Python 是编程语言。")
        assert r.score == 0.0

    def test_empty_answer(self):
        """空回答 → 0，不崩溃"""
        e = FaithfulnessEvaluator()
        r = e.evaluate(question="Python？", context="Python 是编程语言。", answer="")
        assert r.score == 0.0
