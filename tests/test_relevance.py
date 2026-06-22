"""Relevance（相关性）测试"""
from app.evaluator.relevance import RelevanceEvaluator


class TestRelevance:

    def test_highly_relevant(self):
        """直接切题 → 高分"""
        e = RelevanceEvaluator()
        r = e.evaluate(
            question="光合作用需要什么条件？",
            context="光合作用需要光照、叶绿体、二氧化碳和水。",
            answer="光合作用需要光照、叶绿素、二氧化碳和水这些条件。",
        )
        assert r.score > 0.6

    def test_irrelevant(self, sample_irrelevant):
        """答非所问 → 低分"""
        e = RelevanceEvaluator()
        r = e.evaluate(sample_irrelevant["question"],
                       sample_irrelevant["context"],
                       sample_irrelevant["answer"])
        assert r.score < 0.3

    def test_semantic_match(self):
        """embedding 能抓到语义层面的相关性"""
        e = RelevanceEvaluator()
        r = e.evaluate(
            question="Python 是什么？",
            context="Python 是由 Guido van Rossum 开发的一门编程语言。",
            answer="一门由 Guido 创建的计算机语言。",
        )
        assert r.score > 0.2   # 字面不同但语义相关（轻量模型得分偏低）

    def test_empty_question(self):
        """空问题 → 不崩溃"""
        e = RelevanceEvaluator()
        r = e.evaluate(question="", context="Python 是编程语言。", answer="Python 是编程语言。")
        assert r.score >= 0.0
