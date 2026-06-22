"""Completeness（完整性）测试"""
from app.evaluator.completeness import CompletenessEvaluator


class TestCompleteness:

    def test_all_points_covered(self):
        """所有关键点全覆盖 → 1.0"""
        e = CompletenessEvaluator()
        r = e.evaluate(
            question="二甲双胍有什么副作用？",
            context="二甲双胍的副作用包括恶心、腹泻和腹部不适。",
            answer="副作用包括恶心、腹泻和腹部不适。",
            expected_key_points=["恶心", "腹泻", "腹部不适"],
        )
        assert r.score == 1.0

    def test_partial_coverage(self):
        """4 个关键点命中 1 个（仅"编程语言"字面匹配）→ 0.25"""
        e = CompletenessEvaluator()
        r = e.evaluate(
            question="Python 是什么？",
            context="Python 是由 Guido van Rossum 创建的编程语言，语法简洁。",
            answer="Python 是由 Guido 创建的编程语言。",
            expected_key_points=["Guido van Rossum", "编程语言", "语法简洁", "面向对象"],
        )
        assert 0.2 <= r.score < 0.8  # 1/4 子串命中，其余语义匹配不到
        assert "hits" in r.details
        assert "misses" in r.details

    def test_semantic_match(self):
        """语义级匹配——"感到恶心想吐" 能匹配关键点 "恶心" """
        e = CompletenessEvaluator()
        r = e.evaluate(
            question="二甲双胍有什么副作用？",
            context="二甲双胍常见副作用包括恶心、腹泻。",
            answer="服用后可能会感到恶心想吐，拉肚子。",
            expected_key_points=["恶心", "腹泻"],
        )
        assert r.score > 0   # 至少命中一个

    def test_no_keypoints(self):
        """没定义关键点 → 默认满分"""
        e = CompletenessEvaluator()
        r = e.evaluate(
            question="Python？", context="...", answer="...",
            expected_key_points=None,
        )
        assert r.score == 1.0
