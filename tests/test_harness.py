"""Harness 测试"""
from pathlib import Path
from app.harness.loader import TestCaseLoader, TestCase
from app.harness.runner import TestRunner


class TestHarness:
    def test_load_yaml(self):
        yaml_path = Path(__file__).parent.parent / "examples" / "sample_test_cases.yaml"
        cases = TestCaseLoader.load(str(yaml_path))
        assert len(cases) == 2
        assert cases[0].thresholds["faithfulness"] == 0.7

    def test_run_single_case(self, sample_faithful):
        runner = TestRunner()
        r = runner.run_single(sample_faithful["question"],
                              sample_faithful["context"],
                              sample_faithful["answer"])
        assert r.overall_score > 0
        assert r.faithfulness.score > 0
        assert r.relevance.score > 0

    def test_run_batch(self, sample_faithful):
        runner = TestRunner()
        tc = TestCase(
            question=sample_faithful["question"],
            context=sample_faithful["context"],
            answer=sample_faithful["answer"],
            expected_key_points=sample_faithful["expected_key_points"],
            thresholds={"faithfulness": 0.5, "relevance": 0.5},
        )
        result = runner.run_batch([tc])
        assert result.summary["total"] == 1
