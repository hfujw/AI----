"""
测试执行器——两种模式统一入口

fast 模式：4 个本地 evaluator 分别打分（embedding，确定性）
deep 模式：1 次 DeepSeek API 调用返回 4 维分数（语义级，非确定性）

设计：策略模式——run_single / run_batch 不关心底层是谁在打分
"""

from app.evaluator.faithfulness import FaithfulnessEvaluator
from app.evaluator.relevance import RelevanceEvaluator
from app.evaluator.completeness import CompletenessEvaluator
from app.evaluator.hallucination import HallucinationEvaluator
from app.models.schemas import EvaluateResponse, BatchEvaluateResponse

# 四个标准评估维度——run_batch 用此常量而非 self.metrics.keys()
# 因为 deep 模式下 metrics key 是 "deepseek" 而非标准维度名
DIMENSIONS = ["faithfulness", "relevance", "completeness", "hallucination"]


class TestRunner:
    def __init__(self, mode: str = "fast"):
        from app.config import EVAL_MODE
        self.mode = mode or EVAL_MODE

        if self.mode == "deep":
            from app.judge.deepseek_judge import DeepSeekJudge
            self._judge = DeepSeekJudge()
            self._is_deep = True
        else:
            self.metrics = {
                "faithfulness": FaithfulnessEvaluator(),
                "relevance": RelevanceEvaluator(),
                "completeness": CompletenessEvaluator(),
                "hallucination": HallucinationEvaluator(),
            }
            self._is_deep = False

    def run_single(self, question, context, answer, expected_key_points=None) -> EvaluateResponse:
        if self._is_deep:
            dims = self._judge.judge(question, context, answer, expected_key_points)
            overall = sum(r.score for r in dims.values()) / 4
            return EvaluateResponse(
                faithfulness=dims["faithfulness"],
                relevance=dims["relevance"],
                completeness=dims["completeness"],
                hallucination=dims["hallucination"],
                overall_score=round(overall, 4),
            )

        results = {}
        for name, metric in self.metrics.items():
            results[name] = metric.evaluate(question, context, answer, expected_key_points)

        overall = sum(r.score for r in results.values()) / 4
        return EvaluateResponse(
            faithfulness=results["faithfulness"],
            relevance=results["relevance"],
            completeness=results["completeness"],
            hallucination=results["hallucination"],
            overall_score=round(overall, 4),
        )

    def run_batch(self, test_cases: list) -> BatchEvaluateResponse:
        results = []
        passed = 0
        for tc in test_cases:
            r = self.run_single(tc.question, tc.context, tc.answer, tc.expected_key_points)
            is_pass = all(
                getattr(r, dim).score >= tc.thresholds.get(dim, 0)
                for dim in DIMENSIONS
                if dim in tc.thresholds
            )
            if is_pass:
                passed += 1
            results.append(r)

        avg = {
            dim: round(sum(getattr(r, dim).score for r in results) / len(results), 4)
            for dim in DIMENSIONS
        }
        return BatchEvaluateResponse(
            results=results,
            summary={
                "total": len(test_cases),
                "passed": passed,
                "failed": len(test_cases) - passed,
                "pass_rate": round(passed / len(test_cases), 4) if test_cases else 0,
                "average_scores": avg,
            },
        )
