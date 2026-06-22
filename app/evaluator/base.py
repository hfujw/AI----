from abc import ABC, abstractmethod
from app.models.schemas import MetricResult


class BaseMetric(ABC):
    """评估指标抽象基类

    所有评估器（Faithfulness / Relevance / Completeness / Hallucination）
    以及 LLM 评判器（DeepSeekJudge）都实现此接口。

    统一签名保证 TestRunner 可以无差别调用任意 evaluator。
    """

    @abstractmethod
    def evaluate(self, question: str, context: str, answer: str,
                 expected_key_points: list[str] | None = None) -> MetricResult:
        """评估单条回答的某个质量维度

        Args:
            question: 用户问题
            context: RAG 检索到的参考上下文
            answer: LLM 生成的回答
            expected_key_points: 预期覆盖的关键点（Completeness 使用）

        Returns:
            MetricResult(score=0.0~1.0, details={...})
        """
        pass
