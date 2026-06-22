from pydantic import BaseModel

class EvaluateRequest(BaseModel):
    """单条评估请求"""
    question: str                              # 用户问题
    context: str                               # RAG 检索到的上下文
    answer: str                                # LLM 生成的回答
    expected_key_points: list[str] | None = None  # 预期覆盖的关键点（可选）

class MetricResult(BaseModel):
    """单个指标的评估结果"""
    score: float        # 0.0 ~ 1.0
    details: dict = {}  # 各指标自定义的详细信息

class EvaluateResponse(BaseModel):
    """单条评估的完整响应"""
    faithfulness: MetricResult
    relevance: MetricResult
    completeness: MetricResult
    hallucination: MetricResult
    overall_score: float   # 四项平均

class BatchEvaluateRequest(BaseModel):
    """批量评估请求"""
    cases: list[EvaluateRequest]

class BatchEvaluateResponse(BaseModel):
    """批量评估响应"""
    results: list[EvaluateResponse]
    summary: dict   # {"total": N, "passed": N, "failed": N, "average_scores": {...}}

class CompareRequest(BaseModel):
    """多模型对比请求"""
    question: str
    context: str
    answers: dict[str, str]  # {"GPT-4": "回答1", "Claude-3": "回答2", ...}

class CompareResponse(BaseModel):
    """多模型对比响应"""
    question: str
    results: dict[str, EvaluateResponse]  # 每个模型的四项指标
    ranking: list[str]                    # 按 overall_score 降序排列的模型名