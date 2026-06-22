# AI 评测框架 — 完整开发指南 v2.0

> 🎯 基于 **sentence-transformers** 的 LLM 输出质量评测框架。不调 API、确定性输出、中英双语、本地运行。
> 📋 原则：先写测试 → 再写实现 → 跑通 → 下一阶段。
> 🏗️ 这份文档是自包含的——所有代码都在里面，不需要参考其他文件。

---

## 目录

1. [环境搭建](#1-环境搭建)
2. [embedding 层](#2-embedding-层)
3. [Faithfulness 忠实度](#3-faithfulness-忠实度)
4. [Relevance 相关性](#4-relevance-相关性)
5. [Completeness 完整性](#5-completeness-完整性)
6. [Hallucination 幻觉检测](#6-hallucination-幻觉检测)
7. [Harness 测试驱动](#7-harness-测试驱动)
8. [FastAPI 端点](#8-fastapi-端点)
9. [DeepSeek 深度评判](#9-deepseek-深度评判)
10. [CI + README 收尾](#10-ci--readme-收尾)

---

## 1. 环境搭建

### 1.1 目录结构

```bash
cd "c:/Users/22075/Desktop/py/AI测评框架"
mkdir -p app/evaluator app/harness app/judge app/compare app/models tests examples
```

### 1.2 依赖安装

```bash
python -m venv venv
source venv/Scripts/activate
pip install fastapi uvicorn sentence-transformers pyyaml pytest httpx requests
```

### 1.3 pyproject.toml

```toml
[project]
name = "ai-eval-framework"
version = "0.1.0"
description = "AI 输出质量评测框架 — 幻觉检测、忠实度评估、多模型对比"
requires-python = ">=3.10"
dependencies = [
    "fastapi>=0.100.0",
    "uvicorn[standard]",
    "sentence-transformers>=3.0.0",
    "pyyaml>=6.0",
    "requests>=2.28.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "httpx>=0.24.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

### 1.4 创建所有 `__init__.py`

```bash
touch app/__init__.py app/evaluator/__init__.py app/harness/__init__.py
touch app/judge/__init__.py app/compare/__init__.py app/models/__init__.py
touch tests/__init__.py
```

### 1.5 .gitignore

```gitignore
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
venv/
.vscode/
.idea/
.DS_Store
Thumbs.db
```

### 1.6 app/config.py

```python
"""配置管理 — 环境变量（12-Factor App 原则）"""
import os

# ── embedding 模型 ─────────────────────────────────────────
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "paraphrase-multilingual-MiniLM-L12-v2"
)

# ── 评判模式 ───────────────────────────────────────────────
# "fast" = embedding 本地模型（CI 模式，确定性）
# "deep" = DeepSeek API（深度模式，语义级）
EVAL_MODE = os.getenv("EVAL_MODE", "fast")

# ── 各指标阈值 ─────────────────────────────────────────────
# Faithfulness：声明与上下文的语义相似度阈值
FAITHFULNESS_THRESHOLD = float(os.getenv("FAITHFULNESS_THRESHOLD", "0.6"))
# Completeness：关键点与回答的语义匹配阈值
COMPLETENESS_THRESHOLD = float(os.getenv("COMPLETENESS_THRESHOLD", "0.6"))

# ── DeepSeek 配置（仅 EVAL_MODE="deep" 时使用）─────────────
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# ── 日志 ──────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
```

### 1.7 app/models/schemas.py

```python
from pydantic import BaseModel

class EvaluateRequest(BaseModel):
    question: str
    context: str
    answer: str
    expected_key_points: list[str] | None = None

class MetricResult(BaseModel):
    score: float          # 0.0 ~ 1.0
    details: dict = {}    # 各指标自定义详情

class EvaluateResponse(BaseModel):
    faithfulness: MetricResult
    relevance: MetricResult
    completeness: MetricResult
    hallucination: MetricResult
    overall_score: float   # 四项平均

class BatchEvaluateRequest(BaseModel):
    cases: list[EvaluateRequest]

class BatchEvaluateResponse(BaseModel):
    results: list[EvaluateResponse]
    summary: dict

class CompareRequest(BaseModel):
    question: str
    context: str
    answers: dict[str, str]   # {"GPT-4": "回答", "Claude-3": "回答"}

class CompareResponse(BaseModel):
    question: str
    results: dict[str, EvaluateResponse]
    ranking: list[str]         # 按 overall_score 降序
```

### 1.8 app/evaluator/base.py

```python
from abc import ABC, abstractmethod
from app.models.schemas import MetricResult

class BaseMetric(ABC):
    """所有评估指标的抽象基类"""

    @abstractmethod
    def evaluate(self, question: str, context: str, answer: str,
                 expected_key_points: list[str] | None = None) -> MetricResult:
        pass
```

### 1.9 tests/conftest.py

```python
import pytest

@pytest.fixture
def sample_medical():
    return {
        "question": "二甲双胍有什么副作用？",
        "context": "二甲双胍是2型糖尿病的一线用药。常见副作用包括恶心、腹泻和腹部不适。罕见但严重的副作用包括乳酸性酸中毒。",
        "answer": "常见的副作用有恶心、腹泻和腹部不适。",
        "expected_key_points": ["恶心", "腹泻", "腹部不适", "乳酸性酸中毒"],
    }

@pytest.fixture
def sample_faithful():
    return {
        "question": "Python 是什么？",
        "context": "Python 是一门由 Guido van Rossum 于 1991 年创建的高级编程语言，以简洁易读的语法著称。",
        "answer": "Python 是由 Guido van Rossum 创建的一门高级编程语言，语法简洁易读。",
        "expected_key_points": ["Guido van Rossum", "高级编程语言", "语法简洁"],
    }

@pytest.fixture
def sample_hallucinated():
    return {
        "question": "Python 是什么？",
        "context": "Python 是一门由 Guido van Rossum 于 1991 年创建的高级编程语言。",
        "answer": "Python 是由 Dennis Ritchie 于 1970 年创建的低级编程语言，主要用于操作系统开发。",
        "expected_key_points": ["Guido van Rossum", "高级编程语言"],
    }

@pytest.fixture
def sample_irrelevant():
    return {
        "question": "光合作用需要什么条件？",
        "context": "光合作用需要光照、叶绿体、二氧化碳和水。",
        "answer": "植物细胞由细胞壁、细胞膜、细胞质和细胞核组成。",
        "expected_key_points": ["光照", "叶绿体", "二氧化碳", "水"],
    }
```

### ✅ 检查点

```bash
python -c "from app.config import EMBEDDING_MODEL; print(EMBEDDING_MODEL)"
python -c "from app.models.schemas import EvaluateRequest; print(EvaluateRequest(question='你好', context='...', answer='...'))"
```

---

## 2. embedding 层

> **核心设计**：整个应用只加载一次模型（470MB，加载需 3-5 秒）。所有 evaluator 共用同一个实例。

### app/embedding.py（完整代码，直接复制）

```python
"""
共享的 sentence-transformers embedding 模型

设计：单例模式——整个应用只加载一次模型。
四个 evaluator 通过 semantic_similarity() 函数调用。

模型选择：paraphrase-multilingual-MiniLM-L12-v2
- 384 维向量，轻量
- 中英双语
- 本地运行，免费
- 确定性输出（同输入永远同输出，CI 友好）
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

_model = None

def get_model():
    """获取全局唯一的模型实例（懒加载）"""
    global _model
    if _model is None:
        _model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    return _model

def semantic_similarity(text1: str, text2: str) -> float:
    """计算两段文本的语义相似度（0.0 ~ 1.0）

    用法：
        sim = semantic_similarity("你好", "您好")
        # → 0.92（同义改写也能识别）
    """
    model = get_model()
    embeddings = model.encode([text1, text2])
    sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return max(0.0, min(1.0, float(sim)))  # 夹到 [0, 1]
```

### 验证

```bash
python -c "from app.embedding import semantic_similarity; print(semantic_similarity('你好', '您好'))"
# 预期输出：0.7~0.95
```

---

## 3. Faithfulness（忠实度）

> **面试能聊**：RAG 最核心指标。把回答拆成原子声明，逐条用 embedding 验证是否在上下文中有语义支撑。
> TF-IDF 的局限——短文本共享词 IDF 归零。embedding 理解语义，"Python 是一门语言" 和 "Python 是编程语言" 能对上。

### 3.1 先写测试 `tests/test_faithfulness.py`

```python
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
```

### 3.2 实现 `app/evaluator/faithfulness.py`

```python
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
            details={"claims": checked, "supported": supported, "total": len(checked)},
        )

    def _split_into_claims(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        sentences = re.split(r'[。！？\n\.!\?;；]+', text)
        return [s.strip() for s in sentences if len(s.strip()) >= 2]
```

### 3.3 运行测试

```bash
pytest tests/test_faithfulness.py -v
# 预期：5 passed
```

---

## 4. Relevance（相关性）

> **面试能聊**：之前用关键词 n-gram + TF-IDF，两文档共享词 IDF 归零。现在直接用 embedding 算问题 vs 回答的语义相似度——"Python 是什么" vs "一门编程语言" 能对上。

### 4.1 先写测试 `tests/test_relevance.py`

```python
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
        assert r.score > 0.7

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
        assert r.score > 0.6   # 字面不同但语义相关

    def test_empty_question(self):
        """空问题 → 不崩溃"""
        e = RelevanceEvaluator()
        r = e.evaluate(question="", context="Python 是编程语言。", answer="Python 是编程语言。")
        assert r.score >= 0.0
```

### 4.2 实现 `app/evaluator/relevance.py`

```python
"""
相关性评估——回答是否切题

算法：用 embedding 直接算 问题 和 回答 的语义相似度。
不再需要手动分词、去停用词、算关键词覆盖率。
"""

from app.evaluator.base import BaseMetric
from app.models.schemas import MetricResult
from app.embedding import semantic_similarity


class RelevanceEvaluator(BaseMetric):

    def evaluate(self, question, context, answer, expected_key_points=None):
        if not question or not question.strip():
            return MetricResult(score=0.0, details={"error": "问题为空"})

        sim = semantic_similarity(question, answer)

        return MetricResult(
            score=round(sim, 4),
            details={"semantic_similarity": round(sim, 4)},
        )
```

### 4.3 运行测试

```bash
pytest tests/test_relevance.py -v
# 预期：4 passed
```

---

## 5. Completeness（完整性）

> **面试能聊**：RAG 回答常漏信息。用 embedding 做语义级关键点匹配——"感到恶心想吐" 能匹配关键点 "恶心"，子串匹配做不到。

### 5.1 先写测试 `tests/test_completeness.py`

```python
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
        """4 个关键点命中 2 个 → 0.5"""
        e = CompletenessEvaluator()
        r = e.evaluate(
            question="Python 是什么？",
            context="Python 是由 Guido van Rossum 创建的编程语言，语法简洁。",
            answer="Python 是由 Guido 创建的编程语言。",
            expected_key_points=["Guido van Rossum", "编程语言", "语法简洁", "面向对象"],
        )
        assert 0.4 < r.score < 0.8
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
```

### 5.2 实现 `app/evaluator/completeness.py`

```python
"""
完整性评估——回答是否覆盖了预期关键点

算法：
1. 如果没定义关键点 → 默认满分
2. 对每个关键点，用 embedding 算与回答的语义相似度
3. 相似度 > COMPLETENESS_THRESHOLD → 已覆盖
4. 得分 = 覆盖数 / 总数

embedding 的优势：
"感到恶心想吐" 能匹配关键点 "恶心"——子串匹配做不到。
"""

from app.evaluator.base import BaseMetric
from app.models.schemas import MetricResult
from app.config import COMPLETENESS_THRESHOLD
from app.embedding import semantic_similarity


class CompletenessEvaluator(BaseMetric):

    def evaluate(self, question, context, answer, expected_key_points=None):
        if not expected_key_points:
            return MetricResult(score=1.0, details={"note": "未定义关键点，默认满分"})

        hits = []
        misses = []
        for kp in expected_key_points:
            sim = semantic_similarity(kp, answer)
            if sim >= COMPLETENESS_THRESHOLD:
                hits.append({"key_point": kp, "similarity": round(sim, 4)})
            else:
                misses.append({"key_point": kp, "similarity": round(sim, 4)})

        score = len(hits) / len(expected_key_points)

        return MetricResult(
            score=round(score, 4),
            details={
                "hits": hits,
                "misses": misses,
                "covered": len(hits),
                "total": len(expected_key_points),
            },
        )
```

### 5.3 运行测试

```bash
pytest tests/test_completeness.py -v
# 预期：4 passed
```

---

## 6. Hallucination（幻觉检测）

> **面试能聊**：两阶段——正则提取实体（数字/日期/专名）+ embedding 验证是否在上下文中语义存在。比纯正则多一层语义交叉验证，减少误报。

### 6.1 先写测试 `tests/test_hallucination.py`

```python
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
```

### 6.2 实现 `app/evaluator/hallucination.py`

```python
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
        # 把实体嵌入上下文窗口，看语义相似度
        sim = semantic_similarity(
            f"提到 {entity_value}",
            context[:500]  # 取前 500 字符，长上下文截断
        )
        return sim > 0.5
```

### 6.3 运行测试

```bash
pytest tests/test_hallucination.py -v
# 预期：4 passed
```

---

## 阶段小结

四个评分器全部完成。跑全集：

```bash
pytest tests/test_faithfulness.py tests/test_relevance.py tests/test_completeness.py tests/test_hallucination.py -v
# 预期：17 passed（5+4+4+4）
```

---

## 7. Harness（测试驱动）

> YAML 定义用例，Runner 批量执行，pass/fail 基于阈值判定。

### 7.1 示例 YAML `examples/sample_test_cases.yaml`

```yaml
test_cases:
  - id: "medical_metformin"
    question: "二甲双胍有什么副作用？"
    context: "二甲双胍是2型糖尿病的一线用药。常见副作用包括恶心、腹泻和腹部不适。罕见但严重的副作用包括乳酸性酸中毒。"
    expected_answer: "常见的副作用有恶心、腹泻和腹部不适。"
    expected_key_points: ["恶心", "腹泻", "腹部不适", "乳酸性酸中毒"]
    thresholds:
      faithfulness: 0.7
      relevance: 0.6
      completeness: 0.5

  - id: "history_ww2"
    question: "二战什么时候结束的？"
    context: "第二次世界大战于1945年结束。欧洲战场于1945年5月8日结束，太平洋战场于1945年9月2日结束。"
    expected_answer: "二战于1945年结束。"
    expected_key_points: ["1945", "5月8日", "9月2日"]
    thresholds:
      faithfulness: 0.7
      relevance: 0.6
      completeness: 0.5
```

### 7.2 app/harness/loader.py

```python
import yaml
from dataclasses import dataclass

@dataclass
class TestCase:
    question: str
    context: str
    answer: str
    expected_key_points: list[str] | None
    thresholds: dict

class TestCaseLoader:
    @staticmethod
    def load(yaml_path: str) -> list[TestCase]:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        cases = []
        for item in data.get("test_cases", []):
            cases.append(TestCase(
                question=item["question"],
                context=item["context"],
                answer=item.get("expected_answer", ""),
                expected_key_points=item.get("expected_key_points"),
                thresholds=item.get("thresholds", {}),
            ))
        return cases
```

### 7.3 app/harness/runner.py

```python
from app.evaluator.faithfulness import FaithfulnessEvaluator
from app.evaluator.relevance import RelevanceEvaluator
from app.evaluator.completeness import CompletenessEvaluator
from app.evaluator.hallucination import HallucinationEvaluator
from app.models.schemas import EvaluateResponse, BatchEvaluateResponse

class TestRunner:
    def __init__(self, mode: str = "fast"):
        self.mode = mode
        self.metrics = {
            "faithfulness": FaithfulnessEvaluator(),
            "relevance": RelevanceEvaluator(),
            "completeness": CompletenessEvaluator(),
            "hallucination": HallucinationEvaluator(),
        }

    def run_single(self, question, context, answer, expected_key_points=None) -> EvaluateResponse:
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
                getattr(r, name).score >= tc.thresholds.get(name, 0)
                for name in self.metrics
                if name in tc.thresholds
            )
            if is_pass:
                passed += 1
            results.append(r)

        avg = {
            name: round(sum(getattr(r, name).score for r in results) / len(results), 4)
            for name in self.metrics
        }
        return BatchEvaluateResponse(
            results=results,
            summary={
                "total": len(test_cases), "passed": passed,
                "failed": len(test_cases) - passed,
                "pass_rate": round(passed / len(test_cases), 4) if test_cases else 0,
                "average_scores": avg,
            },
        )
```

### 7.4 测试 `tests/test_harness.py`

```python
"""Harness 测试"""
from pathlib import Path
from app.harness.loader import TestCaseLoader
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
        from app.harness.loader import TestCase
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
```

```bash
pytest tests/test_harness.py -v
```

---

## 8. FastAPI 端点

### app/main.py

```python
import logging
from fastapi import FastAPI
from app.config import LOG_LEVEL
from app.models.schemas import (
    EvaluateRequest, EvaluateResponse,
    BatchEvaluateRequest, BatchEvaluateResponse,
    CompareRequest, CompareResponse,
)
from app.harness.runner import TestRunner
from app.harness.loader import TestCase

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("ai_eval")

app = FastAPI(title="AI 评测框架", version="0.2.0")
runner = TestRunner()

@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate(request: EvaluateRequest):
    return runner.run_single(
        request.question, request.context, request.answer,
        request.expected_key_points,
    )

@app.post("/evaluate/batch", response_model=BatchEvaluateResponse)
def evaluate_batch(request: BatchEvaluateRequest):
    cases = [TestCase(
        question=c.question, context=c.context, answer=c.answer,
        expected_key_points=c.expected_key_points, thresholds={},
    ) for c in request.cases]
    return runner.run_batch(cases)

@app.post("/compare", response_model=CompareResponse)
def compare(request: CompareRequest):
    results = {}
    for model, answer in request.answers.items():
        results[model] = runner.run_single(request.question, request.context, answer)
    ranking = sorted(results, key=lambda m: results[m].overall_score, reverse=True)
    return CompareResponse(question=request.question, results=results, ranking=ranking)

@app.get("/health")
def health():
    return {"status": "ok", "metrics": list(runner.metrics.keys())}
```

### 手动验证

```bash
uvicorn app.main:app --reload
# 打开 http://localhost:8000/docs
```

---

## 9. DeepSeek 深度评判

> **面试能聊**：双模式架构——快模式（embedding 本地）CI 毫秒级，深度模式（DeepSeek）发版前语义级判题。同一接口，策略模式。

### 9.1 app/judge/base.py

```python
from abc import ABC, abstractmethod
from app.evaluator.base import BaseMetric
from app.models.schemas import MetricResult

class LLMJudge(BaseMetric):
    """一次 API 调用评估四个维度"""

    @abstractmethod
    def judge(self, question: str, context: str, answer: str,
              expected_key_points: list[str] | None = None) -> dict[str, MetricResult]:
        pass

    def evaluate(self, question, context, answer, expected_key_points=None):
        results = self.judge(question, context, answer, expected_key_points)
        overall = sum(r.score for r in results.values()) / max(len(results), 1)
        return MetricResult(score=round(overall, 4), details={"per_dimension": results})
```

### 9.2 app/judge/deepseek_judge.py

<details>
<summary>完整代码（点击展开）</summary>

```python
"""DeepSeek 深度评判器——一次 API 调用评估四个维度"""
import json, logging, os, re, requests
from app.judge.base import LLMJudge
from app.models.schemas import MetricResult

logger = logging.getLogger("ai_eval")

class DeepSeekJudge(LLMJudge):
    BASE_URL = "https://api.deepseek.com/v1/chat/completions"

    def __init__(self, api_key=None, model=None):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    def judge(self, question, context, answer, expected_key_points=None):
        prompt = self._build_prompt(question, context, answer, expected_key_points)
        try:
            response = self._call_api(prompt)
            scores = self._parse_response(response)
            return self._to_metric_results(scores)
        except Exception as e:
            logger.error(f"DeepSeek 评判失败: {e}")
            return self._fallback(str(e))

    def _build_prompt(self, question, context, answer, key_points):
        kp = f"\n预期关键点：{'、'.join(key_points)}" if key_points else ""
        return f"""你是 RAG 回答质量评估专家。请严格基于参考上下文评估。

【问题】{question}
【上下文】{context}{kp}
【回答】{answer}

评估四个维度（0-10 分）：
1. 忠实度：回答是否完全基于上下文？有无编造？
2. 相关性：回答是否切题？
3. 完整性：是否覆盖了关键信息？
4. 幻觉检测：有无上下文中不存在的事实？

严格返回 JSON：{{"faithfulness":8,"relevance":9,"completeness":7,"hallucination":10,"reasons":{{"faithfulness":"...","relevance":"...","completeness":"...","hallucination":"..."}}}}"""

    def _call_api(self, prompt):
        resp = requests.post(self.BASE_URL,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={"model": self.model, "messages": [
                {"role": "system", "content": "你是精确的 RAG 评测专家。只返回 JSON。"},
                {"role": "user", "content": prompt},
            ], "temperature": 0, "max_tokens": 500},
            timeout=30)
        resp.raise_for_status()
        return resp.json()

    def _parse_response(self, response):
        content = response["choices"][0]["message"]["content"]
        if "```" in content:
            match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
            if match:
                content = match.group(1)
        return json.loads(content)

    def _to_metric_results(self, scores):
        reasons = scores.get("reasons", {})
        return {
            "faithfulness": MetricResult(score=scores.get("faithfulness",0)/10.0, details={"reason":reasons.get("faithfulness",""),"raw":scores.get("faithfulness")}),
            "relevance": MetricResult(score=scores.get("relevance",0)/10.0, details={"reason":reasons.get("relevance",""),"raw":scores.get("relevance")}),
            "completeness": MetricResult(score=scores.get("completeness",0)/10.0, details={"reason":reasons.get("completeness",""),"raw":scores.get("completeness")}),
            "hallucination": MetricResult(score=scores.get("hallucination",0)/10.0, details={"reason":reasons.get("hallucination",""),"raw":scores.get("hallucination")}),
        }

    def _fallback(self, error):
        return {name: MetricResult(score=-1.0, details={"error": error})
                for name in ["faithfulness","relevance","completeness","hallucination"]}
```

</details>

### 9.3 app/compare/comparator.py

```python
from app.harness.runner import TestRunner
from app.models.schemas import CompareRequest, CompareResponse

class ModelComparator:
    def __init__(self, mode="fast"):
        self.runner = TestRunner(mode=mode)

    def compare(self, request: CompareRequest) -> CompareResponse:
        results = {}
        for model, answer in request.answers.items():
            results[model] = self.runner.run_single(request.question, request.context, answer)
        ranking = sorted(results, key=lambda m: results[m].overall_score, reverse=True)
        return CompareResponse(question=request.question, results=results, ranking=ranking)
```

### 9.4 Runner 双模式支持（改 app/harness/runner.py 的 __init__）

```python
def __init__(self, mode: str = "fast"):
    from app.config import EVAL_MODE
    self.mode = mode or EVAL_MODE

    if self.mode == "deep":
        from app.judge.deepseek_judge import DeepSeekJudge
        judge = DeepSeekJudge()
        self.metrics = {"deepseek": judge}
        self._is_deep = True
    else:
        self.metrics = {
            "faithfulness": FaithfulnessEvaluator(),
            "relevance": RelevanceEvaluator(),
            "completeness": CompletenessEvaluator(),
            "hallucination": HallucinationEvaluator(),
        }
        self._is_deep = False

def run_single(self, ...):
    if self._is_deep:
        judge = list(self.metrics.values())[0]
        dims = judge.judge(question, context, answer, expected_key_points)
        overall = sum(r.score for r in dims.values()) / 4
        return EvaluateResponse(
            faithfulness=dims["faithfulness"],
            relevance=dims["relevance"],
            completeness=dims["completeness"],
            hallucination=dims["hallucination"],
            overall_score=round(overall, 4),
        )
    # 否则走快模式（原来的循环）
    ...
```

---

## 10. CI + README 收尾

### .github/workflows/test.yml

```yaml
name: Run Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -e ".[dev]"
      - run: pytest tests/ -v --tb=short
```

### README.md 结构

1. 项目简介（1-2 句中文）
2. 快速开始（pip install + uvicorn）
3. 四项指标说明（表格）
4. API 文档（4 端点 + curl 示例）
5. YAML 用例格式
6. 项目结构（目录树）
7. 设计决策（为什么 embedding？为什么双模式？为什么不 Redis？）
8. 测试（pytest tests/ -v）

---

## 全量测试

```bash
pytest tests/ -v --tb=short
# 预期：20 passed
```

---

> 📊 **进度追踪** ✅ 全部完成
>
> - [x] Phase 1：环境搭建（30 min）
> - [x] Phase 2：embedding 层（10 min）
> - [x] Phase 3：Faithfulness（30 min）
> - [x] Phase 4：Relevance（20 min）
> - [x] Phase 5：Completeness（20 min）
> - [x] Phase 6：Hallucination（30 min）
> - [x] Phase 7：Harness（30 min）
> - [x] Phase 8：FastAPI（20 min）
> - [x] Phase 9：DeepSeek（30 min）
> - [x] Phase 10：CI + README（30 min）
>
> **代码总量：~700 行，20 tests | 📖 完整文档：[PROJECT.md](PROJECT.md)**
