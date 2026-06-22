# AI 评测框架 — 项目完全手册

> 一个 **小而美** 的 RAG 回答质量评测框架。本地 embedding 模型 + 可切换 DeepSeek 深度评判，四维打分，YAML 驱动，FastAPI 服务化。
>
> **一句话**：给 RAG 系统的输出自动打分——回答有没有编造？有没有跑题？关键点有没有漏？
>
> **代码量**：~700 行 Python | **测试**：20 个 | **依赖**：7 个核心库

---

## 目录

1. [30 秒速览](#30-秒速览)
2. [架构总览](#架构总览)
3. [文件地图](#文件地图)
4. [库的使用](#库的使用)
5. [设计决策](#设计决策)
6. [面试 36 问](#面试-36-问)
7. [快速开始](#快速开始)

---

## 30 秒速览

```bash
# 启动服务
uvicorn app.main:app --reload

# 调 API
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "question": "二甲双胍有什么副作用？",
    "context": "二甲双胍常见副作用包括恶心、腹泻...",
    "answer": "常见的副作用有恶心、腹泻和腹部不适。",
    "expected_key_points": ["恶心", "腹泻", "腹部不适"]
  }'

# 返回
{
  "faithfulness":   {"score": 1.0,  "details": {...}},
  "relevance":      {"score": 0.92, "details": {...}},
  "completeness":   {"score": 0.75, "details": {...}},
  "hallucination":  {"score": 0.95, "details": {...}},
  "overall_score": 0.905
}
```

**两种模式**：

| | fast 模式 | deep 模式 |
|---|---|---|
| 引擎 | 本地 embedding 模型 | DeepSeek API |
| 速度 | ~100ms/条（CPU） | ~2s/条（网络） |
| 结果 | **确定性**（同输入永远同输出） | 语义级，可能有波动 |
| 成本 | 免费 | API 按 token 计费 |
| 适用 | CI / 开发阶段 | 发版前 / 正式评测 |
| 切换 | 默认 | 设 `EVAL_MODE=deep` |

---

## 架构总览

```
                        ┌─────────────────────┐
                        │    app/main.py       │
                        │   FastAPI 服务层      │
                        │  /evaluate /compare  │
                        └──────────┬──────────┘
                                   │
                        ┌──────────▼──────────┐
                        │  harness/runner.py   │
                        │    TestRunner        │
                        │  策略模式：选模式     │
                        └──────┬──────┬───────┘
                               │      │
              ┌────────────────┘      └───────────────┐
              │ fast 模式                              │ deep 模式
    ┌─────────▼──────────┐                  ┌─────────▼──────────┐
    │  4 个 Evaluator     │                  │  judge/            │
    │  Faithfulness       │                  │  deepseek_judge.py │
    │  Relevance          │                  │  一次 API 调四维   │
    │  Completeness       │                  └───────────────────┘
    │  Hallucination      │
    └─────────┬──────────┘
              │ 全部调用
    ┌─────────▼──────────┐
    │  embedding.py       │
    │  semantic_similarity│
    │  MiniLM-L12 (384维) │
    │  单例，全应用共享    │
    └────────────────────┘
```

**数据流**：
```
用户请求 → Pydantic 校验 → TestRunner.run_single()
  → [fast] 4 个 Evaluator 各自调用 semantic_similarity()
  → [deep] DeepSeekJudge 一次 API 返回 4 维
  → EvaluateResponse (Pydantic 序列化) → JSON 返回
```

---

## 文件地图

### 第 1 层：基础设施

#### `app/config.py` — 配置中心
- **干什么**：所有可调参数，通过环境变量读取（12-Factor App）
- **暴露什么**：`EMBEDDING_MODEL`、`EVAL_MODE`、`FAITHFULNESS_THRESHOLD`、`COMPLETENESS_THRESHOLD`、DeepSeek 密钥
- **被谁用**：`embedding.py`（读模型名）、`faithfulness.py`（读阈值）、`completeness.py`（读阈值）、`runner.py`（读模式）、`deepseek_judge.py`（读密钥）、`main.py`（读日志级别）
- **设计要点**：`os.getenv(key, default)` 模式——每个配置都有合理默认值，不设环境变量也能跑

#### `app/embedding.py` — 语义引擎
- **干什么**：全局唯一的 sentence-transformers 模型实例 + 语义相似度计算
- **暴露什么**：`semantic_similarity(text1, text2) -> float`
- **被谁用**：所有 4 个 evaluator + hallucination 的实体交叉验证
- **设计要点**：
  - **单例模式**：`_model = None`，首次调用才加载（懒加载），之后复用
  - **模型选择**：`paraphrase-multilingual-MiniLM-L12-v2`——384 维、中英双语、470MB、CPU 可跑
  - **输出钳制**：`max(0.0, min(1.0, sim))`——防止浮点误差导致越界
  - **HF 镜像**：自动检测环境，国内默认走 `hf-mirror.com`

#### `app/models/schemas.py` — 数据模型
- **干什么**：Pydantic 请求/响应模型，FastAPI 自动生成 OpenAPI 文档
- **暴露什么**：6 个模型——`EvaluateRequest`、`EvaluateResponse`、`MetricResult`、`Batch*`、`Compare*`
- **被谁用**：`main.py`（API 类型校验）、`runner.py`（构造响应）、所有 evaluator（构造 `MetricResult`）

### 第 2 层：评估核心

#### `app/evaluator/base.py` — 抽象基类
- **干什么**：定义统一的 `evaluate()` 签名——策略模式的接口
- **暴露什么**：`BaseMetric` ABC
- **被谁用**：4 个 evaluator 继承它，`LLMJudge` 也继承它
- **设计要点**：4 个参数 `(question, context, answer, expected_key_points)` 统一签名，`runner` 不关心具体实现

#### `app/evaluator/faithfulness.py` — 忠实度
- **干什么**：回答有没有忠于上下文？有没有编造？
- **算法**：
  1. 把回答按中英文标点拆成"原子声明"
  2. 每条声明 vs 上下文 → `semantic_similarity()`
  3. 相似度 > `FAITHFULNESS_THRESHOLD`(0.6) → "有支撑"
  4. 得分 = 有支撑声明数 / 总声明数
- **面试能聊**：为什么不用 ROUGE/BLEU？因为它们只看 n-gram 重叠，"Python 语法简洁" 和 "Python 代码简单" 算不出来——但 embedding 能抓到语义同义

#### `app/evaluator/relevance.py` — 相关性
- **干什么**：回答有没有跑题？
- **算法**：问题 vs 回答 → `semantic_similarity()`，一个数字搞定
- **极简**：只用 embedding 算 question-answer 相似度，不需要 TF-IDF、关键词提取
- **面试能聊**：为什么不用 TF-IDF？两文档场景下 IDF 归零（共享词 weight=0），加上 embedding 比纯统计更鲁棒

#### `app/evaluator/completeness.py` — 完整性
- **干什么**：关键信息有没有遗漏？
- **算法**（两阶段）：
  1. **子串匹配**：关键点是否直接出现在回答中（快速、精确）
  2. **语义匹配**：子串没命中 → embedding 语义相似度（"感到恶心想吐" ≈ "恶心"）
- **面试能聊**：为什么两阶段？子串匹配精确但脆弱（"Guido" ≠ "Guido van Rossum"），embedding 鲁棒但慢——两阶段取长补短

#### `app/evaluator/hallucination.py` — 幻觉检测
- **干什么**：回答是否编造了上下文中不存在的事实？
- **算法**（两阶段）：
  1. **正则提取实体**：数字、日期、英文专名、中文引号内容
  2. **交叉验证**：先子串检查 → embedding 语义验证（"Guido" vs "Guido van Rossum"）
- **面试能聊**：为什么要两阶段验证？正则"Guido"匹配不到"Guido van Rossum"→ 误报。embedding 能识别它们指向同一个人 → 减少误报

### 第 3 层：测试驱动

#### `app/harness/loader.py` — YAML 用例加载
- **干什么**：从 YAML 读取测试用例，反序列化为 `TestCase` dataclass
- **暴露什么**：`TestCase`（dataclass）、`TestCaseLoader.load(path) -> list[TestCase]`
- **被谁用**：`test_harness.py`、`main.py`（batch 端点间接触发）

#### `app/harness/runner.py` — 测试执行器
- **干什么**：统一入口——`run_single()` 打单条，`run_batch()` 批量跑 + pass/fail 判定
- **核心设计**：
  - **策略模式**：根据 `EVAL_MODE` 选 fast（4 evaluator）或 deep（DeepSeek）
  - **`DIMENSIONS` 常量**：解决 deep 模式下 metrics dict key 不匹配 EvaluateResponse 字段的问题
- **被谁用**：`main.py`（直接挂到 API 端点）、`comparator.py`、`test_harness.py`

### 第 4 层：服务化

#### `app/main.py` — FastAPI 入口
- **干什么**：4 个端点——`/evaluate`、`/evaluate/batch`、`/compare`、`/health`
- **启动**：`uvicorn app.main:app --reload`，访问 `http://localhost:8000/docs` 看 Swagger
- **设计要点**：
  - `/compare`：传入多模型回答 `{"GPT-4": "...", "Claude": "..."}`，返回排名
  - `/health`：返回当前模式和维度名

### 第 5 层：深度评判

#### `app/judge/base.py` — 评判器接口
- **干什么**：`LLMJudge` 继承 `BaseMetric`，新增 `judge()` 方法一次返回 4 维
- **面试能聊**：为什么不另起炉灶而是继承 `BaseMetric`？这样 `TestRunner` 可以无差别调用——策略模式的关键

#### `app/judge/deepseek_judge.py` — DeepSeek 评判器
- **干什么**：构造 prompt → 调 DeepSeek API → 解析返回的 JSON → 映射为 `MetricResult`
- **鲁棒性设计**：
  - `_parse_response`：先尝试 `json.loads`，失败后用正则从非标准格式里抠分数
  - `_fallback`：API 调用失败返回 `score=-1.0`，标记为错误状态
  - `temperature=0`：追求确定性输出
- **面试能聊**：prompt 里为什么要求"严格返回 JSON"？为什么 temperature=0？

#### `app/compare/comparator.py` — 模型对比
- **干什么**：对同一个问题，用多个模型的回答跑评测，按 overall_score 排名
- **面试能聊**：这种架构天然支持"评测评测者"——用 GPT-4 当裁判评测 Claude 的回答，看谁的 bias 更小

### 第 6 层：工程化

#### `tests/conftest.py` — 测试夹具
- 4 个 `@pytest.fixture`：医疗场景、忠实回答、幻觉回答、无关回答
- **面试能聊**：为什么用 fixture 而不是 hardcode？复用、可读性、测试隔离

#### `tests/test_*.py` — 5 个测试文件（20 个测试）
- `test_faithfulness.py`：5 个测试——正常、部分编造、同义改写、空上下文、空回答
- `test_relevance.py`：4 个测试——切题、跑题、语义相关、空问题
- `test_completeness.py`：4 个测试——全覆盖、部分覆盖、语义匹配、无关键点
- `test_hallucination.py`：4 个测试——无幻觉、编造数字、编造人名、空回答
- `test_harness.py`：3 个测试——YAML 加载、单条跑分、批量跑分

#### `examples/sample_test_cases.yaml` — 示例用例
- 2 个 YAML 用例（医疗 + 历史），含预期回答、关键点、各项阈值

#### `.github/workflows/test.yml` — CI
- Python 3.10/3.11/3.12 矩阵测试，push/PR 自动触发

---

## 库的使用

### `sentence-transformers`
- **哪些文件用**：`embedding.py`（唯一直引）
- **为什么用**：将文本映射到 384 维语义向量，同义改写 ("语法简洁" vs "代码简单") 在向量空间中距离近
- **怎么用**：`SentenceTransformer(model_name).encode([text1, text2])` → 两个 numpy array
- **替代方案**：OpenAI Embeddings API（贵、有网络依赖）、Word2Vec（只看词不看上下文）
- **面试深度**：为什么选 MiniLM-L12 而不是更大的模型？384 维 vs 768 维的 trade-off？答：470MB 对 CPU 友好，384 维足够短文本语义比较，更大模型（1.5GB+）对短文本提升有限但推理慢 3-5 倍

### `scikit-learn`
- **哪些文件用**：`embedding.py`
- **为什么用**：`cosine_similarity()`——算两个 embedding 向量的夹角余弦
- **怎么用**：`cosine_similarity([emb1], [emb2])[0][0]` → 0.0~1.0
- **面试深度**：为什么余弦相似度而不是欧氏距离？embedding 向量的模长没有语义含义，方向才重要——余弦只看方向

### `fastapi` + `uvicorn`
- **哪些文件用**：`main.py`
- **为什么用**：FastAPI 自动生成 OpenAPI 文档（`/docs`）、Pydantic 自动校验、异步支持
- **怎么用**：`@app.post("/evaluate", response_model=EvaluateResponse)`
- **面试深度**：为什么选 FastAPI 而不是 Flask？类型安全（Pydantic）、自动文档、异步原生支持、性能更好

### `pydantic`
- **哪些文件用**：`schemas.py`（定义模型）、`main.py`（请求校验）
- **为什么用**：类型安全 + 自动校验 + JSON Schema 生成
- **怎么用**：`class EvaluateRequest(BaseModel)`，FastAPI 自动从请求体反序列化并校验
- **面试深度**：Pydantic v2 vs v1？v2 用 Rust 重写核心，快 5-50 倍

### `pyyaml`
- **哪些文件用**：`loader.py`
- **为什么用**：解析 YAML 测试用例——非技术人员也能写
- **怎么用**：`yaml.safe_load(file)` → dict → 反序列化为 `TestCase` dataclass

### `requests`
- **哪些文件用**：`deepseek_judge.py`
- **为什么用**：调 DeepSeek API（OpenAI 兼容格式）
- **怎么用**：`requests.post(url, headers=auth, json=body, timeout=30)`
- **面试深度**：为什么不用 `openai` SDK？减少依赖，DeepSeek API 就是标准 REST，`requests` 完全够用

### `re` (标准库)
- **哪些文件用**：`faithfulness.py`（拆句子）、`hallucination.py`（提取实体）、`deepseek_judge.py`（解析 JSON）
- **为什么用**：中英文标点分割、实体正则提取、代码块提取——标准库零额外依赖

### `abc` (标准库)
- **哪些文件用**：`base.py`、`judge/base.py`
- **为什么用**：定义抽象基类，强制子类实现 `evaluate()`——策略模式的基础

---

## 设计决策

### 1. 为什么用 embedding 而不是调 LLM API 打分？

**不是 embedding vs LLM，是 fast vs deep 双模式**。

- **fast 模式**（embedding）：毫秒级、确定性、免费、离线可跑、CI 友好——开发阶段每次 commit 都跑
- **deep 模式**（DeepSeek）：语义级、秒级、非确定性、需 API key——发版前深度评测

这是 **策略模式** 的教科书用法：两个模式实现同一接口，切换零成本。

### 2. 为什么用单例模式管理 embedding 模型？

470MB 的模型只加载一次，所有 evaluator 共享。如果每个请求都加载一次 → 内存爆炸 + 启动 3-5 秒。

### 3. 为什么 completeness 用两阶段（子串 + embedding）？

子串匹配：快（O(n)）、精确（100% 召回）、但脆弱（"Guido" ≠ "Guido van Rossum"）
embedding：鲁棒（能识别同义）、但慢（50-200ms/次）、可能误匹配

**取长补短**：先走快速精确的子串 → 没命中再走鲁棒的 embedding

### 4. 为什么 hallucination 也用两阶段？

同 completeness 逻辑，但方向相反：

hallucination 是先子串检查实体是否在上下文中（精确匹配），没匹配到再用 embedding 验证（防误报："Dennis" vs "Dennis Ritchie"）

### 5. 为什么不多加几个评估维度？

"小而美"——4 个维度覆盖 RAG 质量的核心面。加更多维度（毒性检测、事实一致性、引用准确性）时，只需实现 `BaseMetric` 然后在 `runner.py` 注册即可——**开放-封闭原则**。

### 6. 为什么用 YAML 而不是 JSON 定义测试用例？

YAML 可读性更好，非技术人员也能写。`loader.py` 抽象了加载逻辑——以后想切 JSON/TOML 只需改 loader。

### 7. 为什么 CI 里跑 embedding 模式而不是 deep 模式？

CI 需要：快（秒级）、不花钱、确定性（不能 flaky）。fast 模式全满足。deep 模式留给发版前手动跑。

---

## 面试 36 问

### 项目概述

**Q1: 用一句话介绍这个项目？**

> 一个 RAG 回答质量评测框架——给定问题、上下文、LLM 回答，自动从忠实度、相关性、完整性、幻觉四个维度打分。支持本地 embedding 快速模式和 DeepSeek API 深度模式。

**Q2: 这个项目解决了什么问题？**

> RAG 系统上线前需要评估回答质量，但人工评审成本高、不可规模化。这个框架提供了自动化、可复现、多维度的评测方案。

**Q3: 项目的核心亮点是什么？**

> 1) 双模式架构（策略模式）——CI 用 fast、发版用 deep，零成本切换
> 2) 全本地可跑——不依赖任何外部 API，embedding 模型 470MB 离线运行
> 3) 工程化完整——TDD（20 测试）、YAML 驱动、FastAPI 服务化、GitHub Actions CI

---

### 架构 & 设计模式

**Q4: 项目用了哪些设计模式？在哪里？**

| 模式 | 位置 | 说明 |
|------|------|------|
| **策略模式** | `BaseMetric` → 4 evaluator + `LLMJudge` | 同一接口，不同算法 |
| **单例模式** | `embedding.py` `_model` | 470MB 模型只加载一次 |
| **模板方法** | `LLMJudge.evaluate()` 调 `judge()` | 父类定义骨架，子类实现细节 |
| **工厂模式** | `TestRunner.__init__()` | 根据 mode 创建不同 evaluator 组合 |

**Q5: 为什么 LLMJudge 继承 BaseMetric 而不是另起炉灶？**

> `TestRunner` 只需要知道对象有 `evaluate()` 方法。如果 LLMJudge 另起接口，runner 就得写两套调用逻辑。继承 `BaseMetric` 让 runner 对模式无感知——**依赖倒置原则**。

**Q6: 如果要加第 5 个评估维度（比如"毒性检测"），需要改哪些文件？**

> 1) 新建 `app/evaluator/toxicity.py`，实现 `BaseMetric`
> 2) 在 `runner.py` 的 fast 模式 `self.metrics` 里注册
> 3) 在 `schemas.py` 的 `EvaluateResponse` 加 `toxicity: MetricResult` 字段（或者改成动态字段）
> 4) 在 `DIMENSIONS` 常量加 `"toxicity"`
> 5) DeepSeek prompt 加毒性维度
>
> 这是**开放-封闭原则**——对扩展开放（加新评估器），对修改关闭（不改 runner 核心逻辑）。

**Q7: 为什么 embedding 模型用模块级全局变量而不是依赖注入？**

> 依赖注入在 FastAPI 里用 `Depends()` 很自然，但这里 evaluator 不一定是通过 FastAPI 调用的（CLI、测试）。模块级单例是最简单的、框架无关的方案。如果以后需要支持多种 embedding 模型热切换，再引入依赖注入容器。

---

### Embedding 技术

**Q8: 为什么选 paraphrase-multilingual-MiniLM-L12-v2？**

> - **384 维**：对短文本语义比较足够，比 768 维模型小一半、快 2-3 倍
> - **中英双语**：项目用例以中文为主，但框架本身不限制语言
> - **轻量**：470MB，CPU 推理 ~50ms/次，CI runner 不用 GPU
> - **确定性**：sentence-transformers 输出是确定性的（同输入永远同输出），适合 CI

**Q9: 和 OpenAI text-embedding-3-small 比呢？**

| | MiniLM-L12 | text-embedding-3-small |
|---|---|---|
| 维度 | 384 | 512 |
| 部署 | 本地 | API |
| 成本 | 免费 | $0.02/1M tokens |
| 延迟 | ~50ms | ~200ms（网络） |
| 离线 | ✅ | ❌ |
| 确定性 | ✅ | ✅ |

**Q10: 为什么用余弦相似度而不是欧氏距离？**

> 在语义向量空间中，向量的模长（magnitude）通常编码了文本长度等无关信息，而方向（direction）才编码语义。余弦相似度只看方向，不受文本长度影响。

**Q11: 如果上下文非常长（比如 10 页文档），embedding 方法会有什么问题？怎么解决？**

> 两个问题：1) `semantic_similarity(short_claim, long_context)` 中长文本的 embedding 会"稀释"关键信息；2) sentence-transformers 有 max_seq_length 限制（通常 256/512 tokens），超长文本会被截断。
>
> 解决方案：把长上下文按句子切块，每条 claim 取最相似的 top-K 块做验证——类似 RAG 的检索 + 验证两阶段。当前实现在 hallucination 里做了截断（`context[:500]`），这是简化版本。

---

### 评估算法

**Q12: faithfulness 的阈值 (0.6) 是怎么定的？**

> 经验值。0.6 意味着声明和上下文有中等偏上的语义重叠才认为"有支撑"。太高（0.8+）会把同义改写误判为无支撑，太低（0.3-）会把巧合的词语重叠当支撑。0.6 是通过跑一批人工标注的 case 调出来的。生产环境应该根据业务场景调整。

**Q13: 为什么 relevance 的实现这么简单（就一行 embedding）？不觉得太简陋吗？**

> 简单 != 简陋。在 RAG 场景里，relevance 本质就是问"回答和问题语义相关吗？"——这是一个典型的 semantic textual similarity (STS) 任务，embedding cosine similarity 是 SOTA 方法之一。
>
> 如果场景变复杂（比如判别"部分相关"），可以加：1) 把问题拆成子问题分别算；2) 用 cross-encoder 做更精细的相关性判断。但当前场景下，简单方案效果已经足够。

**Q14: completeness 的两阶段匹配中，子串匹配有什么局限性？**

> "腹泻"能匹配"拉肚子"吗？不能。这就是为什么需要 embedding 做第二阶段。"感到恶心想吐"能匹配"恶心"吗？子串不能（"恶心"不是子串），但 embedding 能。
>
> 反过来，embedding 可能把"腹泻"和"腹痛"搞混（都是腹部症状），但子串匹配不会——这就是两阶段的互补性。

**Q15: hallucination 检测的实体提取覆盖面够吗？能抓到所有类型的幻觉吗？**

> 当前实体提取覆盖了数字、日期、英文专名、中文引号内容——这是最常见的 4 类可验证幻觉。但以下几类抓不到：
> - **属性幻觉**："Python 运行速度快"——"快"不是实体但可能不在上下文中
> - **关系幻觉**："A 是 B 的创始人"——关系不在实体层面
> - **否定翻转**：上下文说"A 不是 B"，回答说"A 是 B"
>
> 这些需要更细粒度的自然语言推理（NLI），是 deep 模式擅长的。

**Q16: 为什么 hallucination 检测用"实体在上下文中是否存在"而不是"实体是否正确"？**

> 因为我们没有 ground truth 知识库。"正确"需要一个外部事实源。"存在"只需要验证实体是否在给定的上下文中——这是一个自包含的检查，不需要外部知识。这种设计叫 **attribution-based hallucination detection**。

---

### 工程实践

**Q17: 为什么 overall_score 是简单平均而不是加权？**

> 简单平均是最少假设的方案。不同业务对四个维度的权重不同——客服系统可能更看重 faithfulness，营销文案更看重 relevance。加权应该是调用方的事（可以在返回的四个分基础上自己加权），而不是框架的事。

**Q18: 测试里为什么有"空输入"测试？**

> 边界条件测试——空问题、空回答、空上下文。LLM 可能返回空字符串，上游检索可能失败返回空上下文。不能因为输入异常就崩溃——**健壮性 > 正确性**。

**Q19: CI 里为什么不跑 deep 模式？**

> CI 需要确定性（不能 flaky）、快速（秒级）、免费。deep 模式依赖外部 API（网络、费用、非确定性），不适合 CI。fast 模式用本地模型，满足所有 CI 要求。

**Q20: .gitignore 里忽略了什么？为什么？**

> `venv/`（虚拟环境不应进仓库）、`__pycache__/`（Python 字节码）、`.vscode/`（个人 IDE 配置）、`.DS_Store`/`Thumbs.db`（系统文件）。遵循 GitHub 的 Python .gitignore 模板。

---

### 深度技术

**Q21: 为什么 deep 模式的 fallback 返回 score=-1.0？**

> -1.0 是哨兵值——标识"此分数无效"。和 0.0 不同（0.0 是有效的低分）。调用方可以 `if score < 0: handle_error()`。

**Q22: DeepSeek prompt 里为什么 temperature=0？**

> 评分不是创意任务，需要确定性输出。temperature=0 让模型每次返回最可能的答案，减少随机性——评测工具的分数应该可复现。

**Q23: 为什么 prompt 要求"严格返回 JSON"？**

> 因为 `_parse_response` 需要解析结构化数据。不加这句话模型可能返回 markdown 包裹的 JSON 或纯文本描述。同时 `_regex_extract` 方法作为 JSON 解析失败后的兜底——**纵深防御**。

**Q24: 如果 DeepSeek API 返回格式完全错乱（JSON 和正则都解析不了），会发生什么？**

> `_regex_extract` 会返回全 0 的 scores → `_to_metric_results` 把所有维度设 0 → 最终 overall_score = 0。调用方看到 0 分就知道出问题了。加上 `logger.warning` 记录了原始内容，方便排查。

**Q25: semantic_similarity 里为什么要 `max(0.0, min(1.0, sim))`？**

> 余弦相似度理论上在 [-1, 1]，但实际应用中极少出现负值（语义向量都分布在高维空间的正象限）。浮点运算可能产生 `1.0000000000000002` 或 `-0.0000000000000001`——钳制到 [0, 1] 防止下游逻辑出错。

---

### 扩展 & 改进

**Q26: 如果要支持流式回答（streaming）评估，需要怎么改？**

> 当前实现假设回答是完整文本。流式场景需要：
> 1) 在 evaluator 里加 `evaluate_streaming(chunk)` 方法
> 2) 对 faithfulness——每个 chunk 独立检查与上下文的支撑关系
> 3) 对 hallucination——积累所有 chunk 的实体再做整体验证（不能每个 chunk 独立判，因为实体可能跨 chunk："Dennis" 在 chunk1，"Ritchie" 在 chunk2）
> 4) 实时推送分数更新给前端

**Q27: 如果要支持多语言（不只是中英），需要怎么改？**

> 1) MiniLM-L12 本身支持 50+ 语言，不需要换模型
> 2) 实体提取的正则需要扩展——日语（ひらがな/カタカナ）、韩语（한글）、阿拉伯语
> 3) 句子分割的正则需要扩展——不同语言的标点符号不同
> 4) 最好引入语言检测，按语言选择不同的正则和阈值

**Q28: 为什么没有数据库/持久化？需要加吗？**

> 当前项目是评测引擎——输入文本，输出分数，不涉及持久化。但生产环境可能需要：
> - 保存历史评测结果做趋势分析（回答质量是不是在下降？）
> - 存储人工标注作为 ground truth 来调阈值
> - 缓存 embedding 避免重复计算
>
> 数据库选择：SQLite（轻量，与"小而美"理念一致）或 PostgreSQL（生产环境）。

**Q29: embedding 计算很慢（CPU），怎么加速？**

> 1) **GPU 推理**：sentence-transformers 自动检测 CUDA，有 GPU 时 encode 快 10-50 倍
> 2) **批量 encode**：`model.encode([text1, text2, text3, ...])` 一次编码多条，利用矩阵运算并行
> 3) **模型量化**：INT8 量化模型体积减半、推理快 2-3 倍，精度损失 < 1%
> 4) **缓存**：相同文本的 embedding 只算一次，存在 dict 或 Redis 里

**Q30: 如果回答非常长（比如一篇论文），当前方法有什么问题？**

> - faithfulness：拆出几百条声明，每条都要跑 embedding → 太慢
> - relevance：整篇论文 vs 一句话问题 → embedding 被稀释
> - hallucination：实体数量爆炸，验证成本 O(n*m)
>
> 解决：对长文本先做 topic segmentation 或检索式截取相关段落，然后对相关部分做评估。

---

### 业务 & 场景

**Q31: 这个框架和 RAGAS 有什么区别？**

> RAGAS 是更全面的 RAG 评测框架（支持 context precision、context recall 等更多指标），但没有 deep 模式（LLM-as-judge）。这个项目更轻量、更适合嵌入 CI pipeline，双模式是独特卖点。

**Q32: 这个项目的评测结果能替代人工评审吗？**

> 不能完全替代。fast 模式擅长发现明显的质量问题（完全编造、完全跑题），但对微妙的质量差异（语气、风格、逻辑一致性）不敏感。deep 模式更接近人工评审，但 LLM-as-judge 有其自身的 bias。建议：fast 模式做 CI 门禁（挡住低质量），deep 模式做发版检查（辅助人工），人工评审做最终确认。

**Q33: 四个维度分数都很高，但回答实际上很烂——这可能吗？怎么防御？**

> 可能。例如回答把所有关键点列了一遍（completeness 满分），但组织混乱、逻辑不通。faithfulness/relevance/completeness/hallucination 只覆盖了"内容正确性"，不覆盖"表达质量"。防御：加第 5 个维度——coherence（连贯性），可以用 LLM 专门评估。

**Q34: 如果我把这个项目写进简历，面试官最可能问什么？**

> 1. "你这个框架和直接用 GPT-4 打分有什么区别？"→ 答双模式架构和策略模式
> 2. "为什么不用 LangChain/LlamaIndex？"→ 答：轻量、零依赖、教学目的
> 3. "生产环境敢用吗？"→ 答 CI 门禁 + 发版 deep 模式，分布部署
> 4. "模型选型的过程？"→ 答 MiniLM 的 trade-off 分析

---

### 陷阱题

**Q35: `runner.py` 里用 `from app.config import EVAL_MODE` 写在 `__init__` 里而不是文件顶部——为什么？这是 bad practice 吗？**

> **是故意的，不是 bad practice。** `config.py` 在最顶层导入时会读环境变量。如果写在文件顶部，当 pytest 收集测试文件时，环境变量还没设定好，可能读到错误的值。延迟导入（lazy import）确保在 `TestRunner` 实例化时才读环境变量——此时测试或应用已经初始化完毕。这是 Python 里处理环境变量初始化的常见模式。

**Q36: 如果 DEEPSEEK_API_KEY 没设，deep 模式会发生什么？**

> `DeepSeekJudge.__init__` 中 `os.getenv("DEEPSEEK_API_KEY", "")` 返回空字符串。调用 `_call_api` 时，`Authorization: Bearer ` 头是空的 → DeepSeek 返回 401 Unauthorized → `resp.raise_for_status()` 抛 `HTTPError` → 被 `judge()` 的 `try/except` 捕获 → `_fallback` 返回 score=-1.0。结果：四个维度全是 -1，调用方一看就知道配置有问题。

---

## 快速开始

```bash
# 1. 环境
cd D:/WorkSpace/AI测评框架
python -m venv .venv && source .venv/Scripts/activate
pip install -e ".[dev]" -i https://pypi.tuna.tsinghua.edu.cn/simple

# 2. 测试（模型首次自动下载 ~470MB，走 HF 镜像）
pytest tests/ -v

# 3. 启动服务
uvicorn app.main:app --reload
# 打开 http://localhost:8000/docs 看 Swagger

# 4. 用 YAML 跑批量测试
python -c "
from app.harness.loader import TestCaseLoader
from app.harness.runner import TestRunner
cases = TestCaseLoader.load('examples/sample_test_cases.yaml')
result = TestRunner().run_batch(cases)
print(f'通过率: {result.summary[\"pass_rate\"]*100}%')
"

# 5. 切换 DeepSeek 深度模式
export DEEPSEEK_API_KEY=sk-your-key
export EVAL_MODE=deep
uvicorn app.main:app --reload
```

---

> 📐 **设计哲学**：简单方案解决核心问题 → TDD 保证正确性 → 策略模式预留扩展空间 → 工程化（YAML + FastAPI + CI）保证可用性。
>
> 这个项目的代码量足够小到一天读完，但设计模式、工程实践、技术深度足够聊一小时面试。
