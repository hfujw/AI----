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
COMPLETENESS_THRESHOLD = float(os.getenv("COMPLETENESS_THRESHOLD", "0.5"))

# ── DeepSeek 配置（仅 EVAL_MODE="deep" 时使用）─────────────
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# ── 日志 ──────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
