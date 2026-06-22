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

import os
# 国内默认走 HF 镜像，避免直连超时
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

_model = None


def get_model():
    """获取全局唯一的模型实例（懒加载）"""
    global _model
    if _model is None:
        from app.config import EMBEDDING_MODEL
        _model = SentenceTransformer(EMBEDDING_MODEL)
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
