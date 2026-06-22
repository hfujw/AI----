"""DeepSeek 深度评判器——一次 API 调用评估四个维度"""
import json
import logging
import os
import re
import requests
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
        resp = requests.post(
            self.BASE_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "你是精确的 RAG 评测专家。只返回 JSON。"},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
                "max_tokens": 500,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def _parse_response(self, response) -> dict:
        content = response["choices"][0]["message"]["content"]
        # 提取 ```json ... ``` 代码块中的 JSON
        if "```" in content:
            match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
            if match:
                content = match.group(1)
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # 尝试用正则从非标准格式中提取分数
            logger.warning(f"JSON 解析失败，尝试正则提取: {content[:200]}")
            return self._regex_extract(content)

    def _regex_extract(self, content: str) -> dict:
        """非标准 JSON 格式的降级提取——从任意文本中抠分数"""
        scores = {}
        for dim in ["faithfulness", "relevance", "completeness", "hallucination"]:
            m = re.search(rf'"{dim}"\s*:\s*(\d+)', content)
            scores[dim] = int(m.group(1)) if m else 0
        return scores

    def _to_metric_results(self, scores):
        reasons = scores.get("reasons", {})
        return {
            "faithfulness": MetricResult(
                score=scores.get("faithfulness", 0) / 10.0,
                details={
                    "reason": reasons.get("faithfulness", ""),
                    "raw": scores.get("faithfulness"),
                },
            ),
            "relevance": MetricResult(
                score=scores.get("relevance", 0) / 10.0,
                details={
                    "reason": reasons.get("relevance", ""),
                    "raw": scores.get("relevance"),
                },
            ),
            "completeness": MetricResult(
                score=scores.get("completeness", 0) / 10.0,
                details={
                    "reason": reasons.get("completeness", ""),
                    "raw": scores.get("completeness"),
                },
            ),
            "hallucination": MetricResult(
                score=scores.get("hallucination", 0) / 10.0,
                details={
                    "reason": reasons.get("hallucination", ""),
                    "raw": scores.get("hallucination"),
                },
            ),
        }

    def _fallback(self, error):
        return {
            name: MetricResult(score=-1.0, details={"error": error})
            for name in ["faithfulness", "relevance", "completeness", "hallucination"]
        }
