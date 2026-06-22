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
