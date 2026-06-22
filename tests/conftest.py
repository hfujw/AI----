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
