import json


def serialize_json(value, **kwargs) -> str:
    """序列化为 JSON，并将 JSON 不支持的对象转换为字符串。"""
    kwargs.setdefault("ensure_ascii", False)
    kwargs.setdefault("default", str)
    return json.dumps(value, **kwargs)

