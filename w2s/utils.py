from typing import Any, Type, TypeVar, cast
from simple_parsing import Serializable

T = TypeVar("T")


def assert_type(typ: Type[T], obj: Any) -> T:
    """Assert that an object is of a given type at runtime and return it."""
    if not isinstance(obj, typ):
        raise TypeError(f"Expected {typ.__name__}, got {type(obj).__name__}")

    return cast(typ, obj)

NICKNAMES = {
    "Qwen/Qwen1.5-0.5B": "Qw0.5",
    "meta-llama/Meta-Llama-3-8B": "Ll8",
    "./results": "rs",
    "cosine": "c",
}

def get_config_foldername(config: dict) -> str:
    def shorten_key(key: str) -> str:
        return "".join(word[0] for word in key.split("_"))

    def shorten_value(value) -> str:
        if isinstance(value, bool):
            return "1" if value else "0"
        elif isinstance(value, str):
            if value in NICKNAMES:
                return NICKNAMES[value]
            value = value.split("/")[-1]
            if "_" in value:
                return "_".join(word[:4] for word in value.split("_"))
            else:
                return value
        else:
            return str(value)

    config = flatten_dict(config)
    return "-".join(
        f"{shorten_key(k)}={shorten_value(v)}" for k, v in sorted(config.items())
    )


def flatten_dict(d: dict, parent_key: str = "", sep: str = "_") -> dict:
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, Serializable):  # can't use LossConfig, etc to avoid circular import
            items.extend(flatten_dict(v.to_dict(), new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
