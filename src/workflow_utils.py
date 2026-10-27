"""Lightweight utilities shared by workflow planning and simulation code."""


def str2bool(value):
    if isinstance(value, bool):
        return value
    if not isinstance(value, str):
        raise ValueError(f"Cannot convert {value!r} to bool")
    value = value.strip().lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean string: {value!r}")
