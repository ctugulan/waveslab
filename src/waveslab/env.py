from __future__ import annotations

import os
from pathlib import Path


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def env_optional_float(name: str, default: float | None = None) -> float | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    if raw.strip().lower() in {"none", "null", "default", "inherit"}:
        return None
    return float(raw)


def env_str(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


def env_float_tuple(name: str, default: tuple[float, ...]) -> tuple[float, ...]:
    raw = os.environ.get(name)
    if not raw:
        return default
    return tuple(float(x.strip()) for x in raw.replace(";", ",").split(",") if x.strip())


def env_str_tuple(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.environ.get(name)
    if not raw:
        return default
    return tuple(x.strip().lower() for x in raw.replace(";", ",").split(",") if x.strip())


def env_path(name: str, default: str | Path | None = None) -> Path | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        if default is None:
            return None
        return Path(default).expanduser().resolve()
    return Path(raw).expanduser().resolve()


def slug_float(value: float) -> str:
    text = f"{float(value):.6g}"
    return text.replace("-", "m").replace("+", "p").replace(".", "p")


def safe_stem(*parts: object) -> str:
    text = "__".join(str(p) for p in parts if str(p).strip())
    text = text.replace(" ", "_").replace("/", "_").replace("\\", "_")
    out = []
    for ch in text:
        out.append(ch if (ch.isalnum() or ch in {"_", "-", "."}) else "_")
    stem = "".join(out)
    while "___" in stem:
        stem = stem.replace("___", "__")
    return stem.strip("_")
