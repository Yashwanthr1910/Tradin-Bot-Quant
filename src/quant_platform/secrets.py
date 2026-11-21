"""Minimal local .env handling for provider credentials; values are never logged."""
from __future__ import annotations

import os
from pathlib import Path


def load_local_env(path: str | Path = ".env") -> dict[str, str]:
    values: dict[str, str] = {}
    file_path = Path(path)
    if file_path.exists():
        for raw_line in file_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    for key in list(values):
        if os.environ.get(key):
            values[key] = os.environ[key]
    return values


def update_local_env(key: str, value: str, path: str | Path = ".env") -> None:
    file_path = Path(path)
    lines = file_path.read_text(encoding="utf-8").splitlines() if file_path.exists() else []
    prefix = f"{key}="
    replaced = False
    output: list[str] = []
    for line in lines:
        if line.startswith(prefix):
            output.append(f"{prefix}{value}")
            replaced = True
        else:
            output.append(line)
    if not replaced:
        output.append(f"{prefix}{value}")
    file_path.write_text("\n".join(output) + "\n", encoding="utf-8")
