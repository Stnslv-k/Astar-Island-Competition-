from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass


def _load_local_env_files() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    candidates = (Path.cwd() / ".env.local", Path.cwd() / ".env", repo_root / ".env.local", repo_root / ".env")
    for path in candidates:
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key or key in os.environ:
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            os.environ[key] = value


@dataclass(frozen=True)
class Config:
    base_url: str
    token: str
    auth_mode: str
    cache_db: str
    timeout_seconds: int
    insecure_ssl: bool

    @classmethod
    def from_env(cls) -> "Config":
        _load_local_env_files()
        token = os.environ.get("ASTAR_TOKEN", "").strip()
        if not token:
            raise ValueError("ASTAR_TOKEN is required")

        auth_mode = os.environ.get("ASTAR_AUTH_MODE", "bearer").strip().lower()
        if auth_mode not in {"bearer", "cookie"}:
            raise ValueError("ASTAR_AUTH_MODE must be 'bearer' or 'cookie'")

        timeout_raw = os.environ.get("ASTAR_TIMEOUT", "30").strip()
        timeout_seconds = int(timeout_raw)

        return cls(
            base_url=os.environ.get("ASTAR_BASE", "https://api.ainm.no").rstrip("/"),
            token=token,
            auth_mode=auth_mode,
            cache_db=os.environ.get("ASTAR_CACHE_DB", ".astar_cache.sqlite3"),
            timeout_seconds=timeout_seconds,
            insecure_ssl=os.environ.get("ASTAR_INSECURE_SSL", "").strip() in {"1", "true", "TRUE", "yes"},
        )
