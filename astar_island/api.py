from __future__ import annotations

import json
from dataclasses import asdict, dataclass
import ssl
import time
from typing import Any
from urllib import error, request

from astar_island.cache import SqliteJsonCache
from astar_island.config import Config
from astar_island.tiling import QuerySpec


class ApiError(RuntimeError):
    """Raised when the Astar Island API returns an error."""


@dataclass
class SubmitResult:
    seed_index: int
    response: Any

    def to_dict(self) -> dict[str, Any]:
        return {"seed_index": self.seed_index, "response": self.response}


class AstarIslandClient:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.cache = SqliteJsonCache(config.cache_db)
        self._last_simulate_at = 0.0
        self._last_submit_at = 0.0

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.config.auth_mode == "bearer":
            headers["Authorization"] = f"Bearer {self.config.token}"
        else:
            headers["Cookie"] = f"access_token={self.config.token}"
        return headers

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        url = f"{self.config.base_url}{path}"
        headers = self._headers()
        data: bytes | None = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(payload).encode("utf-8")

        req = request.Request(url=url, headers=headers, data=data, method=method.upper())

        try:
            ssl_context = None
            if self.config.insecure_ssl:
                ssl_context = ssl._create_unverified_context()
            with request.urlopen(req, timeout=self.config.timeout_seconds, context=ssl_context) as resp:
                raw = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise ApiError(f"HTTP {exc.code} for {method} {path}: {body}") from exc
        except error.URLError as exc:
            raise ApiError(f"Request failed for {method} {path}: {exc}") from exc

        if not raw:
            return {}

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"raw": raw}

    def _throttle(self, kind: str) -> None:
        now = time.monotonic()
        if kind == "simulate":
            min_interval = 0.22
            elapsed = now - self._last_simulate_at
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            self._last_simulate_at = time.monotonic()
            return

        if kind == "submit":
            min_interval = 0.55
            elapsed = now - self._last_submit_at
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            self._last_submit_at = time.monotonic()

    def get_rounds(self, force: bool = False) -> list[dict[str, Any]]:
        cache_key = "get:/astar-island/rounds"
        if not force:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached
        payload = self._request_json("GET", "/astar-island/rounds")
        self.cache.set(cache_key, payload)
        return payload

    def get_active_round(self, force: bool = False) -> dict[str, Any]:
        rounds = self.get_rounds(force=force)
        active = next((item for item in rounds if item.get("status") == "active"), None)
        if active is None:
            raise ApiError("No active round found")
        return active

    def get_round_detail(self, round_id: int, force: bool = False) -> dict[str, Any]:
        cache_key = f"get:/astar-island/rounds/{round_id}"
        if not force:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached
        payload = self._request_json("GET", f"/astar-island/rounds/{round_id}")
        self.cache.set(cache_key, payload)
        return payload

    def get_budget(self, force: bool = False) -> dict[str, Any]:
        cache_key = "get:/astar-island/budget"
        if not force:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached
        payload = self._request_json("GET", "/astar-island/budget")
        self.cache.set(cache_key, payload)
        return payload

    def get_my_rounds(self, force: bool = False) -> list[dict[str, Any]]:
        cache_key = "get:/astar-island/my-rounds"
        if not force:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached
        payload = self._request_json("GET", "/astar-island/my-rounds")
        self.cache.set(cache_key, payload)
        return payload

    def get_my_predictions(self, round_id: str, force: bool = False) -> list[dict[str, Any]]:
        cache_key = f"get:/astar-island/my-predictions/{round_id}"
        if not force:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached
        payload = self._request_json("GET", f"/astar-island/my-predictions/{round_id}")
        self.cache.set(cache_key, payload)
        return payload

    def get_analysis(self, round_id: str, seed_index: int) -> dict[str, Any]:
        return self._request_json("GET", f"/astar-island/analysis/{round_id}/{seed_index}")

    def simulate(self, query: QuerySpec) -> dict[str, Any]:
        self._throttle("simulate")
        payload = self._request_json("POST", "/astar-island/simulate", asdict(query))
        return payload

    def submit(self, round_id: str, seed_index: int, prediction: list[list[list[float]]]) -> SubmitResult:
        self._throttle("submit")
        payload = {
            "round_id": round_id,
            "seed_index": seed_index,
            "prediction": prediction,
        }
        response = self._request_json("POST", "/astar-island/submit", payload)
        return SubmitResult(seed_index=seed_index, response=response)
