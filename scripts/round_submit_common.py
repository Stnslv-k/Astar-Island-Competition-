from __future__ import annotations

import json
import subprocess
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from astar_island.api import AstarIslandClient
from astar_island.config import Config
from astar_island.gbdt import compare_bundle_paths_on_recent_history


def repo_root() -> Path:
    return ROOT


def load_report(path: str | Path) -> dict[str, Any] | None:
    report_path = Path(path)
    if not report_path.exists():
        return None
    return json.loads(report_path.read_text(encoding="utf-8"))


def choose_model_path(
    stable_model_path: str | Path,
    stable_report_path: str | Path,
    candidate_model_path: str | Path,
    candidate_report_path: str | Path,
    min_improvement: float,
    runs_dir: str | Path = "runs",
    cache_db: str | Path = ".astar_cache.sqlite3",
    recent_benchmark_rounds: int = 6,
) -> tuple[Path, str]:
    stable_model = Path(stable_model_path)
    candidate_model = Path(candidate_model_path)
    stable_report = load_report(stable_report_path)
    candidate_report = load_report(candidate_report_path)

    stable_cv = None if stable_report is None else stable_report.get("cv_prequential_logloss")
    candidate_cv = None if candidate_report is None else candidate_report.get("cv_prequential_logloss")

    if candidate_model.exists() and stable_model.exists():
        try:
            benchmark = compare_bundle_paths_on_recent_history(
                stable_model_path=stable_model,
                candidate_model_path=candidate_model,
                runs_dir=runs_dir,
                cache_db=cache_db,
                recent_rounds=recent_benchmark_rounds,
            )
        except Exception:
            benchmark = None
        else:
            delta = benchmark.get("candidate_minus_stable")
            if isinstance(delta, float) and delta <= -float(min_improvement):
                candidate_avg = benchmark["candidate"]["average_prequential_logloss"]
                stable_avg = benchmark["stable"]["average_prequential_logloss"]
                return candidate_model, f"benchmark_recent:{candidate_avg:.6f}<{stable_avg:.6f}"

    if (
        candidate_report is not None
        and candidate_model.exists()
        and isinstance(candidate_cv, (int, float))
        and isinstance(stable_cv, (int, float))
        and float(candidate_cv) <= float(stable_cv) - float(min_improvement)
    ):
        return candidate_model, f"candidate:{candidate_cv:.6f}"

    return stable_model, "stable"


def round_status(client: AstarIslandClient, round_id: str) -> dict[str, Any] | None:
    my_rounds = client.get_my_rounds(force=True)
    return next((item for item in my_rounds if str(item.get("id")) == str(round_id)), None)


def should_submit_round(client: AstarIslandClient, round_id: str) -> bool:
    status = round_status(client, round_id)
    if status is None:
        detail = client.get_round_detail(round_id, force=True)
        return detail.get("status") == "active"
    if status.get("status") != "active":
        return False
    return int(status.get("seeds_submitted") or 0) < 5


def run_live(round_id: str, model_path: str | Path) -> int:
    cmd = [
        "python3",
        "-m",
        "astar_island.cli",
        "run",
        "--round-id",
        str(round_id),
        "--model-path",
        str(model_path),
    ]
    completed = subprocess.run(cmd, cwd=repo_root())
    return int(completed.returncode)
