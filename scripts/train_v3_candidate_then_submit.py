from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from astar_island.api import AstarIslandClient
from astar_island.config import Config

from round_submit_common import choose_model_path, should_submit_round, run_live, repo_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round-id", required=True)
    parser.add_argument("--candidate-model-path", required=True)
    parser.add_argument("--candidate-report-path", required=True)
    parser.add_argument("--stable-model-path", required=True)
    parser.add_argument("--stable-report-path", required=True)
    parser.add_argument("--backend", default="sklearn-hgbt")
    parser.add_argument("--feature-version", type=int, default=3)
    parser.add_argument("--optuna-trials", type=int, default=18)
    parser.add_argument("--mixer-optuna-trials", type=int, default=12)
    parser.add_argument("--min-improvement", type=float, default=0.003)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    env.setdefault("LOKY_MAX_CPU_COUNT", "8")

    train_cmd = [
        str(root / ".venv" / "bin" / "python"),
        "-m",
        "astar_island.cli",
        "train-model",
        "--backend",
        args.backend,
        "--feature-version",
        str(args.feature_version),
        "--optuna-trials",
        str(args.optuna_trials),
        "--mixer-optuna-trials",
        str(args.mixer_optuna_trials),
        "--output-path",
        args.candidate_model_path,
        "--report-path",
        args.candidate_report_path,
    ]
    train_result = subprocess.run(train_cmd, cwd=root, env=env)
    if train_result.returncode != 0:
        return int(train_result.returncode)

    client = AstarIslandClient(Config.from_env())
    if not should_submit_round(client, args.round_id):
        return 0

    model_path, reason = choose_model_path(
        stable_model_path=args.stable_model_path,
        stable_report_path=args.stable_report_path,
        candidate_model_path=args.candidate_model_path,
        candidate_report_path=args.candidate_report_path,
        min_improvement=args.min_improvement,
        runs_dir=root / "runs",
        cache_db=root / ".astar_cache.sqlite3",
    )
    print(f"Submitting round {args.round_id} with {model_path} ({reason})", flush=True)
    return run_live(args.round_id, model_path)


if __name__ == "__main__":
    raise SystemExit(main())
