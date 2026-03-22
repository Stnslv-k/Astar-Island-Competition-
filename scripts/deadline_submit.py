from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from astar_island.api import AstarIslandClient
from astar_island.config import Config

from round_submit_common import choose_model_path, run_live, should_submit_round


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round-id", required=True)
    parser.add_argument("--candidate-model-path", required=True)
    parser.add_argument("--candidate-report-path", required=True)
    parser.add_argument("--stable-model-path", required=True)
    parser.add_argument("--stable-report-path", required=True)
    parser.add_argument("--min-improvement", type=float, default=0.003)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    client = AstarIslandClient(Config.from_env())
    if not should_submit_round(client, args.round_id):
        return 0

    model_path, reason = choose_model_path(
        stable_model_path=args.stable_model_path,
        stable_report_path=args.stable_report_path,
        candidate_model_path=args.candidate_model_path,
        candidate_report_path=args.candidate_report_path,
        min_improvement=args.min_improvement,
    )
    print(f"Deadline submit for round {args.round_id} with {model_path} ({reason})", flush=True)
    return run_live(args.round_id, model_path)


if __name__ == "__main__":
    raise SystemExit(main())
