import json
import glob

for summary_file in sorted(glob.glob("runs/*/summary.json")):
    with open(summary_file) as f:
        data = json.load(f)
        status = data.get("status_payload", {}) or {}
        team_status = status.get("team_round_status", {}) or {}
        score = team_status.get("score")
        rank = team_status.get("rank")
        print(f"Run: {summary_file}")
        print(f"  Score: {score}")
        print(f"  Rank:  {rank}")
