from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any

from astar_island.api import AstarIslandClient, ApiError
from astar_island.learned import build_augmented_priors, build_learned_prior_model
from astar_island.config import Config
from astar_island.predict import ObservationAccumulator, summarize_prediction
from astar_island.tiling import (
    MAX_QUERY_BUDGET,
    QuerySpec,
    build_default_queries,
    build_full_cover_queries,
    choose_adaptive_refinement_queries,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Astar Island MVP runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("round-info", help="Show active round metadata")
    subparsers.add_parser("plan", help="Show the default 50-query plan")
    subparsers.add_parser("budget", help="Show current team query budget")

    train_parser = subparsers.add_parser("train-model", help="Train and tune the GBDT prior model")
    train_parser.add_argument("--runs-dir", default="runs", help="Historical run directory")
    train_parser.add_argument("--cache-db", default=".astar_cache.sqlite3", help="Local API cache database")
    train_parser.add_argument("--output-path", default="models/gbdt_prior.joblib", help="Serialized model path")
    train_parser.add_argument("--report-path", default="models/gbdt_prior_report.json", help="Training report path")
    train_parser.add_argument("--optuna-trials", type=int, default=18, help="Number of Optuna trials")
    train_parser.add_argument(
        "--feature-version",
        type=int,
        default=None,
        help="Optional feature version override for GBDT training",
    )
    train_parser.add_argument(
        "--mixer-optuna-trials",
        type=int,
        default=None,
        help="Optional override for posterior mixer tuning trials",
    )
    train_parser.add_argument(
        "--backend",
        default="auto",
        choices=("auto", "sklearn-hgbt", "lightgbm", "xgboost", "catboost"),
        help="Training backend to use",
    )

    bakeoff_parser = subparsers.add_parser("bakeoff-models", help="Run an offline backend bake-off")
    bakeoff_parser.add_argument("--runs-dir", default="runs", help="Historical run directory")
    bakeoff_parser.add_argument("--cache-db", default=".astar_cache.sqlite3", help="Local API cache database")
    bakeoff_parser.add_argument("--output-dir", default="models/bakeoff", help="Output directory for bake-off artifacts")
    bakeoff_parser.add_argument("--optuna-trials", type=int, default=18, help="Number of Optuna trials per backend")
    bakeoff_parser.add_argument(
        "--mixer-optuna-trials",
        type=int,
        default=6,
        help="Posterior mixer tuning trials per backend during bake-off screening",
    )
    bakeoff_parser.add_argument(
        "--backends",
        nargs="+",
        default=["sklearn-hgbt", "lightgbm", "xgboost", "catboost"],
        choices=("sklearn-hgbt", "lightgbm", "xgboost", "catboost"),
        help="Backends to evaluate",
    )
    bakeoff_parser.add_argument(
        "--baseline-logloss",
        type=float,
        default=0.5350,
        help="Current offline baseline that a new backend must beat",
    )
    bakeoff_parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.01,
        help="Required offline improvement over the baseline to pass the gate",
    )

    benchmark_parser = subparsers.add_parser("benchmark-models", help="Compare two model bundles on the same historical rounds")
    benchmark_parser.add_argument("--stable-model-path", required=True, help="Reference model bundle path")
    benchmark_parser.add_argument("--candidate-model-path", required=True, help="Candidate model bundle path")
    benchmark_parser.add_argument("--runs-dir", default="runs", help="Historical run directory")
    benchmark_parser.add_argument("--cache-db", default=".astar_cache.sqlite3", help="Local API cache database")
    benchmark_parser.add_argument("--recent-rounds", type=int, default=6, help="How many recent historical rounds to benchmark")

    status_parser = subparsers.add_parser("status", help="Show team score/submission status for a round")
    status_parser.add_argument("--round-id", default=None, help="Use a specific round ID")

    run_parser = subparsers.add_parser("run", help="Fetch queries, build predictions, optionally submit")
    run_parser.add_argument("--round-id", default=None, help="Use a specific round ID")
    run_parser.add_argument("--window", type=int, default=15, help="Viewport size")
    run_parser.add_argument("--refinement-budget", type=int, default=5, help="Extra repeated windows")
    run_parser.add_argument("--prior-strength", type=float, default=None, help="Empirical prior concentration")
    run_parser.add_argument("--floor", type=float, default=0.0025, help="Minimum probability floor")
    run_parser.add_argument("--adaptive-stride", type=int, default=5, help="Stride for adaptive refinement candidates")
    run_parser.add_argument("--smoothing-passes", type=int, default=0, help="Number of smoothing passes before submit")
    run_parser.add_argument("--smoothing-strength", type=float, default=0.18, help="Neighbor smoothing strength")
    run_parser.add_argument(
        "--smoothing-mode",
        choices=("plain", "edge-aware"),
        default="plain",
        help="Posterior smoothing strategy",
    )
    run_parser.add_argument("--learned-prior-blend", type=float, default=0.40, help="Blend weight for learned priors")
    run_parser.add_argument("--runs-dir", default="runs", help="Historical run directory for learned priors")
    run_parser.add_argument("--cache-db", default=".astar_cache.sqlite3", help="Local API cache database")
    run_parser.add_argument("--model-path", default="models/gbdt_prior.joblib", help="GBDT prior model path")
    run_parser.add_argument("--disable-gbdt-prior", action="store_true", help="Disable the trained GBDT prior")
    run_parser.add_argument("--disable-learned-prior", action="store_true", help="Use only heuristic priors")
    run_parser.add_argument("--force", action="store_true", help="Ignore local cache")
    run_parser.add_argument("--no-submit", action="store_true", help="Skip /submit calls")
    run_parser.add_argument("--skip-budget-check", action="store_true", help="Skip budget safety check")
    run_parser.add_argument("--output-dir", default=None, help="Directory for summary artifacts")

    return parser.parse_args()


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _ensure_output_dir(path: str | None, round_id: int) -> Path:
    if path:
        output_dir = Path(path)
    else:
        output_dir = Path("runs") / f"{round_id}_{_utc_stamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _load_round(client: AstarIslandClient, round_id: str | None, force: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    if round_id is None:
        active = client.get_active_round(force=True)
        round_id = str(active["id"])
        force = True
    else:
        active = {"id": round_id}
    detail = client.get_round_detail(round_id, force=force)
    return active, detail


def _query_to_dict(query: QuerySpec) -> dict[str, Any]:
    return {
        "round_id": query.round_id,
        "seed_index": query.seed_index,
        "viewport_x": query.viewport_x,
        "viewport_y": query.viewport_y,
        "viewport_w": query.viewport_w,
        "viewport_h": query.viewport_h,
    }


def _print_round_info(detail: dict[str, Any]) -> None:
    print(json.dumps(
        {
            "round_id": detail["id"],
            "round_number": detail.get("round_number"),
            "map_width": detail["map_width"],
            "map_height": detail["map_height"],
            "seeds_count": detail["seeds_count"],
            "initial_state_count": len(detail.get("initial_states", [])),
        },
        ensure_ascii=True,
        indent=2,
    ))


def _print_budget(budget: dict[str, Any]) -> None:
    print(json.dumps(budget, ensure_ascii=True, indent=2))


def _print_plan(queries: list[QuerySpec]) -> None:
    by_seed: dict[int, int] = {}
    for query in queries:
        by_seed[query.seed_index] = by_seed.get(query.seed_index, 0) + 1
    print(json.dumps(
        {
            "query_count": len(queries),
            "per_seed": by_seed,
            "queries": [_query_to_dict(query) for query in queries],
        },
        ensure_ascii=True,
        indent=2,
    ))


def _extract_score_like_fields(payload: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}

    def walk(node: Any, prefix: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                new_prefix = f"{prefix}.{key}" if prefix else key
                if any(token in key.lower() for token in ("score", "point", "rank")):
                    result[new_prefix] = value
                walk(value, new_prefix)
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{prefix}[{index}]")

    walk(payload, "")
    return result


def _find_round_status_entry(rounds: list[dict[str, Any]], round_id: str) -> dict[str, Any] | None:
    return next((item for item in rounds if str(item.get("id")) == str(round_id)), None)


def _emit_status(client: AstarIslandClient, round_id: str) -> dict[str, Any]:
    my_rounds = client.get_my_rounds(force=True)
    my_predictions = client.get_my_predictions(round_id, force=True)
    round_entry = _find_round_status_entry(my_rounds, round_id)

    status_payload = {
        "round_id": round_id,
        "team_round_status": round_entry,
        "predictions": my_predictions,
    }
    print(json.dumps(status_payload, ensure_ascii=True, indent=2))
    return status_payload


def _run(args: argparse.Namespace) -> int:
    config = Config.from_env()
    client = AstarIslandClient(config)
    _, detail = _load_round(client, args.round_id, args.force)
    round_id = str(detail["id"])

    if not args.skip_budget_check:
        budget = client.get_budget(force=True)
        available_queries = int(budget["queries_max"]) - int(budget["queries_used"])
        mandatory_cover_queries = len(
            build_full_cover_queries(
                round_id=round_id,
                seeds_count=int(detail["seeds_count"]),
                width=int(detail["map_width"]),
                height=int(detail["map_height"]),
                window=args.window,
            )
        )
        if available_queries < mandatory_cover_queries:
            raise ApiError(
                f"Need at least {mandatory_cover_queries} queries for full coverage, "
                f"but only {available_queries} remain"
            )
        args.refinement_budget = min(args.refinement_budget, available_queries - mandatory_cover_queries)

    output_dir = _ensure_output_dir(args.output_dir, round_id)
    cover_queries = build_full_cover_queries(
        round_id=round_id,
        seeds_count=int(detail["seeds_count"]),
        width=int(detail["map_width"]),
        height=int(detail["map_height"]),
        window=args.window,
    )

    width = int(detail["map_width"])
    height = int(detail["map_height"])
    seeds_count = int(detail["seeds_count"])
    gbdt_bundle = None
    posterior_mixer = None
    gbdt_runtime_fallback_reason: str | None = None
    if not args.disable_gbdt_prior:
        model_path = Path(args.model_path)
        if model_path.exists():
            from astar_island.gbdt import load_gbdt_prior_bundle

            gbdt_bundle = load_gbdt_prior_bundle(model_path)
            posterior_mixer = gbdt_bundle.posterior_mixer()
        else:
            print(f"GBDT prior skipped because model file was not found at {model_path}")

    learned_model = None
    if gbdt_bundle is not None:
        try:
            gbdt_bundle.predict_prior_grid(detail["initial_states"][0])
        except Exception as exc:
            gbdt_runtime_fallback_reason = str(exc)
            print(f"GBDT prior disabled due to runtime incompatibility: {exc}")
            gbdt_bundle = None
            posterior_mixer = None
    if gbdt_bundle is None and not args.disable_learned_prior:
        learned_model = build_learned_prior_model(args.runs_dir, args.cache_db)

    resolved_prior_strength = (
        args.prior_strength
        if args.prior_strength is not None
        else (gbdt_bundle.prior_strength if gbdt_bundle is not None else 0.35)
    )
    learned_diagnostics: list[dict[str, Any]] = []
    accumulators = []
    for seed_index in range(seeds_count):
        initial_state = detail["initial_states"][seed_index]
        if gbdt_bundle is not None:
            cell_priors = gbdt_bundle.predict_prior_grid(initial_state)
            diagnostics = {
                "enabled": 1,
                "training_examples": gbdt_bundle.training_examples,
                "matched_cells": len(cell_priors) * (len(cell_priors[0]) if cell_priors else 0),
                "total_cells": len(cell_priors) * (len(cell_priors[0]) if cell_priors else 0),
                "source": "gbdt",
                "model_blend": gbdt_bundle.model_blend,
                "temperature": gbdt_bundle.temperature,
                "cv_prequential_logloss": gbdt_bundle.cv_prequential_logloss,
            }
        else:
            cell_priors, diagnostics = build_augmented_priors(
                initial_state,
                learned_model,
                learned_blend=args.learned_prior_blend,
            )
            diagnostics["source"] = "frequency"
        grid = initial_state["grid"]
        accumulators.append(
            ObservationAccumulator(
                width=len(grid[0]) if grid else 0,
                height=len(grid),
                cell_priors=cell_priors,
                posterior_mixer=posterior_mixer,
            )
        )
        learned_diagnostics.append({"seed_index": seed_index, **diagnostics})
    observations_log: list[dict[str, Any]] = []
    simulate_payloads: list[dict[str, Any]] = []

    refinement_queries: list[QuerySpec] = []
    total_queries = len(cover_queries) + args.refinement_budget

    for index, query in enumerate(cover_queries, start=1):
        result = client.simulate(query)
        accumulators[query.seed_index].add_patch(result["viewport"], result["grid"])
        observations_log.append(
            {
                "query": _query_to_dict(query),
                "queries_used": result.get("queries_used"),
                "queries_max": result.get("queries_max"),
                "settlements_count": len(result.get("settlements", [])),
            }
        )
        simulate_payloads.append(
            {
                "query": _query_to_dict(query),
                "viewport": result.get("viewport"),
                "grid": result.get("grid"),
                "settlements": result.get("settlements", []),
                "queries_used": result.get("queries_used"),
                "queries_max": result.get("queries_max"),
                "phase": "cover",
            }
        )
        print(
            f"[{index:02d}/{total_queries}] seed={query.seed_index} "
            f"x={query.viewport_x} y={query.viewport_y} "
            f"w={query.viewport_w} h={query.viewport_h} "
            f"used={result.get('queries_used')}/{result.get('queries_max')}"
        )

    excluded_refinement_windows: set[tuple[int, int, int]] = set()
    for offset in range(len(cover_queries) + 1, total_queries + 1):
        next_queries = choose_adaptive_refinement_queries(
            round_id=round_id,
            detail=detail,
            accumulators=accumulators,
            budget=1,
            window=args.window,
            stride=args.adaptive_stride,
            prior_strength=resolved_prior_strength,
            excluded_windows=excluded_refinement_windows,
        )
        if not next_queries:
            break
        query = next_queries[0]
        excluded_refinement_windows.add((query.seed_index, query.viewport_x, query.viewport_y))
        refinement_queries.append(query)
        result = client.simulate(query)
        accumulators[query.seed_index].add_patch(result["viewport"], result["grid"])
        observations_log.append(
            {
                "query": _query_to_dict(query),
                "queries_used": result.get("queries_used"),
                "queries_max": result.get("queries_max"),
                "settlements_count": len(result.get("settlements", [])),
                "phase": "adaptive_refinement",
            }
        )
        simulate_payloads.append(
            {
                "query": _query_to_dict(query),
                "viewport": result.get("viewport"),
                "grid": result.get("grid"),
                "settlements": result.get("settlements", []),
                "queries_used": result.get("queries_used"),
                "queries_max": result.get("queries_max"),
                "phase": "adaptive_refinement",
            }
        )
        print(
            f"[{offset:02d}/{total_queries}] seed={query.seed_index} "
            f"x={query.viewport_x} y={query.viewport_y} "
            f"w={query.viewport_w} h={query.viewport_h} "
            f"used={result.get('queries_used')}/{result.get('queries_max')} "
            f"phase=adaptive_refinement"
        )

    queries = cover_queries + refinement_queries

    predictions = []
    coverage_summary = []
    for seed_index, accumulator in enumerate(accumulators):
        prediction = accumulator.to_prediction_tensor(
            prior_strength=resolved_prior_strength,
            floor=args.floor,
            smoothing_passes=args.smoothing_passes,
            smoothing_strength=args.smoothing_strength,
            smoothing_mode=args.smoothing_mode,
        )
        predictions.append(prediction)
        coverage_summary.append(
            {
                "seed_index": seed_index,
                "coverage_ratio": accumulator.full_coverage_ratio(),
                "learned_prior": learned_diagnostics[seed_index],
                "prediction_summary": summarize_prediction(prediction),
            }
        )

    submit_responses: list[dict[str, Any]] = []
    status_payload: dict[str, Any] | None = None
    if args.no_submit:
        print("Submit skipped because --no-submit was passed.")
    else:
        for seed_index, prediction in enumerate(predictions):
            result = client.submit(round_id=round_id, seed_index=seed_index, prediction=prediction)
            score_like = _extract_score_like_fields(result.response)
            print(f"submit seed={seed_index} response={json.dumps(result.response, ensure_ascii=True)}")
            if score_like:
                print(f"submit seed={seed_index} score_fields={json.dumps(score_like, ensure_ascii=True)}")
            submit_responses.append(result.to_dict())
        time.sleep(1.0)
        status_payload = _emit_status(client, round_id)

    summary = {
        "round_id": round_id,
        "run_config": {
            "window": args.window,
            "refinement_budget": args.refinement_budget,
            "prior_strength": resolved_prior_strength,
            "floor": args.floor,
            "adaptive_stride": args.adaptive_stride,
            "smoothing_passes": args.smoothing_passes,
            "smoothing_strength": args.smoothing_strength,
            "smoothing_mode": args.smoothing_mode,
            "learned_prior_blend": args.learned_prior_blend,
            "model_path": args.model_path,
            "disable_gbdt_prior": args.disable_gbdt_prior,
            "disable_learned_prior": args.disable_learned_prior,
            "runs_dir": args.runs_dir,
            "cache_db": args.cache_db,
        },
        "query_count": len(queries),
        "gbdt_prior": {
            "enabled": 0 if gbdt_bundle is None else 1,
            "runtime_fallback_reason": gbdt_runtime_fallback_reason,
            "training_examples": 0 if gbdt_bundle is None else gbdt_bundle.training_examples,
            "posterior_mode": None if gbdt_bundle is None else gbdt_bundle.posterior_mode,
            "ensemble_params": None if gbdt_bundle is None else gbdt_bundle.ensemble_params,
            "model_blend": None if gbdt_bundle is None else gbdt_bundle.model_blend,
            "temperature": None if gbdt_bundle is None else gbdt_bundle.temperature,
            "cv_prequential_logloss": None if gbdt_bundle is None else gbdt_bundle.cv_prequential_logloss,
            "gate_params": None if gbdt_bundle is None else gbdt_bundle.gate_params,
            "bucket_temperatures": None if gbdt_bundle is None else gbdt_bundle.bucket_temperatures,
        },
        "learned_prior": {
            "enabled": 0 if args.disable_learned_prior else 1,
            "active_source": (
                "gbdt"
                if gbdt_bundle is not None
                else "frequency"
                if learned_model is not None
                else "heuristic"
            ),
            "used_frequency_model": 1 if learned_model is not None and gbdt_bundle is None else 0,
            "model_examples": 0 if learned_model is None else learned_model.total_examples,
            "seed_diagnostics": learned_diagnostics,
        },
        "coverage_summary": coverage_summary,
        "queries": [_query_to_dict(query) for query in queries],
        "observations_log": observations_log,
        "submit_responses": submit_responses,
        "status_payload": status_payload,
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    (output_dir / "queries.json").write_text(
        json.dumps([_query_to_dict(query) for query in queries], ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    (output_dir / "submit_responses.json").write_text(
        json.dumps(submit_responses, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    (output_dir / "observations_log.json").write_text(
        json.dumps(observations_log, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    (output_dir / "simulate_payloads.json").write_text(
        json.dumps(simulate_payloads, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    (output_dir / "status_payload.json").write_text(
        json.dumps(status_payload, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    print(f"Artifacts saved to {output_dir}")
    print(json.dumps({"coverage_summary": coverage_summary}, ensure_ascii=True, indent=2))
    return 0


def main() -> int:
    args = _parse_args()
    try:
        if args.command == "train-model":
            from astar_island.gbdt import FEATURE_VERSION, train_gbdt_prior_bundle

            bundle, report = train_gbdt_prior_bundle(
                runs_dir=args.runs_dir,
                cache_db=args.cache_db,
                output_path=args.output_path,
                report_path=args.report_path,
                optuna_trials=args.optuna_trials,
                backend=args.backend,
                mixer_optuna_trials=args.mixer_optuna_trials,
                feature_version=args.feature_version if args.feature_version is not None else FEATURE_VERSION,
            )
            print(
                json.dumps(
                    {
                        "output_path": str(args.output_path),
                        "backend": bundle.backend,
                        "mixer_optuna_trials": args.mixer_optuna_trials,
                        "training_examples": bundle.training_examples,
                        "training_round_ids": bundle.training_round_ids,
                        "posterior_mode": bundle.posterior_mode,
                        "ensemble_params": bundle.ensemble_params,
                        "model_blend": bundle.model_blend,
                        "prior_strength": bundle.prior_strength,
                        "temperature": bundle.temperature,
                        "cv_prequential_logloss": bundle.cv_prequential_logloss,
                        "gate_params": bundle.gate_params,
                        "bucket_temperatures": bundle.bucket_temperatures,
                        "report": report,
                    },
                    ensure_ascii=True,
                    indent=2,
                )
            )
            return 0

        if args.command == "bakeoff-models":
            from astar_island.gbdt import run_backend_bakeoff

            report = run_backend_bakeoff(
                runs_dir=args.runs_dir,
                cache_db=args.cache_db,
                output_dir=args.output_dir,
                optuna_trials=args.optuna_trials,
                mixer_optuna_trials=args.mixer_optuna_trials,
                backends=args.backends,
                baseline_logloss=args.baseline_logloss,
                min_improvement=args.min_improvement,
            )
            print(json.dumps(report, ensure_ascii=True, indent=2))
            return 0

        if args.command == "benchmark-models":
            from astar_island.gbdt import compare_bundle_paths_on_recent_history

            report = compare_bundle_paths_on_recent_history(
                stable_model_path=args.stable_model_path,
                candidate_model_path=args.candidate_model_path,
                runs_dir=args.runs_dir,
                cache_db=args.cache_db,
                recent_rounds=args.recent_rounds,
            )
            print(json.dumps(report, ensure_ascii=True, indent=2))
            return 0

        config = Config.from_env()
        client = AstarIslandClient(config)

        if args.command == "round-info":
            _, detail = _load_round(client, None, False)
            _print_round_info(detail)
            return 0

        if args.command == "plan":
            _, detail = _load_round(client, None, False)
            queries = build_default_queries(
                round_id=str(detail["id"]),
                detail=detail,
            )
            _print_plan(queries)
            return 0

        if args.command == "budget":
            _print_budget(client.get_budget(force=True))
            return 0

        if args.command == "status":
            _, detail = _load_round(client, args.round_id, False)
            _emit_status(client, str(detail["id"]))
            return 0

        if args.command == "run":
            return _run(args)
    except (ApiError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
