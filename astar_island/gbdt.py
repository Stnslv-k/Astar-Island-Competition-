from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass, field, replace
from math import log
from pathlib import Path
import sqlite3
from typing import Any

from astar_island.predict import ConfidenceEnsembleMixer, LearnedPosteriorMixer, build_cell_posterior
from astar_island.terrain import N_CLASSES, build_initial_priors, normalize, terrain_value_to_submission_class

try:
    import joblib
    import numpy as np
except ImportError as exc:  # pragma: no cover - exercised in environments without ML deps
    joblib = None
    np = None
    _ML_IMPORT_ERROR = exc
else:
    _ML_IMPORT_ERROR = None

try:  # pragma: no cover - tuning is optional at runtime
    import optuna
except ImportError:
    optuna = None

try:  # pragma: no cover - backend availability depends on local runtime
    from lightgbm import LGBMClassifier
except Exception as exc:
    LGBMClassifier = None
    _LIGHTGBM_IMPORT_ERROR = exc
else:
    _LIGHTGBM_IMPORT_ERROR = None

try:  # pragma: no cover - backend availability depends on local runtime
    from xgboost import XGBClassifier
except Exception as exc:
    XGBClassifier = None
    _XGBOOST_IMPORT_ERROR = exc
else:
    _XGBOOST_IMPORT_ERROR = None

try:  # pragma: no cover - backend availability depends on local runtime
    from catboost import CatBoostClassifier
except Exception as exc:
    CatBoostClassifier = None
    _CATBOOST_IMPORT_ERROR = exc
else:
    _CATBOOST_IMPORT_ERROR = None

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
except ImportError:
    HistGradientBoostingClassifier = None


DEFAULT_MODEL_PATH = Path("models/gbdt_prior.joblib")
DEFAULT_REPORT_PATH = Path("models/gbdt_prior_report.json")
FEATURE_VERSION = 4
EPS = 1e-12
RAW_TERRAIN_VALUES = (0, 1, 2, 3, 4, 5, 10, 11)
RAW_TERRAIN_TO_INDEX = {value: index for index, value in enumerate(RAW_TERRAIN_VALUES)}
OCEAN_RAW_VALUE = 10
SUPPORTED_BACKENDS = ("auto", "sklearn-hgbt", "lightgbm", "xgboost", "catboost")


def _require_ml_dependencies() -> None:
    if _ML_IMPORT_ERROR is not None:
        raise RuntimeError(
            "ML dependencies are not available. Install lightgbm, scikit-learn-compatible "
            "dependencies, and optuna before using the GBDT pipeline."
        ) from _ML_IMPORT_ERROR


def _backend_availability() -> dict[str, tuple[bool, Exception | None]]:
    return {
        "sklearn-hgbt": (HistGradientBoostingClassifier is not None, None if HistGradientBoostingClassifier is not None else RuntimeError("scikit-learn HistGradientBoostingClassifier is unavailable")),
        "lightgbm": (LGBMClassifier is not None, _LIGHTGBM_IMPORT_ERROR),
        "xgboost": (XGBClassifier is not None, _XGBOOST_IMPORT_ERROR),
        "catboost": (CatBoostClassifier is not None, _CATBOOST_IMPORT_ERROR),
    }


def available_training_backends() -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for backend, (available, error) in _backend_availability().items():
        summary[backend] = {
            "available": available,
            "reason": None if available or error is None else str(error),
        }
    return summary


def resolve_training_backend(backend: str) -> str:
    normalized = str(backend).strip().lower()
    if normalized not in SUPPORTED_BACKENDS:
        raise ValueError(f"Unsupported backend '{backend}'. Expected one of: {', '.join(SUPPORTED_BACKENDS)}")

    availability = _backend_availability()
    if normalized == "auto":
        return "lightgbm" if availability["lightgbm"][0] else "sklearn-hgbt"

    available, error = availability[normalized]
    if not available:
        reason = "" if error is None else f": {error}"
        raise RuntimeError(f"Requested backend '{normalized}' is not available{reason}")
    return normalized


def _one_hot(size: int, index: int) -> list[float]:
    values = [0.0] * size
    if 0 <= index < size:
        values[index] = 1.0
    return values


def _normalize_distance(distance: int | None, scale: float = 40.0) -> float:
    if distance is None:
        return 1.0
    return min(float(distance) / scale, 1.0)


@dataclass
class HistoricalRun:
    round_id: str
    round_number: int | None
    initial_states: list[dict[str, Any]]
    payloads: list[dict[str, Any]]
    run_dir: Path
    _feature_matrix_cache: dict[int, list[Any]] = field(default_factory=dict, repr=False)
    _observation_cache: list[dict[tuple[int, int], list[int]]] | None = field(default=None, repr=False)


class StateFeatureContext:
    def __init__(self, initial_state: dict[str, Any], feature_version: int = FEATURE_VERSION) -> None:
        self.initial_state = initial_state
        self.feature_version = int(feature_version)
        self.raw_grid = [[int(value) for value in row] for row in initial_state["grid"]]
        self.class_grid = [
            [terrain_value_to_submission_class(value) for value in row]
            for row in self.raw_grid
        ]
        self.height = len(self.raw_grid)
        self.width = len(self.raw_grid[0]) if self.height else 0
        self._cache: dict[tuple[int, int], list[float]] = {}
        self.settlements = [
            {
                "x": int(item["x"]),
                "y": int(item["y"]),
                "alive": bool(item.get("alive", True)),
                "has_port": bool(item.get("has_port", False)),
            }
            for item in initial_state.get("settlements", [])
        ]
        self.settlement_lookup = {
            (item["x"], item["y"]): (
                3 if not item["alive"] else 2 if item["has_port"] else 1
            )
            for item in self.settlements
        }
        self.ocean_distance_map = self._distance_map_for_raw_values({OCEAN_RAW_VALUE})

    def _distance_map_for_raw_values(self, raw_values: set[int]) -> list[list[int | None]]:
        distances: list[list[int | None]] = [[None for _ in range(self.width)] for _ in range(self.height)]
        queue: deque[tuple[int, int]] = deque()
        for y in range(self.height):
            for x in range(self.width):
                if self.raw_grid[y][x] in raw_values:
                    distances[y][x] = 0
                    queue.append((x, y))

        while queue:
            x, y = queue.popleft()
            next_distance = int(distances[y][x]) + 1
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if 0 <= nx < self.width and 0 <= ny < self.height and distances[ny][nx] is None:
                    distances[ny][nx] = next_distance
                    queue.append((nx, ny))
        return distances

    def _raw_histogram(self, x: int, y: int, radius: int) -> list[float]:
        counts = [0.0] * len(RAW_TERRAIN_VALUES)
        total = 0.0
        for ny in range(max(0, y - radius), min(self.height, y + radius + 1)):
            for nx in range(max(0, x - radius), min(self.width, x + radius + 1)):
                counts[RAW_TERRAIN_TO_INDEX[self.raw_grid[ny][nx]]] += 1.0
                total += 1.0
        return [value / max(total, 1.0) for value in counts]

    def _class_histogram(self, x: int, y: int, radius: int) -> list[float]:
        counts = [0.0] * N_CLASSES
        total = 0.0
        for ny in range(max(0, y - radius), min(self.height, y + radius + 1)):
            for nx in range(max(0, x - radius), min(self.width, x + radius + 1)):
                counts[self.class_grid[ny][nx]] += 1.0
                total += 1.0
        return [value / max(total, 1.0) for value in counts]

    def _axis_distance_features(self, x: int, y: int, raw_values: set[int]) -> list[float]:
        up = None
        for ny in range(y - 1, -1, -1):
            if self.raw_grid[ny][x] in raw_values:
                up = y - ny
                break

        down = None
        for ny in range(y + 1, self.height):
            if self.raw_grid[ny][x] in raw_values:
                down = ny - y
                break

        left = None
        for nx in range(x - 1, -1, -1):
            if self.raw_grid[y][nx] in raw_values:
                left = x - nx
                break

        right = None
        for nx in range(x + 1, self.width):
            if self.raw_grid[y][nx] in raw_values:
                right = nx - x
                break

        return [
            _normalize_distance(up, scale=20.0),
            _normalize_distance(down, scale=20.0),
            _normalize_distance(left, scale=20.0),
            _normalize_distance(right, scale=20.0),
        ]

    def _settlement_features(self, x: int, y: int) -> list[float]:
        if not self.settlements:
            return [1.0, 1.0, 1.0, 1.0, 0.0, 0.0]

        nearest_any: int | None = None
        nearest_alive: int | None = None
        nearest_port: int | None = None
        nearest_dead: int | None = None
        settlements_r2 = 0
        settlements_r4 = 0

        for settlement in self.settlements:
            distance = abs(settlement["x"] - x) + abs(settlement["y"] - y)
            nearest_any = distance if nearest_any is None else min(nearest_any, distance)
            if settlement["alive"]:
                nearest_alive = distance if nearest_alive is None else min(nearest_alive, distance)
            if settlement["has_port"]:
                nearest_port = distance if nearest_port is None else min(nearest_port, distance)
            if not settlement["alive"]:
                nearest_dead = distance if nearest_dead is None else min(nearest_dead, distance)
            if distance <= 2:
                settlements_r2 += 1
            if distance <= 4:
                settlements_r4 += 1

        return [
            _normalize_distance(nearest_any),
            _normalize_distance(nearest_alive),
            _normalize_distance(nearest_port),
            _normalize_distance(nearest_dead),
            min(settlements_r2 / 4.0, 1.0),
            min(settlements_r4 / 8.0, 1.0),
        ]

    def features_for_cell(self, x: int, y: int) -> list[float]:
        key = (x, y)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        raw_center = self.raw_grid[y][x]
        class_center = self.class_grid[y][x]
        raw_hist_3 = self._raw_histogram(x, y, radius=1)
        raw_hist_5 = self._raw_histogram(x, y, radius=2)
        class_hist_3 = self._class_histogram(x, y, radius=1)
        class_hist_5 = self._class_histogram(x, y, radius=2)
        dominant_3 = max(range(N_CLASSES), key=class_hist_3.__getitem__)
        dominant_5 = max(range(N_CLASSES), key=class_hist_5.__getitem__)
        edge_distance = min(x, y, self.width - 1 - x, self.height - 1 - y)
        max_edge = max(min(self.width, self.height) - 1, 1)
        settlement_kind = self.settlement_lookup.get((x, y), 0)

        features = [
            x / max(self.width - 1, 1),
            y / max(self.height - 1, 1),
            edge_distance / max_edge,
        ]
        features.extend(_one_hot(len(RAW_TERRAIN_VALUES), RAW_TERRAIN_TO_INDEX[raw_center]))
        features.extend(_one_hot(N_CLASSES, class_center))
        features.extend(raw_hist_3)
        features.extend(raw_hist_5)
        features.extend(class_hist_3)
        features.extend(class_hist_5)
        features.extend(_one_hot(N_CLASSES, dominant_3))
        features.extend(_one_hot(N_CLASSES, dominant_5))
        features.extend(
            [
                class_hist_3[class_center],
                class_hist_5[class_center],
                raw_hist_3[RAW_TERRAIN_TO_INDEX[raw_center]],
                raw_hist_5[RAW_TERRAIN_TO_INDEX[raw_center]],
            ]
        )
        features.extend(_one_hot(4, settlement_kind))
        features.extend(self._settlement_features(x, y))

        if self.feature_version >= 4:
            raw_hist_7 = self._raw_histogram(x, y, radius=3)
            raw_hist_9 = self._raw_histogram(x, y, radius=4)
            class_hist_7 = self._class_histogram(x, y, radius=3)
            class_hist_9 = self._class_histogram(x, y, radius=4)
            dominant_7 = max(range(N_CLASSES), key=class_hist_7.__getitem__)
            dominant_9 = max(range(N_CLASSES), key=class_hist_9.__getitem__)
            ocean_distance = self.ocean_distance_map[y][x]
            ocean_axis_features = self._axis_distance_features(x, y, {OCEAN_RAW_VALUE})
            ocean_ratio_5 = raw_hist_5[RAW_TERRAIN_TO_INDEX[OCEAN_RAW_VALUE]]
            ocean_ratio_9 = raw_hist_9[RAW_TERRAIN_TO_INDEX[OCEAN_RAW_VALUE]]

            features.extend(raw_hist_7)
            features.extend(raw_hist_9)
            features.extend(class_hist_7)
            features.extend(class_hist_9)
            features.extend(_one_hot(N_CLASSES, dominant_7))
            features.extend(_one_hot(N_CLASSES, dominant_9))
            features.extend(
                [
                    class_hist_7[class_center],
                    class_hist_9[class_center],
                    raw_hist_7[RAW_TERRAIN_TO_INDEX[raw_center]],
                    raw_hist_9[RAW_TERRAIN_TO_INDEX[raw_center]],
                ]
            )
            features.extend(
                [
                    _normalize_distance(ocean_distance, scale=20.0),
                    1.0 if ocean_distance is not None and ocean_distance <= 1 else 0.0,
                    1.0 if ocean_distance is not None and ocean_distance <= 2 else 0.0,
                    ocean_ratio_5,
                    ocean_ratio_9,
                ]
            )
            features.extend(ocean_axis_features)

        self._cache[key] = features
        return features


@dataclass
class GBDTPriorBundle:
    model: Any
    backend: str
    temperature: float
    model_blend: float
    prior_strength: float
    feature_version: int
    params: dict[str, Any]
    training_round_ids: list[str]
    training_examples: int
    cv_prequential_logloss: float | None = None
    posterior_mode: str | None = None
    ensemble_params: dict[str, Any] | None = None
    gate_params: dict[str, Any] | None = None
    bucket_temperatures: list[float] | None = None

    def predict_prior_grid(self, initial_state: dict[str, Any]) -> list[list[list[float]]]:
        _require_ml_dependencies()
        matrix = _build_feature_matrix_for_state(initial_state, self.feature_version)
        return _predict_prior_grid_from_feature_matrix(self, initial_state, matrix)

    def posterior_mixer(self) -> LearnedPosteriorMixer | ConfidenceEnsembleMixer | None:
        if getattr(self, "posterior_mode", None) == "ensemble" or getattr(self, "ensemble_params", None):
            return ConfidenceEnsembleMixer.from_params(getattr(self, "ensemble_params", None))
        gate_params = getattr(self, "gate_params", None)
        bucket_temperatures = getattr(self, "bucket_temperatures", None)
        if not gate_params and not bucket_temperatures:
            return None
        return LearnedPosteriorMixer.from_parts(
            gate_params=gate_params,
            bucket_temperatures=bucket_temperatures,
        )


def apply_temperature(probabilities: Any, temperature: float) -> Any:
    _require_ml_dependencies()
    clipped = np.clip(np.asarray(probabilities, dtype=np.float64), EPS, 1.0)
    power = 1.0 / max(float(temperature), 1e-3)
    scaled = np.power(clipped, power)
    return scaled / scaled.sum(axis=1, keepdims=True)


def save_gbdt_prior_bundle(bundle: GBDTPriorBundle, output_path: str | Path) -> Path:
    _require_ml_dependencies()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_gbdt_prior_bundle(path: str | Path) -> GBDTPriorBundle:
    _require_ml_dependencies()
    return joblib.load(Path(path))


def load_historical_runs(runs_dir: str | Path, cache_db: str | Path) -> list[HistoricalRun]:
    runs_path = Path(runs_dir)
    if not runs_path.exists():
        return []

    history: list[HistoricalRun] = []
    conn = sqlite3.connect(str(cache_db))
    try:
        for run_dir in sorted(runs_path.iterdir()):
            if not run_dir.is_dir():
                continue
            payload_path = run_dir / "simulate_payloads.json"
            summary_path = run_dir / "summary.json"
            if not payload_path.exists() or not summary_path.exists():
                continue
            summary = json.loads(summary_path.read_text())
            round_id = str(summary.get("round_id", "")).strip()
            if not round_id:
                continue
            row = conn.execute(
                "SELECT payload_json FROM cache_entries WHERE cache_key = ?",
                (f"get:/astar-island/rounds/{round_id}",),
            ).fetchone()
            if row is None:
                continue
            detail = json.loads(row[0])
            initial_states = detail.get("initial_states", [])
            if not initial_states:
                continue
            payloads = json.loads(payload_path.read_text())
            if not payloads:
                continue
            history.append(
                HistoricalRun(
                    round_id=round_id,
                    round_number=detail.get("round_number"),
                    initial_states=initial_states,
                    payloads=payloads,
                    run_dir=run_dir,
                )
            )
    finally:
        conn.close()

    return history


def _build_feature_matrix_for_state(initial_state: dict[str, Any], feature_version: int) -> Any:
    context = StateFeatureContext(initial_state, feature_version=feature_version)
    features = []
    for y in range(context.height):
        for x in range(context.width):
            features.append(context.features_for_cell(x, y))
    return np.asarray(features, dtype=np.float32)


def _feature_matrices_for_run(run: HistoricalRun, feature_version: int) -> list[Any]:
    cached = run._feature_matrix_cache.get(int(feature_version))
    if cached is not None:
        return cached
    matrices = [
        _build_feature_matrix_for_state(initial_state, int(feature_version))
        for initial_state in run.initial_states
    ]
    run._feature_matrix_cache[int(feature_version)] = matrices
    return matrices


def _observations_for_run(run: HistoricalRun) -> list[dict[tuple[int, int], list[int]]]:
    if run._observation_cache is not None:
        return run._observation_cache
    per_seed: list[dict[tuple[int, int], list[int]]] = [
        defaultdict(list)
        for _ in range(len(run.initial_states))
    ]
    for payload in run.payloads:
        query = payload["query"]
        seed_index = int(query["seed_index"])
        viewport = payload["viewport"]
        grid = payload["grid"]
        base_x = int(viewport["x"])
        base_y = int(viewport["y"])
        for dy, row in enumerate(grid):
            for dx, raw_value in enumerate(row):
                x = base_x + dx
                y = base_y + dy
                per_seed[seed_index][(x, y)].append(
                    terrain_value_to_submission_class(int(raw_value))
                )
    run._observation_cache = per_seed
    return per_seed


def sort_historical_runs(history: list[HistoricalRun]) -> list[HistoricalRun]:
    return sorted(
        history,
        key=lambda run: (
            -1 if run.round_number is None else int(run.round_number),
            str(run.round_id),
        ),
    )


def recent_historical_round_ids(
    history: list[HistoricalRun],
    count: int,
    exclude_round_ids: set[str] | None = None,
) -> list[str]:
    excluded = set() if exclude_round_ids is None else {str(round_id) for round_id in exclude_round_ids}
    ordered = [
        run.round_id
        for run in sort_historical_runs(history)
        if str(run.round_id) not in excluded
    ]
    if count <= 0:
        return ordered
    return ordered[-count:]


def evaluate_bundle_on_round_ids(
    bundle: GBDTPriorBundle,
    history: list[HistoricalRun],
    round_ids: list[str],
) -> dict[str, Any]:
    runs_by_id = {run.round_id: run for run in history}
    selected_runs = [runs_by_id[round_id] for round_id in round_ids if round_id in runs_by_id]
    per_round = []
    for run in selected_runs:
        loss = evaluate_bundle_prequential_logloss(run, bundle)
        per_round.append(
            {
                "round_id": run.round_id,
                "round_number": run.round_number,
                "prequential_logloss": float(loss),
            }
        )
    average = None if not per_round else sum(item["prequential_logloss"] for item in per_round) / len(per_round)
    return {
        "round_ids": [item["round_id"] for item in per_round],
        "average_prequential_logloss": average,
        "per_round": per_round,
    }


def compare_bundle_paths_on_recent_history(
    *,
    stable_model_path: str | Path,
    candidate_model_path: str | Path,
    runs_dir: str | Path,
    cache_db: str | Path,
    recent_rounds: int = 6,
    exclude_round_ids: set[str] | None = None,
) -> dict[str, Any]:
    history = load_historical_runs(runs_dir, cache_db)
    stable_bundle = load_gbdt_prior_bundle(stable_model_path)
    candidate_bundle = load_gbdt_prior_bundle(candidate_model_path)
    round_ids = recent_historical_round_ids(history, recent_rounds, exclude_round_ids=exclude_round_ids)
    stable_eval = evaluate_bundle_on_round_ids(stable_bundle, history, round_ids)
    candidate_eval = evaluate_bundle_on_round_ids(candidate_bundle, history, round_ids)
    stable_avg = stable_eval["average_prequential_logloss"]
    candidate_avg = candidate_eval["average_prequential_logloss"]
    delta = None
    if isinstance(stable_avg, float) and isinstance(candidate_avg, float):
        delta = candidate_avg - stable_avg
    return {
        "benchmark_round_ids": round_ids,
        "recent_rounds": recent_rounds,
        "stable_model_path": str(stable_model_path),
        "candidate_model_path": str(candidate_model_path),
        "stable": stable_eval,
        "candidate": candidate_eval,
        "candidate_minus_stable": delta,
    }


def build_training_matrix(
    history: list[HistoricalRun],
    feature_version: int = FEATURE_VERSION,
) -> tuple[Any, Any, list[str]]:
    _require_ml_dependencies()
    features: list[list[float]] = []
    labels: list[int] = []
    round_ids: list[str] = []

    for run in history:
        feature_matrices = _feature_matrices_for_run(run, int(feature_version))
        for payload in run.payloads:
            query = payload["query"]
            seed_index = int(query["seed_index"])
            initial_state = run.initial_states[seed_index]
            width = len(initial_state["grid"][0]) if initial_state.get("grid") else 0
            viewport = payload["viewport"]
            grid = payload["grid"]
            base_x = int(viewport["x"])
            base_y = int(viewport["y"])
            for dy, row in enumerate(grid):
                for dx, raw_value in enumerate(row):
                    x = base_x + dx
                    y = base_y + dy
                    features.append(feature_matrices[seed_index][y * width + x])
                    labels.append(terrain_value_to_submission_class(int(raw_value)))
                    round_ids.append(run.round_id)

    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64), round_ids


def _fit_classifier(features: Any, labels: Any, params: dict[str, Any], backend: str) -> Any:
    _require_ml_dependencies()
    resolved_backend = resolve_training_backend(backend)
    if resolved_backend == "lightgbm":
        model = LGBMClassifier(
            objective="multiclass",
            num_class=N_CLASSES,
            random_state=0,
            n_estimators=int(params["n_estimators"]),
            learning_rate=float(params["learning_rate"]),
            num_leaves=int(params["num_leaves"]),
            min_child_samples=int(params["min_child_samples"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            reg_alpha=float(params["reg_alpha"]),
            reg_lambda=float(params["reg_lambda"]),
            verbose=-1,
            n_jobs=-1,
        )
        model.fit(features, labels)
        return model

    if resolved_backend == "xgboost":
        model = XGBClassifier(
            objective="multi:softprob",
            num_class=N_CLASSES,
            random_state=0,
            n_estimators=int(params["n_estimators"]),
            learning_rate=float(params["learning_rate"]),
            max_leaves=int(params["max_leaves"]),
            max_depth=int(params["max_depth"]),
            min_child_weight=float(params["min_child_weight"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            reg_alpha=float(params["reg_alpha"]),
            reg_lambda=float(params["reg_lambda"]),
            tree_method="hist",
            grow_policy="lossguide",
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(features, labels)
        return model

    if resolved_backend == "catboost":
        model = CatBoostClassifier(
            loss_function="MultiClass",
            random_seed=0,
            iterations=int(params["iterations"]),
            learning_rate=float(params["learning_rate"]),
            depth=int(params["depth"]),
            l2_leaf_reg=float(params["l2_leaf_reg"]),
            random_strength=float(params["random_strength"]),
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(features, labels, verbose=False)
        return model

    if resolved_backend == "sklearn-hgbt":
        model = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=float(params["learning_rate"]),
            max_iter=int(params["n_estimators"]),
            max_leaf_nodes=int(params["num_leaves"]),
            min_samples_leaf=int(params["min_child_samples"]),
            l2_regularization=float(params["reg_lambda"]),
            max_bins=255,
            random_state=0,
        )
        model.fit(features, labels)
        return model

    raise RuntimeError(f"Unhandled backend '{resolved_backend}'")


def _predict_prior_grid_from_feature_matrix(
    bundle: GBDTPriorBundle,
    initial_state: dict[str, Any],
    feature_matrix: Any,
) -> list[list[list[float]]]:
    base_priors = build_initial_priors(initial_state)
    probabilities = bundle.model.predict_proba(feature_matrix)
    probabilities = apply_temperature(probabilities, bundle.temperature)

    height = len(base_priors)
    width = len(base_priors[0]) if height else 0
    priors: list[list[list[float]]] = []
    index = 0
    for y in range(height):
        row: list[list[float]] = []
        for x in range(width):
            model_prior = probabilities[index]
            row.append(
                normalize(
                    [
                        (1.0 - bundle.model_blend) * base_priors[y][x][class_idx]
                        + bundle.model_blend * float(model_prior[class_idx])
                        for class_idx in range(N_CLASSES)
                    ]
                )
            )
            index += 1
        priors.append(row)
    return priors


def evaluate_bundle_prequential_logloss(run: HistoricalRun, bundle: GBDTPriorBundle) -> float:
    total_loss = 0.0
    total_events = 0
    observations = _observations_for_run(run)
    feature_matrices = _feature_matrices_for_run(run, bundle.feature_version)
    posterior_mixer = bundle.posterior_mixer()

    for seed_index, initial_state in enumerate(run.initial_states):
        priors = _predict_prior_grid_from_feature_matrix(
            bundle,
            initial_state,
            feature_matrices[seed_index],
        )
        for (x, y), sequence in observations[seed_index].items():
            counts = [0] * N_CLASSES
            for observed_class in sequence:
                posterior = build_cell_posterior(
                    counts=counts,
                    model_prior=priors[y][x],
                    prior_strength=bundle.prior_strength,
                    posterior_mixer=posterior_mixer,
                )
                probability = posterior[observed_class]
                total_loss -= log(max(probability, EPS))
                counts[observed_class] += 1
                total_events += 1

    return total_loss / max(total_events, 1)


def _cross_validate_bundle_params(
    *,
    features: Any,
    labels: Any,
    round_ids: list[str],
    unique_round_ids: list[str],
    runs_by_id: dict[str, HistoricalRun],
    backend: str,
    params: dict[str, Any],
    model_blend: float,
    prior_strength: float,
    temperature: float,
    feature_version: int,
) -> float:
    fold_losses: list[float] = []
    for holdout_round_id in unique_round_ids:
        train_mask = np.asarray([round_id != holdout_round_id for round_id in round_ids], dtype=bool)
        model = _fit_classifier(features[train_mask], labels[train_mask], params, backend=backend)
        bundle = GBDTPriorBundle(
            model=model,
            backend=backend,
            temperature=temperature,
            model_blend=model_blend,
            prior_strength=prior_strength,
            feature_version=feature_version,
            params=params,
            training_round_ids=[round_id for round_id in unique_round_ids if round_id != holdout_round_id],
            training_examples=int(train_mask.sum()),
        )
        fold_losses.append(evaluate_bundle_prequential_logloss(runs_by_id[holdout_round_id], bundle))
    return sum(fold_losses) / len(fold_losses)


def _default_gate_search_space(trial: Any) -> tuple[dict[str, Any], list[float]]:
    gate_params = {
        "bias": trial.suggest_float("gate_bias", -3.0, 3.0),
        "empirical_confidence_weight": trial.suggest_float("gate_empirical_confidence_weight", -3.0, 4.0),
        "model_confidence_weight": trial.suggest_float("gate_model_confidence_weight", -3.0, 3.0),
        "disagreement_weight": trial.suggest_float("gate_disagreement_weight", -4.0, 4.0),
        "agreement_weight": trial.suggest_float("gate_agreement_weight", -3.0, 3.0),
        "bucket_biases": [
            0.0,
            trial.suggest_float("gate_bucket_bias_1", -3.0, 3.0),
            trial.suggest_float("gate_bucket_bias_2", -3.0, 3.0),
            trial.suggest_float("gate_bucket_bias_3", -3.0, 3.0),
        ],
    }
    bucket_temperatures = [
        trial.suggest_float("bucket_temperature_0", 0.75, 1.85),
        trial.suggest_float("bucket_temperature_1", 0.75, 1.85),
        trial.suggest_float("bucket_temperature_2", 0.75, 1.85),
        trial.suggest_float("bucket_temperature_3", 0.75, 1.85),
    ]
    return gate_params, bucket_temperatures


def _tune_posterior_mixer(
    base_bundle: GBDTPriorBundle,
    runs_by_id: dict[str, HistoricalRun],
    optuna_trials: int,
    mixer_optuna_trials: int | None = None,
) -> tuple[GBDTPriorBundle, dict[str, Any]]:
    effective_trials = max(24, optuna_trials * 2) if mixer_optuna_trials is None else int(mixer_optuna_trials)
    if optuna is None or effective_trials <= 0:
        return base_bundle, {}

    ordered_runs = [runs_by_id[round_id] for round_id in sorted(runs_by_id)]
    baseline_losses = [
        evaluate_bundle_prequential_logloss(run, base_bundle)
        for run in ordered_runs
    ]
    baseline_value = sum(baseline_losses) / len(baseline_losses)

    study = optuna.create_study(direction="minimize")

    def objective(trial: Any) -> float:
        gate_params, bucket_temperatures = _default_gate_search_space(trial)
        candidate_bundle = replace(
            base_bundle,
            gate_params=gate_params,
            bucket_temperatures=bucket_temperatures,
        )
        fold_losses = [
            evaluate_bundle_prequential_logloss(run, candidate_bundle)
            for run in ordered_runs
        ]
        return sum(fold_losses) / len(fold_losses)

    mixer_trials = effective_trials
    study.optimize(objective, n_trials=mixer_trials)
    best_gate_params = {
        "bias": float(study.best_params["gate_bias"]),
        "empirical_confidence_weight": float(study.best_params["gate_empirical_confidence_weight"]),
        "model_confidence_weight": float(study.best_params["gate_model_confidence_weight"]),
        "disagreement_weight": float(study.best_params["gate_disagreement_weight"]),
        "agreement_weight": float(study.best_params["gate_agreement_weight"]),
        "bucket_biases": [
            0.0,
            float(study.best_params["gate_bucket_bias_1"]),
            float(study.best_params["gate_bucket_bias_2"]),
            float(study.best_params["gate_bucket_bias_3"]),
        ],
    }
    best_bucket_temperatures = [
        float(study.best_params["bucket_temperature_0"]),
        float(study.best_params["bucket_temperature_1"]),
        float(study.best_params["bucket_temperature_2"]),
        float(study.best_params["bucket_temperature_3"]),
    ]
    tuned_bundle = replace(
        base_bundle,
        cv_prequential_logloss=float(study.best_value),
        gate_params=best_gate_params,
        bucket_temperatures=best_bucket_temperatures,
    )
    return tuned_bundle, {
        "baseline_cv_prequential_logloss": baseline_value,
        "posterior_mixer_trials": mixer_trials,
        "posterior_mixer_best_value": float(study.best_value),
        "posterior_mixer_best_params": {
            "gate_params": best_gate_params,
            "bucket_temperatures": best_bucket_temperatures,
        },
    }


def _default_ensemble_search_space(trial: Any) -> dict[str, Any]:
    weight_bucket_1 = trial.suggest_float("ensemble_weight_bucket_1", 0.18, 0.62)
    weight_bucket_2 = trial.suggest_float("ensemble_weight_bucket_2", weight_bucket_1 + 0.05, 0.82)
    weight_bucket_3 = trial.suggest_float("ensemble_weight_bucket_3", weight_bucket_2 + 0.03, 0.94)
    return {
        "bucket_empirical_weights": [
            0.0,
            weight_bucket_1,
            weight_bucket_2,
            weight_bucket_3,
        ],
        "empirical_confidence_bonus": trial.suggest_float("ensemble_empirical_confidence_bonus", 0.0, 0.24),
        "disagreement_penalty": trial.suggest_float("ensemble_disagreement_penalty", 0.0, 0.32),
        "agreement_bonus": trial.suggest_float("ensemble_agreement_bonus", 0.0, 0.12),
        "min_model_weight": trial.suggest_float("ensemble_min_model_weight", 0.05, 0.30),
    }


def _tune_confidence_ensemble(
    base_bundle: GBDTPriorBundle,
    runs_by_id: dict[str, HistoricalRun],
    optuna_trials: int,
    mixer_optuna_trials: int | None = None,
) -> tuple[GBDTPriorBundle, dict[str, Any]]:
    effective_trials = max(24, optuna_trials * 3) if mixer_optuna_trials is None else int(mixer_optuna_trials)
    if optuna is None or effective_trials <= 0:
        return base_bundle, {}

    ordered_runs = [runs_by_id[round_id] for round_id in sorted(runs_by_id)]
    baseline_losses = [
        evaluate_bundle_prequential_logloss(run, base_bundle)
        for run in ordered_runs
    ]
    baseline_value = sum(baseline_losses) / len(baseline_losses)

    study = optuna.create_study(direction="minimize")

    def objective(trial: Any) -> float:
        ensemble_params = _default_ensemble_search_space(trial)
        candidate_bundle = replace(
            base_bundle,
            posterior_mode="ensemble",
            ensemble_params=ensemble_params,
            gate_params=None,
            bucket_temperatures=None,
        )
        fold_losses = [
            evaluate_bundle_prequential_logloss(run, candidate_bundle)
            for run in ordered_runs
        ]
        return sum(fold_losses) / len(fold_losses)

    ensemble_trials = effective_trials
    study.optimize(objective, n_trials=ensemble_trials)
    best_ensemble_params = {
        "bucket_empirical_weights": [
            0.0,
            float(study.best_params["ensemble_weight_bucket_1"]),
            float(study.best_params["ensemble_weight_bucket_2"]),
            float(study.best_params["ensemble_weight_bucket_3"]),
        ],
        "empirical_confidence_bonus": float(study.best_params["ensemble_empirical_confidence_bonus"]),
        "disagreement_penalty": float(study.best_params["ensemble_disagreement_penalty"]),
        "agreement_bonus": float(study.best_params["ensemble_agreement_bonus"]),
        "min_model_weight": float(study.best_params["ensemble_min_model_weight"]),
    }
    tuned_bundle = replace(
        base_bundle,
        cv_prequential_logloss=float(study.best_value),
        posterior_mode="ensemble",
        ensemble_params=best_ensemble_params,
        gate_params=None,
        bucket_temperatures=None,
    )
    return tuned_bundle, {
        "baseline_cv_prequential_logloss": baseline_value,
        "ensemble_trials": ensemble_trials,
        "ensemble_best_value": float(study.best_value),
        "ensemble_best_params": best_ensemble_params,
    }


def _default_search_space(trial: Any, backend: str) -> tuple[dict[str, Any], float, float, float]:
    resolved_backend = resolve_training_backend(backend)
    if resolved_backend == "lightgbm":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 140, 420),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.18, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 24, 128),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 160),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.5, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 2.5, log=True),
        }
    elif resolved_backend == "xgboost":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 140, 420),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.18, log=True),
            "max_leaves": trial.suggest_int("max_leaves", 24, 128),
            "max_depth": trial.suggest_int("max_depth", 0, 12),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 8.0),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.5, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 2.5, log=True),
        }
    elif resolved_backend == "catboost":
        params = {
            "iterations": trial.suggest_int("iterations", 160, 420),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.18, log=True),
            "depth": trial.suggest_int("depth", 5, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 15.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 2.5),
        }
    else:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 160, 480),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.16, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 24, 160),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 120),
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_alpha": 0.0,
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 2.5, log=True),
        }
    model_blend = trial.suggest_float("model_blend", 0.35, 0.95)
    prior_strength = trial.suggest_float("prior_strength", 0.05, 2.5, log=True)
    temperature = trial.suggest_float("temperature", 0.65, 1.75)
    return params, model_blend, prior_strength, temperature


def _default_backend_params(backend: str) -> dict[str, Any]:
    resolved_backend = resolve_training_backend(backend)
    if resolved_backend == "lightgbm":
        return {
            "n_estimators": 260,
            "learning_rate": 0.055,
            "num_leaves": 63,
            "min_child_samples": 50,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.02,
            "reg_lambda": 0.08,
            "model_blend": 0.72,
            "prior_strength": 0.42,
            "temperature": 1.0,
        }
    if resolved_backend == "xgboost":
        return {
            "n_estimators": 260,
            "learning_rate": 0.055,
            "max_leaves": 63,
            "max_depth": 0,
            "min_child_weight": 1.5,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.02,
            "reg_lambda": 0.08,
            "model_blend": 0.72,
            "prior_strength": 0.42,
            "temperature": 1.0,
        }
    if resolved_backend == "catboost":
        return {
            "iterations": 260,
            "learning_rate": 0.055,
            "depth": 8,
            "l2_leaf_reg": 5.0,
            "random_strength": 1.0,
            "model_blend": 0.72,
            "prior_strength": 0.42,
            "temperature": 1.0,
        }
    return {
        "n_estimators": 260,
        "learning_rate": 0.055,
        "num_leaves": 63,
        "min_child_samples": 50,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 0.08,
        "model_blend": 0.72,
        "prior_strength": 0.42,
        "temperature": 1.0,
    }


def train_gbdt_prior_bundle(
    runs_dir: str | Path,
    cache_db: str | Path,
    output_path: str | Path = DEFAULT_MODEL_PATH,
    report_path: str | Path = DEFAULT_REPORT_PATH,
    optuna_trials: int = 18,
    backend: str = "auto",
    mixer_optuna_trials: int | None = None,
    feature_version: int = FEATURE_VERSION,
) -> tuple[GBDTPriorBundle, dict[str, Any]]:
    _require_ml_dependencies()
    if optuna_trials > 0 and optuna is None:
        raise RuntimeError("Optuna is not available. Install optuna to run hyperparameter tuning.")
    resolved_backend = resolve_training_backend(backend)
    history = load_historical_runs(runs_dir, cache_db)
    if len(history) < 2:
        raise RuntimeError("Need at least 2 historical runs with simulate_payloads.json to train the GBDT prior")

    features, labels, round_ids = build_training_matrix(history, feature_version=feature_version)
    unique_round_ids = sorted({run.round_id for run in history})
    runs_by_id = {run.round_id: run for run in history}

    best_params = _default_backend_params(resolved_backend)
    best_value: float | None = None

    if optuna_trials > 0:
        study = optuna.create_study(direction="minimize")

        def objective(trial: Any) -> float:
            params, model_blend, prior_strength, temperature = _default_search_space(trial, resolved_backend)
            return _cross_validate_bundle_params(
                features=features,
                labels=labels,
                round_ids=round_ids,
                unique_round_ids=unique_round_ids,
                runs_by_id=runs_by_id,
                backend=resolved_backend,
                params=params,
                model_blend=model_blend,
                prior_strength=prior_strength,
                temperature=temperature,
                feature_version=feature_version,
            )

        study.optimize(objective, n_trials=optuna_trials)
        best_params.update(study.best_params)
        best_value = float(study.best_value)
    else:
        model_params = {
            key: value
            for key, value in best_params.items()
            if key not in {"model_blend", "prior_strength", "temperature"}
        }
        best_value = _cross_validate_bundle_params(
            features=features,
            labels=labels,
            round_ids=round_ids,
            unique_round_ids=unique_round_ids,
            runs_by_id=runs_by_id,
            backend=resolved_backend,
            params=model_params,
            model_blend=float(best_params["model_blend"]),
            prior_strength=float(best_params["prior_strength"]),
            temperature=float(best_params["temperature"]),
            feature_version=feature_version,
        )

    model_params = {
        key: value
        for key, value in best_params.items()
        if key not in {"model_blend", "prior_strength", "temperature"}
    }
    final_model = _fit_classifier(features, labels, model_params, backend=resolved_backend)
    base_bundle = GBDTPriorBundle(
        model=final_model,
        backend=resolved_backend,
        temperature=float(best_params["temperature"]),
        model_blend=float(best_params["model_blend"]),
        prior_strength=float(best_params["prior_strength"]),
        feature_version=feature_version,
        params=model_params,
        training_round_ids=unique_round_ids,
        training_examples=int(len(labels)),
        cv_prequential_logloss=best_value,
    )
    bundle, ensemble_report = _tune_confidence_ensemble(
        base_bundle=base_bundle,
        runs_by_id=runs_by_id,
        optuna_trials=optuna_trials,
        mixer_optuna_trials=mixer_optuna_trials,
    )
    save_gbdt_prior_bundle(bundle, output_path)

    report = {
        "output_path": str(Path(output_path)),
        "report_path": str(Path(report_path)),
        "training_round_ids": unique_round_ids,
        "training_examples": int(len(labels)),
        "feature_version": feature_version,
        "best_params": best_params,
        "backend": bundle.backend,
        "mixer_optuna_trials": mixer_optuna_trials,
        "cv_prequential_logloss": bundle.cv_prequential_logloss,
        "base_model_cv_prequential_logloss": best_value,
        "posterior_mode": bundle.posterior_mode,
        "ensemble_params": bundle.ensemble_params,
        "gate_params": bundle.gate_params,
        "bucket_temperatures": bundle.bucket_temperatures,
        "posterior_mixer": ensemble_report,
    }
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    return bundle, report


def run_backend_bakeoff(
    runs_dir: str | Path,
    cache_db: str | Path,
    output_dir: str | Path,
    optuna_trials: int = 18,
    mixer_optuna_trials: int | None = None,
    backends: list[str] | None = None,
    baseline_logloss: float | None = None,
    min_improvement: float = 0.01,
) -> dict[str, Any]:
    _require_ml_dependencies()
    requested_backends = backends or ["sklearn-hgbt", "lightgbm", "xgboost", "catboost"]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    for backend_name in requested_backends:
        try:
            resolved_backend = resolve_training_backend(backend_name)
        except Exception as exc:
            results.append(
                {
                    "backend": backend_name,
                    "status": "unavailable",
                    "reason": str(exc),
                }
            )
            continue

        backend_model_path = output_path / f"{resolved_backend}.joblib"
        backend_report_path = output_path / f"{resolved_backend}_report.json"
        bundle, report = train_gbdt_prior_bundle(
            runs_dir=runs_dir,
            cache_db=cache_db,
            output_path=backend_model_path,
            report_path=backend_report_path,
            optuna_trials=optuna_trials,
            backend=resolved_backend,
            mixer_optuna_trials=mixer_optuna_trials,
        )
        results.append(
            {
                "backend": resolved_backend,
                "status": "ok",
                "model_path": str(backend_model_path),
                "report_path": str(backend_report_path),
                "cv_prequential_logloss": bundle.cv_prequential_logloss,
                "training_examples": bundle.training_examples,
                "posterior_mode": bundle.posterior_mode,
                "best_params": report.get("best_params"),
            }
        )

    successful = [item for item in results if item.get("status") == "ok" and item.get("cv_prequential_logloss") is not None]
    winner = None
    gate_passed = False
    if successful:
        winner = min(successful, key=lambda item: float(item["cv_prequential_logloss"]))
        if baseline_logloss is not None:
            gate_passed = float(winner["cv_prequential_logloss"]) <= float(baseline_logloss) - float(min_improvement)

    summary = {
        "baseline_logloss": baseline_logloss,
        "optuna_trials": optuna_trials,
        "mixer_optuna_trials": mixer_optuna_trials,
        "min_improvement": min_improvement,
        "gate_threshold": None if baseline_logloss is None else float(baseline_logloss) - float(min_improvement),
        "results": results,
        "winner": winner,
        "gate_passed": gate_passed,
        "availability": available_training_backends(),
    }
    (output_path / "bakeoff_report.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return summary
