from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from astar_island.terrain import N_CLASSES, build_initial_priors, normalize, terrain_value_to_submission_class


def _settlement_lookup(initial_state: dict[str, object]) -> dict[tuple[int, int], int]:
    lookup: dict[tuple[int, int], int] = {}
    for settlement in initial_state.get("settlements", []):
        x = int(settlement["x"])
        y = int(settlement["y"])
        if not settlement.get("alive", True):
            lookup[(x, y)] = 3
        elif settlement.get("has_port", False):
            lookup[(x, y)] = 2
        else:
            lookup[(x, y)] = 1
    return lookup


def _feature_keys(initial_state: dict[str, object], x: int, y: int) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    grid = initial_state["grid"]
    height = len(grid)
    width = len(grid[0]) if height else 0
    center = terrain_value_to_submission_class(int(grid[y][x]))
    settlement_kind = _settlement_lookup(initial_state).get((x, y), 0)

    counts = [0] * N_CLASSES
    for ny in range(max(0, y - 1), min(height, y + 2)):
        for nx in range(max(0, x - 1), min(width, x + 2)):
            terrain = terrain_value_to_submission_class(int(grid[ny][nx]))
            counts[terrain] += 1

    dominant = max(range(N_CLASSES), key=lambda class_idx: (counts[class_idx], -abs(class_idx - center)))
    same_bucket = min(4, int((counts[center] * 5) / 9))
    edge_distance = min(x, y, width - 1 - x, height - 1 - y)
    edge_bucket = 0 if edge_distance <= 1 else 1 if edge_distance <= 3 else 2

    full_key = (center, dominant, same_bucket, edge_bucket, settlement_kind)
    coarse_key = (center, dominant, settlement_kind)
    center_key = (center, settlement_kind)
    return full_key, coarse_key, center_key


def _empty_counts() -> list[float]:
    return [0.0 for _ in range(N_CLASSES)]


def _add_counts(target: list[float], source: list[float], weight: float) -> None:
    for class_idx in range(N_CLASSES):
        target[class_idx] += source[class_idx] * weight


@dataclass
class LearnedPriorModel:
    full_stats: dict[tuple[int, ...], list[int]]
    coarse_stats: dict[tuple[int, ...], list[int]]
    center_stats: dict[tuple[int, ...], list[int]]
    total_examples: int

    def _distribution_from_stats(
        self,
        stats: dict[tuple[int, ...], list[int]],
        key: tuple[int, ...],
    ) -> tuple[list[float] | None, int]:
        counts = stats.get(key)
        if counts is None:
            return None, 0
        total = sum(counts)
        if total <= 0:
            return None, 0
        return [value / total for value in counts], total

    def learned_distribution(
        self,
        initial_state: dict[str, object],
        x: int,
        y: int,
        fallback_prior: list[float],
    ) -> tuple[list[float], dict[str, int]]:
        full_key, coarse_key, center_key = _feature_keys(initial_state, x, y)
        exact_dist, exact_n = self._distribution_from_stats(self.full_stats, full_key)
        coarse_dist, coarse_n = self._distribution_from_stats(self.coarse_stats, coarse_key)
        center_dist, center_n = self._distribution_from_stats(self.center_stats, center_key)

        center_terrain = terrain_value_to_submission_class(int(initial_state["grid"][y][x]))
        center_only = [0.01] * N_CLASSES
        center_only[center_terrain] = 0.95
        center_only = normalize(center_only)

        blended = _empty_counts()
        diagnostics = {"exact_samples": exact_n, "coarse_samples": coarse_n, "center_samples": center_n}

        if exact_dist is not None:
            _add_counts(blended, exact_dist, 0.45 * min(exact_n / 18.0, 1.0))
        if coarse_dist is not None:
            _add_counts(blended, coarse_dist, 0.30 * min(coarse_n / 40.0, 1.0))
        if center_dist is not None:
            _add_counts(blended, center_dist, 0.20 * min(center_n / 90.0, 1.0))

        _add_counts(blended, center_only, 0.25)
        _add_counts(blended, fallback_prior, 0.25)
        return normalize(blended), diagnostics


def build_learned_prior_model(runs_dir: str | Path, cache_db: str | Path) -> LearnedPriorModel | None:
    runs_path = Path(runs_dir)
    if not runs_path.exists():
        return None

    conn = sqlite3.connect(str(cache_db))
    full_stats: dict[tuple[int, ...], list[int]] = defaultdict(lambda: [0] * N_CLASSES)
    coarse_stats: dict[tuple[int, ...], list[int]] = defaultdict(lambda: [0] * N_CLASSES)
    center_stats: dict[tuple[int, ...], list[int]] = defaultdict(lambda: [0] * N_CLASSES)
    total_examples = 0

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
            for payload in payloads:
                query = payload["query"]
                seed_index = int(query["seed_index"])
                if not (0 <= seed_index < len(initial_states)):
                    continue
                initial_state = initial_states[seed_index]
                viewport = payload["viewport"]
                grid = payload["grid"]
                base_x = int(viewport["x"])
                base_y = int(viewport["y"])
                for dy, row_values in enumerate(grid):
                    for dx, raw_value in enumerate(row_values):
                        x = base_x + dx
                        y = base_y + dy
                        observed_class = terrain_value_to_submission_class(int(raw_value))
                        full_key, coarse_key, center_key = _feature_keys(initial_state, x, y)
                        full_stats[full_key][observed_class] += 1
                        coarse_stats[coarse_key][observed_class] += 1
                        center_stats[center_key][observed_class] += 1
                        total_examples += 1
    finally:
        conn.close()

    if total_examples == 0:
        return None
    return LearnedPriorModel(
        full_stats=dict(full_stats),
        coarse_stats=dict(coarse_stats),
        center_stats=dict(center_stats),
        total_examples=total_examples,
    )


def build_augmented_priors(
    initial_state: dict[str, object],
    model: LearnedPriorModel | None,
    learned_blend: float = 0.40,
) -> tuple[list[list[list[float]]], dict[str, int | float]]:
    base_priors = build_initial_priors(initial_state)
    if model is None or learned_blend <= 0.0:
        return base_priors, {"enabled": 0, "training_examples": 0, "matched_cells": 0}

    grid = initial_state["grid"]
    height = len(grid)
    width = len(grid[0]) if height else 0
    priors: list[list[list[float]]] = []
    matched_cells = 0
    for y in range(height):
        row: list[list[float]] = []
        for x in range(width):
            learned_dist, diagnostics = model.learned_distribution(
                initial_state,
                x,
                y,
                fallback_prior=base_priors[y][x],
            )
            if diagnostics["center_samples"] > 0:
                matched_cells += 1
            row.append(
                normalize([
                    (1.0 - learned_blend) * base_priors[y][x][class_idx] + learned_blend * learned_dist[class_idx]
                    for class_idx in range(N_CLASSES)
                ])
            )
        priors.append(row)

    return priors, {
        "enabled": 1,
        "training_examples": model.total_examples,
        "matched_cells": matched_cells,
        "total_cells": width * height,
    }
