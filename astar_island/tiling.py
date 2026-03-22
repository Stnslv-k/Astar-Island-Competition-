from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from astar_island.predict import ObservationAccumulator, build_cell_posterior, entropy


MAX_QUERY_BUDGET = 50


@dataclass(frozen=True)
class QuerySpec:
    round_id: str
    seed_index: int
    viewport_x: int
    viewport_y: int
    viewport_w: int
    viewport_h: int


def compute_cover_starts(size: int, window: int) -> list[int]:
    if window <= 0:
        raise ValueError("window must be > 0")
    if size <= window:
        return [0]

    starts = [0]
    while starts[-1] + window < size:
        next_start = starts[-1] + max(window - 2, 1)
        last_start = size - window
        if next_start >= last_start:
            starts.append(last_start)
            break
        starts.append(next_start)
    return starts


def build_full_cover_queries(
    round_id: str,
    seeds_count: int,
    width: int,
    height: int,
    window: int = 15,
) -> list[QuerySpec]:
    if not 5 <= window <= 15:
        raise ValueError("window must be in the range [5, 15]")
    xs = compute_cover_starts(width, window)
    ys = compute_cover_starts(height, window)
    queries: list[QuerySpec] = []
    for seed_index in range(seeds_count):
        for viewport_y in ys:
            for viewport_x in xs:
                queries.append(
                    QuerySpec(
                        round_id=round_id,
                        seed_index=seed_index,
                        viewport_x=viewport_x,
                        viewport_y=viewport_y,
                        viewport_w=window,
                        viewport_h=window,
                    )
                )
    return queries


def _terrain_diversity_score(grid: list[list[int]], x: int, y: int, window: int) -> float:
    values: set[int] = set()
    for row in grid[y : y + window]:
        values.update(row[x : x + window])
    return float(len(values))


def _settlement_window_score(settlements: list[dict[str, Any]], x: int, y: int, window: int) -> float:
    score = 0.0
    for settlement in settlements:
        sx = int(settlement["x"])
        sy = int(settlement["y"])
        if x <= sx < x + window and y <= sy < y + window:
            score += 4.0
            if settlement.get("alive", False):
                score += 1.0
            if settlement.get("has_port", False):
                score += 3.0
    return score


def choose_refinement_queries(
    round_id: str,
    detail: dict[str, Any],
    budget: int = 5,
    window: int = 15,
    stride: int = 5,
) -> list[QuerySpec]:
    if not 5 <= window <= 15:
        raise ValueError("window must be in the range [5, 15]")
    width = int(detail["map_width"])
    height = int(detail["map_height"])
    seeds_count = int(detail["seeds_count"])

    max_x = max(width - window, 0)
    max_y = max(height - window, 0)
    xs = sorted(set(list(range(0, max_x + 1, stride)) + [max_x]))
    ys = sorted(set(list(range(0, max_y + 1, stride)) + [max_y]))

    scored: list[tuple[float, QuerySpec]] = []
    for seed_index in range(seeds_count):
        state = detail["initial_states"][seed_index]
        grid = state["grid"]
        settlements = state.get("settlements", [])
        for viewport_y in ys:
            for viewport_x in xs:
                score = _settlement_window_score(settlements, viewport_x, viewport_y, window)
                score += _terrain_diversity_score(grid, viewport_x, viewport_y, window)
                scored.append(
                    (
                        score,
                        QuerySpec(
                            round_id=round_id,
                            seed_index=seed_index,
                            viewport_x=viewport_x,
                            viewport_y=viewport_y,
                            viewport_w=window,
                            viewport_h=window,
                        ),
                    )
                )

    scored.sort(key=lambda item: item[0], reverse=True)

    chosen: list[QuerySpec] = []
    per_seed_counts = {seed_index: 0 for seed_index in range(seeds_count)}
    for _, query in scored:
        if len(chosen) >= budget:
            break
        if per_seed_counts[query.seed_index] >= 2:
            continue
        chosen.append(query)
        per_seed_counts[query.seed_index] += 1
    return chosen


def _window_cells(width: int, height: int, x: int, y: int, window: int) -> list[tuple[int, int]]:
    return [
        (cell_x, cell_y)
        for cell_y in range(y, min(y + window, height))
        for cell_x in range(x, min(x + window, width))
    ]


def _window_overlap_ratio(
    ax: int,
    ay: int,
    bx: int,
    by: int,
    window: int,
) -> float:
    overlap_w = max(0, min(ax + window, bx + window) - max(ax, bx))
    overlap_h = max(0, min(ay + window, by + window) - max(ay, by))
    area = window * window
    if area <= 0:
        return 0.0
    return (overlap_w * overlap_h) / area


def _expected_information_gain_map(
    accumulator: ObservationAccumulator,
    distributions: list[list[list[float]]],
    model_priors: list[list[list[float]]],
    prior_strength: float,
) -> list[list[float]]:
    scores: list[list[float]] = []
    for y in range(accumulator.height):
        row: list[float] = []
        for x in range(accumulator.width):
            current = distributions[y][x]
            counts = accumulator.counts[y][x]
            current_entropy = entropy(current)
            expected_next_entropy = 0.0
            for observed_class, probability in enumerate(current):
                if probability <= 0.0:
                    continue
                next_counts = counts[:]
                next_counts[observed_class] += 1
                next_distribution = build_cell_posterior(
                    counts=next_counts,
                    model_prior=model_priors[y][x],
                    prior_strength=prior_strength,
                    posterior_mixer=accumulator.posterior_mixer,
                )
                expected_next_entropy += probability * entropy(next_distribution)
            row.append(max(current_entropy - expected_next_entropy, 0.0))
        scores.append(row)
    return scores


def choose_adaptive_refinement_queries(
    round_id: str,
    detail: dict[str, Any],
    accumulators: list[ObservationAccumulator],
    budget: int = 5,
    window: int = 15,
    stride: int = 5,
    prior_strength: float = 0.35,
    excluded_windows: set[tuple[int, int, int]] | None = None,
) -> list[QuerySpec]:
    if budget <= 0:
        return []
    if not 5 <= window <= 15:
        raise ValueError("window must be in the range [5, 15]")

    width = int(detail["map_width"])
    height = int(detail["map_height"])
    max_x = max(width - window, 0)
    max_y = max(height - window, 0)
    xs = sorted(set(list(range(0, max_x + 1, stride)) + [max_x]))
    ys = sorted(set(list(range(0, max_y + 1, stride)) + [max_y]))

    excluded_windows = excluded_windows or set()
    scored: list[tuple[float, QuerySpec]] = []
    for seed_index, accumulator in enumerate(accumulators):
        distributions = accumulator.distribution_map(prior_strength=prior_strength)
        information_gain = _expected_information_gain_map(
            accumulator=accumulator,
            distributions=distributions,
            model_priors=accumulator.model_prior_map(),
            prior_strength=prior_strength,
        )
        settlements = detail["initial_states"][seed_index].get("settlements", [])
        settlement_cells = {(int(item["x"]), int(item["y"])) for item in settlements}

        for viewport_y in ys:
            for viewport_x in xs:
                score = 0.0
                frontier_cells = 0
                settlement_hits = 0
                scarcity_bonus = 0.0
                for cell_x, cell_y in _window_cells(width, height, viewport_x, viewport_y, window):
                    score += information_gain[cell_y][cell_x]
                    scarcity_bonus += 1.0 / max(accumulator.coverage_count(cell_x, cell_y), 1)
                    center_class = distributions[cell_y][cell_x]
                    center_argmax = max(range(len(center_class)), key=center_class.__getitem__)
                    for dy, dx in ((0, 1), (1, 0)):
                        nx = cell_x + dx
                        ny = cell_y + dy
                        if nx >= width or ny >= height:
                            continue
                        neighbor_class = distributions[ny][nx]
                        neighbor_argmax = max(range(len(neighbor_class)), key=neighbor_class.__getitem__)
                        if neighbor_argmax != center_argmax:
                            frontier_cells += 1
                    if (cell_x, cell_y) in settlement_cells:
                        settlement_hits += 1

                score += 0.08 * frontier_cells
                score += 0.06 * scarcity_bonus
                score += 0.55 * settlement_hits
                scored.append(
                    (
                        score,
                        QuerySpec(
                            round_id=round_id,
                            seed_index=seed_index,
                            viewport_x=viewport_x,
                            viewport_y=viewport_y,
                            viewport_w=window,
                            viewport_h=window,
                        ),
                    )
                )

    scored.sort(key=lambda item: item[0], reverse=True)

    chosen: list[QuerySpec] = []
    per_seed_counts = {seed_index: 0 for seed_index in range(len(accumulators))}
    seen_windows: set[tuple[int, int, int]] = set()
    prior_windows_by_seed: dict[int, list[tuple[int, int]]] = {}
    for seed_index, viewport_x, viewport_y in excluded_windows:
        prior_windows_by_seed.setdefault(seed_index, []).append((viewport_x, viewport_y))
    while scored and len(chosen) < budget:
        best_index = 0
        best_value = float("-inf")
        for index, (raw_score, query) in enumerate(scored[: min(len(scored), 50)]):
            window_key = (query.seed_index, query.viewport_x, query.viewport_y)
            if window_key in excluded_windows:
                continue
            adjusted = raw_score / (1.0 + 0.35 * per_seed_counts[query.seed_index])
            max_overlap = 0.0
            for existing_x, existing_y in prior_windows_by_seed.get(query.seed_index, []):
                max_overlap = max(
                    max_overlap,
                    _window_overlap_ratio(
                        query.viewport_x,
                        query.viewport_y,
                        existing_x,
                        existing_y,
                        window,
                    ),
                )
            adjusted *= 1.0 - 0.55 * max_overlap
            if adjusted > best_value:
                best_value = adjusted
                best_index = index

        raw_score, query = scored.pop(best_index)
        window_key = (query.seed_index, query.viewport_x, query.viewport_y)
        if window_key in excluded_windows or window_key in seen_windows or raw_score <= 0.0:
            continue
        chosen.append(query)
        seen_windows.add(window_key)
        per_seed_counts[query.seed_index] += 1
        prior_windows_by_seed.setdefault(query.seed_index, []).append((query.viewport_x, query.viewport_y))

    return chosen


def build_default_queries(
    round_id: str,
    detail: dict[str, Any],
    window: int = 15,
    refinement_budget: int = 5,
) -> list[QuerySpec]:
    cover = build_full_cover_queries(
        round_id=round_id,
        seeds_count=int(detail["seeds_count"]),
        width=int(detail["map_width"]),
        height=int(detail["map_height"]),
        window=window,
    )
    refinement = choose_refinement_queries(
        round_id=round_id,
        detail=detail,
        budget=refinement_budget,
        window=window,
    )
    queries = cover + refinement
    if len(queries) > MAX_QUERY_BUDGET:
        raise ValueError(
            f"default plan uses {len(queries)} queries, which exceeds the budget of {MAX_QUERY_BUDGET}"
        )
    return queries
