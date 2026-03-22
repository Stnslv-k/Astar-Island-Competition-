from __future__ import annotations


N_CLASSES = 6


def normalize(probabilities: list[float]) -> list[float]:
    total = sum(probabilities)
    if total <= 0.0:
        return [1.0 / len(probabilities) for _ in probabilities]
    return [value / total for value in probabilities]


def terrain_value_to_submission_class(value: int) -> int:
    if value in {0, 10, 11}:
        return 0
    if value in {1, 2, 3, 4, 5}:
        return value
    raise ValueError(f"Unsupported terrain value: {value}")


def base_prior_for_class(class_idx: int) -> list[float]:
    priors = {
        0: [0.93, 0.01, 0.01, 0.01, 0.03, 0.01],
        1: [0.10, 0.58, 0.08, 0.18, 0.04, 0.02],
        2: [0.08, 0.12, 0.68, 0.08, 0.02, 0.02],
        3: [0.12, 0.08, 0.04, 0.70, 0.04, 0.02],
        4: [0.11, 0.02, 0.01, 0.03, 0.78, 0.05],
        5: [0.08, 0.01, 0.01, 0.02, 0.05, 0.83],
    }
    return priors[class_idx][:]


def build_initial_priors(initial_state: dict[str, object]) -> list[list[list[float]]]:
    grid = initial_state["grid"]
    height = len(grid)
    width = len(grid[0]) if height else 0

    priors: list[list[list[float]]] = []
    for row in grid:
        prior_row: list[list[float]] = []
        for terrain_value in row:
            class_idx = terrain_value_to_submission_class(int(terrain_value))
            prior_row.append(base_prior_for_class(class_idx))
        priors.append(prior_row)

    for settlement in initial_state.get("settlements", []):
        x = int(settlement["x"])
        y = int(settlement["y"])
        if not (0 <= x < width and 0 <= y < height):
            continue

        if not settlement.get("alive", True):
            priors[y][x] = base_prior_for_class(3)
        elif settlement.get("has_port", False):
            priors[y][x] = [0.05, 0.14, 0.70, 0.07, 0.02, 0.02]
        else:
            priors[y][x] = [0.08, 0.63, 0.08, 0.15, 0.04, 0.02]

    for y in range(height):
        for x in range(width):
            priors[y][x] = normalize(priors[y][x])

    return priors

