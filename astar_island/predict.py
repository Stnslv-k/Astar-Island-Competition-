from __future__ import annotations

from dataclasses import dataclass
from math import exp, log

from astar_island.terrain import N_CLASSES, build_initial_priors, normalize, terrain_value_to_submission_class


def _uniform_probabilities(n_classes: int = N_CLASSES) -> list[float]:
    return [1.0 / n_classes for _ in range(n_classes)]


def _apply_floor(probabilities: list[float], floor: float) -> list[float]:
    n = len(probabilities)
    if n == 0:
        return []
    if floor * n >= 1.0:
        return _uniform_probabilities(n)

    clipped = [max(value, 0.0) for value in probabilities]
    total = sum(clipped)
    if total == 0.0:
        clipped = _uniform_probabilities(n)
    else:
        clipped = [value / total for value in clipped]

    remaining_mass = 1.0 - floor * n
    return [floor + remaining_mass * value for value in clipped]


def entropy(probabilities: list[float]) -> float:
    total = 0.0
    for value in probabilities:
        if value > 0.0:
            total -= value * log(value)
    return total


def coverage_bucket(observation_count: int) -> int:
    if observation_count <= 0:
        return 0
    if observation_count == 1:
        return 1
    if observation_count == 2:
        return 2
    return 3


def apply_temperature_1d(probabilities: list[float], temperature: float) -> list[float]:
    clipped = [max(value, 1e-12) for value in probabilities]
    power = 1.0 / max(float(temperature), 1e-3)
    scaled = [value ** power for value in clipped]
    return normalize(scaled)


@dataclass(frozen=True)
class LearnedPosteriorMixer:
    bias: float = 0.0
    empirical_confidence_weight: float = 0.0
    model_confidence_weight: float = 0.0
    disagreement_weight: float = 0.0
    agreement_weight: float = 0.0
    bucket_biases: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    bucket_temperatures: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)

    @classmethod
    def from_parts(
        cls,
        gate_params: dict[str, float] | None = None,
        bucket_temperatures: list[float] | tuple[float, ...] | None = None,
    ) -> "LearnedPosteriorMixer":
        gate_params = gate_params or {}
        raw_biases = list(gate_params.get("bucket_biases", []))
        while len(raw_biases) < 4:
            raw_biases.append(0.0)
        raw_temps = list(bucket_temperatures or gate_params.get("bucket_temperatures", []))
        while len(raw_temps) < 4:
            raw_temps.append(1.0)
        return cls(
            bias=float(gate_params.get("bias", 0.0)),
            empirical_confidence_weight=float(gate_params.get("empirical_confidence_weight", 0.0)),
            model_confidence_weight=float(gate_params.get("model_confidence_weight", 0.0)),
            disagreement_weight=float(gate_params.get("disagreement_weight", 0.0)),
            agreement_weight=float(gate_params.get("agreement_weight", 0.0)),
            bucket_biases=tuple(float(value) for value in raw_biases[:4]),
            bucket_temperatures=tuple(float(value) for value in raw_temps[:4]),
        )

    def gate_weight(
        self,
        observation_count: int,
        empirical_probabilities: list[float],
        model_probabilities: list[float],
    ) -> float:
        if observation_count <= 0:
            return 0.0

        max_entropy = log(len(empirical_probabilities)) if len(empirical_probabilities) > 1 else 1.0
        empirical_confidence = 1.0 - entropy(empirical_probabilities) / max(max_entropy, 1e-12)
        model_confidence = 1.0 - entropy(model_probabilities) / max(max_entropy, 1e-12)
        disagreement = 0.5 * sum(
            abs(empirical_probabilities[class_idx] - model_probabilities[class_idx])
            for class_idx in range(len(empirical_probabilities))
        )
        agreement = 1.0 if empirical_probabilities.index(max(empirical_probabilities)) == model_probabilities.index(max(model_probabilities)) else 0.0
        bucket = coverage_bucket(observation_count)
        logit = (
            self.bias
            + self.bucket_biases[bucket]
            + self.empirical_confidence_weight * empirical_confidence
            + self.model_confidence_weight * model_confidence
            + self.disagreement_weight * disagreement
            + self.agreement_weight * agreement
        )
        logit = max(min(logit, 18.0), -18.0)
        return 1.0 / (1.0 + exp(-logit))

    def apply_bucket_calibration(
        self,
        probabilities: list[float],
        observation_count: int,
    ) -> list[float]:
        return apply_temperature_1d(
            probabilities,
            self.bucket_temperatures[coverage_bucket(observation_count)],
        )

    def combine(
        self,
        counts: list[int],
        model_prior: list[float],
        prior_strength: float = 0.35,
    ) -> list[float]:
        total_obs = sum(counts)
        if total_obs == 0:
            return self.apply_bucket_calibration(model_prior[:], observation_count=0)

        empirical = normalize([float(value) for value in counts])
        gate_weight = self.gate_weight(
            observation_count=total_obs,
            empirical_probabilities=empirical,
            model_probabilities=model_prior,
        )
        blended = normalize([
            gate_weight * empirical[class_idx] + (1.0 - gate_weight) * model_prior[class_idx]
            for class_idx in range(len(counts))
        ])
        return self.apply_bucket_calibration(blended, observation_count=total_obs)


@dataclass(frozen=True)
class ConfidenceEnsembleMixer:
    bucket_empirical_weights: tuple[float, float, float, float] = (0.0, 0.40, 0.62, 0.78)
    empirical_confidence_bonus: float = 0.08
    disagreement_penalty: float = 0.10
    agreement_bonus: float = 0.04
    min_model_weight: float = 0.10

    @classmethod
    def from_params(cls, params: dict[str, float] | None = None) -> "ConfidenceEnsembleMixer":
        params = params or {}
        raw_weights = list(params.get("bucket_empirical_weights", []))
        if not raw_weights:
            raw_weights = [0.0, 0.40, 0.62, 0.78]
        while len(raw_weights) < 4:
            raw_weights.append(raw_weights[-1] if raw_weights else 0.0)
        return cls(
            bucket_empirical_weights=tuple(float(value) for value in raw_weights[:4]),
            empirical_confidence_bonus=float(params.get("empirical_confidence_bonus", 0.08)),
            disagreement_penalty=float(params.get("disagreement_penalty", 0.10)),
            agreement_bonus=float(params.get("agreement_bonus", 0.04)),
            min_model_weight=float(params.get("min_model_weight", 0.10)),
        )

    def combine(
        self,
        counts: list[int],
        model_prior: list[float],
        prior_strength: float = 0.35,
    ) -> list[float]:
        total_obs = sum(counts)
        if total_obs == 0:
            return model_prior[:]

        empirical = normalize([float(value) for value in counts])
        max_entropy = log(len(empirical)) if len(empirical) > 1 else 1.0
        empirical_confidence = 1.0 - entropy(empirical) / max(max_entropy, 1e-12)
        disagreement = 0.5 * sum(
            abs(empirical[class_idx] - model_prior[class_idx])
            for class_idx in range(len(counts))
        )
        agreement = 1.0 if empirical.index(max(empirical)) == model_prior.index(max(model_prior)) else 0.0

        bucket = coverage_bucket(total_obs)
        empirical_weight = self.bucket_empirical_weights[bucket]
        empirical_weight += self.empirical_confidence_bonus * empirical_confidence
        empirical_weight -= self.disagreement_penalty * disagreement
        empirical_weight += self.agreement_bonus * agreement

        min_empirical_weight = 0.0 if total_obs <= 0 else 0.05
        max_empirical_weight = 1.0 - max(self.min_model_weight, 0.0)
        empirical_weight = min(max(empirical_weight, min_empirical_weight), max_empirical_weight)
        return normalize([
            empirical_weight * empirical[class_idx] + (1.0 - empirical_weight) * model_prior[class_idx]
            for class_idx in range(len(counts))
        ])


def build_cell_posterior(
    counts: list[int],
    model_prior: list[float],
    prior_strength: float = 0.35,
    posterior_mixer: LearnedPosteriorMixer | ConfidenceEnsembleMixer | None = None,
) -> list[float]:
    total_obs = sum(counts)
    if posterior_mixer is None:
        if total_obs == 0:
            return model_prior[:]
        return normalize([
            (counts[class_idx] + prior_strength * model_prior[class_idx]) / (total_obs + prior_strength)
            for class_idx in range(len(counts))
        ])

    return posterior_mixer.combine(counts=counts, model_prior=model_prior, prior_strength=prior_strength)


def smooth_prediction(
    prediction: list[list[list[float]]],
    observation_counts: list[list[int]],
    cell_priors: list[list[list[float]]] | None = None,
    passes: int = 1,
    strength: float = 0.18,
    mode: str = "edge-aware",
) -> list[list[list[float]]]:
    if not prediction or not prediction[0] or passes <= 0 or strength <= 0.0:
        return prediction

    height = len(prediction)
    width = len(prediction[0])
    n_classes = len(prediction[0][0])
    current = [[cell[:] for cell in row] for row in prediction]

    for _ in range(passes):
        next_prediction: list[list[list[float]]] = []
        for y in range(height):
            row: list[list[float]] = []
            for x in range(width):
                center = current[y][x]
                center_prior = cell_priors[y][x] if cell_priors is not None else center
                weighted_neighbors: list[tuple[list[float], float]] = []
                for ny in range(max(0, y - 1), min(height, y + 2)):
                    for nx in range(max(0, x - 1), min(width, x + 2)):
                        if nx == x and ny == y:
                            continue
                        neighbor = current[ny][nx]
                        if mode == "plain" or cell_priors is None:
                            weighted_neighbors.append((neighbor, 1.0))
                            continue

                        neighbor_prior = cell_priors[ny][nx]
                        prior_similarity = sum(
                            min(center_prior[class_idx], neighbor_prior[class_idx])
                            for class_idx in range(n_classes)
                        )
                        posterior_similarity = sum(
                            min(center[class_idx], neighbor[class_idx])
                            for class_idx in range(n_classes)
                        )
                        edge_weight = max(0.04, (0.72 * prior_similarity + 0.28 * posterior_similarity) ** 2)
                        weighted_neighbors.append((neighbor, edge_weight))

                if not weighted_neighbors:
                    row.append(center[:])
                    continue

                total_weight = sum(weight for _, weight in weighted_neighbors)
                neighbor_mean = [
                    sum(neighbor[class_idx] * weight for neighbor, weight in weighted_neighbors) / total_weight
                    for class_idx in range(n_classes)
                ]
                observed = observation_counts[y][x]
                if mode == "plain" or cell_priors is None:
                    anchor_strength = 0.0
                else:
                    uniform = 1.0 / n_classes
                    prior_peak = max(center_prior)
                    prior_anchor = max(0.0, (prior_peak - uniform) / max(1.0 - uniform, 1e-9))
                    observation_anchor = min(observed / 3.0, 1.0)
                    anchor_strength = max(observation_anchor, 0.90 * prior_anchor)
                local_strength = strength * max(0.05, 1.0 - 0.95 * anchor_strength) / (1.0 + 0.35 * observed)
                row.append(
                    normalize([
                        (1.0 - local_strength) * center[class_idx]
                        + local_strength * neighbor_mean[class_idx]
                        for class_idx in range(n_classes)
                    ])
                )
            next_prediction.append(row)
        current = next_prediction

    return current


@dataclass
class ObservationAccumulator:
    width: int
    height: int
    cell_priors: list[list[list[float]]] | None = None
    posterior_mixer: LearnedPosteriorMixer | ConfidenceEnsembleMixer | None = None
    n_classes: int = N_CLASSES

    def __post_init__(self) -> None:
        self.counts = [
            [[0 for _ in range(self.n_classes)] for _ in range(self.width)]
            for _ in range(self.height)
        ]

    @classmethod
    def from_initial_state(cls, initial_state: dict[str, object]) -> "ObservationAccumulator":
        grid = initial_state["grid"]
        height = len(grid)
        width = len(grid[0]) if height else 0
        return cls(width=width, height=height, cell_priors=build_initial_priors(initial_state))

    def add_patch(self, viewport: dict[str, int], grid: list[list[int]]) -> None:
        base_x = int(viewport["x"])
        base_y = int(viewport["y"])
        for dy, row in enumerate(grid):
            for dx, class_idx in enumerate(row):
                x = base_x + dx
                y = base_y + dy
                if 0 <= x < self.width and 0 <= y < self.height:
                    mapped_class = terrain_value_to_submission_class(int(class_idx))
                    self.counts[y][x][mapped_class] += 1

    def coverage_count(self, x: int, y: int) -> int:
        return sum(self.counts[y][x])

    def full_coverage_ratio(self) -> float:
        covered = 0
        total = self.width * self.height
        for y in range(self.height):
            for x in range(self.width):
                if self.coverage_count(x, y) > 0:
                    covered += 1
        return covered / total if total else 0.0

    def global_prior(self) -> list[float]:
        totals = [0 for _ in range(self.n_classes)]
        for y in range(self.height):
            for x in range(self.width):
                for class_idx, value in enumerate(self.counts[y][x]):
                    totals[class_idx] += value
        observed_total = sum(totals)

        prior_mean = [0.0 for _ in range(self.n_classes)]
        if self.cell_priors is not None:
            cell_count = self.width * self.height
            for y in range(self.height):
                for x in range(self.width):
                    for class_idx in range(self.n_classes):
                        prior_mean[class_idx] += self.cell_priors[y][x][class_idx]
            if cell_count > 0:
                prior_mean = [value / cell_count for value in prior_mean]
        else:
            prior_mean = _uniform_probabilities(self.n_classes)

        if observed_total == 0:
            return prior_mean

        observed = [value / observed_total for value in totals]
        return normalize([
            0.80 * observed[class_idx] + 0.20 * prior_mean[class_idx]
            for class_idx in range(self.n_classes)
        ])

    def _blended_prior_for_cell(self, x: int, y: int, global_prior: list[float]) -> list[float]:
        cell_prior = self.cell_priors[y][x][:] if self.cell_priors is not None else global_prior[:]
        return normalize([
            0.88 * cell_prior[class_idx] + 0.12 * global_prior[class_idx]
            for class_idx in range(self.n_classes)
        ])

    def model_prior_for_cell(
        self,
        x: int,
        y: int,
        global_prior: list[float] | None = None,
    ) -> list[float]:
        prior = global_prior if global_prior is not None else self.global_prior()
        return self._blended_prior_for_cell(x, y, prior)

    def cell_distribution(
        self,
        x: int,
        y: int,
        prior_strength: float = 0.35,
        global_prior: list[float] | None = None,
    ) -> list[float]:
        prior = global_prior if global_prior is not None else self.global_prior()
        blended_prior = self._blended_prior_for_cell(x, y, prior)
        return build_cell_posterior(
            counts=self.counts[y][x],
            model_prior=blended_prior,
            prior_strength=prior_strength,
            posterior_mixer=self.posterior_mixer,
        )

    def distribution_map(self, prior_strength: float = 0.35) -> list[list[list[float]]]:
        prior = self.global_prior()
        return [
            [
                self.cell_distribution(x, y, prior_strength=prior_strength, global_prior=prior)
                for x in range(self.width)
            ]
            for y in range(self.height)
        ]

    def model_prior_map(self) -> list[list[list[float]]]:
        prior = self.global_prior()
        return [
            [
                self.model_prior_for_cell(x, y, global_prior=prior)
                for x in range(self.width)
            ]
            for y in range(self.height)
        ]

    def uncertainty_map(
        self,
        prior_strength: float = 0.35,
        distributions: list[list[list[float]]] | None = None,
    ) -> list[list[float]]:
        distributions = distributions if distributions is not None else self.distribution_map(prior_strength=prior_strength)
        max_entropy = log(self.n_classes) if self.n_classes > 1 else 1.0
        scores: list[list[float]] = []
        for y in range(self.height):
            row: list[float] = []
            for x in range(self.width):
                probabilities = distributions[y][x]
                normalized_entropy = entropy(probabilities) / max_entropy if max_entropy > 0.0 else 0.0
                row.append(normalized_entropy * (1.0 + 0.20 / max(self.coverage_count(x, y), 1)))
            scores.append(row)
        return scores

    def observation_counts(self) -> list[list[int]]:
        return [
            [sum(self.counts[y][x]) for x in range(self.width)]
            for y in range(self.height)
        ]

    def to_prediction_tensor(
        self,
        prior_strength: float = 0.35,
        floor: float = 0.0025,
        smoothing_passes: int = 1,
        smoothing_strength: float = 0.18,
        smoothing_mode: str = "edge-aware",
    ) -> list[list[list[float]]]:
        prediction = self.distribution_map(prior_strength=prior_strength)

        prediction = smooth_prediction(
            prediction,
            observation_counts=self.observation_counts(),
            cell_priors=self.cell_priors,
            passes=smoothing_passes,
            strength=smoothing_strength,
            mode=smoothing_mode,
        )
        return [
            [_apply_floor(cell, floor) for cell in row]
            for row in prediction
        ]


def summarize_prediction(prediction: list[list[list[float]]]) -> dict[str, float]:
    height = len(prediction)
    width = len(prediction[0]) if height else 0
    min_prob = 1.0
    max_prob = 0.0
    total_sum_error = 0.0
    cells = 0
    for row in prediction:
        for probs in row:
            min_prob = min(min_prob, min(probs))
            max_prob = max(max_prob, max(probs))
            total_sum_error += abs(sum(probs) - 1.0)
            cells += 1
    return {
        "width": float(width),
        "height": float(height),
        "min_prob": min_prob,
        "max_prob": max_prob,
        "avg_sum_error": total_sum_error / max(cells, 1),
    }
