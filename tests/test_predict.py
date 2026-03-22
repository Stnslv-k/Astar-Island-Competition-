import unittest

from astar_island.predict import (
    ConfidenceEnsembleMixer,
    LearnedPosteriorMixer,
    ObservationAccumulator,
    apply_temperature_1d,
    build_cell_posterior,
    smooth_prediction,
)


class PredictTests(unittest.TestCase):
    def test_prediction_probabilities_sum_to_one(self) -> None:
        accumulator = ObservationAccumulator(width=2, height=2)
        accumulator.add_patch(
            viewport={"x": 0, "y": 0, "w": 2, "h": 2},
            grid=[
                [1, 2],
                [3, 4],
            ],
        )
        prediction = accumulator.to_prediction_tensor(prior_strength=0.10, floor=0.01)

        self.assertEqual(len(prediction), 2)
        self.assertEqual(len(prediction[0]), 2)
        for row in prediction:
            for cell in row:
                self.assertAlmostEqual(sum(cell), 1.0, places=6)
                self.assertGreaterEqual(min(cell), 0.01)

    def test_repeated_observations_create_empirical_distribution(self) -> None:
        accumulator = ObservationAccumulator(width=1, height=1)
        accumulator.add_patch(
            viewport={"x": 0, "y": 0, "w": 1, "h": 1},
            grid=[[1]],
        )
        accumulator.add_patch(
            viewport={"x": 0, "y": 0, "w": 1, "h": 1},
            grid=[[2]],
        )
        prediction = accumulator.to_prediction_tensor(prior_strength=0.10, floor=0.01)
        self.assertGreater(prediction[0][0][1], 0.40)
        self.assertGreater(prediction[0][0][2], 0.40)

    def test_smoothing_keeps_observed_cell_dominant(self) -> None:
        prediction = [
            [[0.90, 0.10], [0.10, 0.90]],
            [[0.10, 0.90], [0.10, 0.90]],
        ]
        smoothed = smooth_prediction(
            prediction=prediction,
            observation_counts=[[3, 0], [0, 0]],
            passes=1,
            strength=0.30,
        )
        self.assertGreater(smoothed[0][0][0], smoothed[0][0][1])

    def test_edge_aware_smoothing_preserves_prior_boundary(self) -> None:
        prediction = [
            [[0.92, 0.08], [0.08, 0.92]],
        ]
        priors = [
            [[0.97, 0.03], [0.03, 0.97]],
        ]
        smoothed = smooth_prediction(
            prediction=prediction,
            observation_counts=[[0, 0]],
            cell_priors=priors,
            passes=1,
            strength=0.35,
            mode="edge-aware",
        )
        plain = smooth_prediction(
            prediction=prediction,
            observation_counts=[[0, 0]],
            cell_priors=priors,
            passes=1,
            strength=0.35,
            mode="plain",
        )
        self.assertGreater(smoothed[0][0][0], plain[0][0][0])
        self.assertGreater(smoothed[0][1][1], plain[0][1][1])

    def test_edge_aware_smoothing_respects_peaked_prior_anchor(self) -> None:
        prediction = [
            [[0.60, 0.40], [0.10, 0.90]],
        ]
        priors = [
            [[0.94, 0.06], [0.05, 0.95]],
        ]
        smoothed = smooth_prediction(
            prediction=prediction,
            observation_counts=[[0, 0]],
            cell_priors=priors,
            passes=1,
            strength=0.30,
            mode="edge-aware",
        )
        self.assertGreater(smoothed[0][0][0], 0.55)

    def test_uncertainty_map_prefers_mixed_cells(self) -> None:
        accumulator = ObservationAccumulator(width=2, height=1)
        accumulator.add_patch(
            viewport={"x": 0, "y": 0, "w": 2, "h": 1},
            grid=[[1, 1]],
        )
        accumulator.add_patch(
            viewport={"x": 1, "y": 0, "w": 1, "h": 1},
            grid=[[2]],
        )
        uncertainty = accumulator.uncertainty_map(prior_strength=0.10)
        self.assertGreater(uncertainty[0][1], uncertainty[0][0])

    def test_learned_gate_shifts_toward_empirical_with_more_coverage(self) -> None:
        mixer = LearnedPosteriorMixer(
            bias=0.0,
            empirical_confidence_weight=0.0,
            model_confidence_weight=0.0,
            disagreement_weight=0.0,
            agreement_weight=0.0,
            bucket_biases=(0.0, -4.0, 0.0, 4.0),
            bucket_temperatures=(1.0, 1.0, 1.0, 1.0),
        )
        model_prior = [0.10, 0.90]
        low_coverage = build_cell_posterior(
            counts=[1, 0],
            model_prior=model_prior,
            posterior_mixer=mixer,
        )
        high_coverage = build_cell_posterior(
            counts=[4, 0],
            model_prior=model_prior,
            posterior_mixer=mixer,
        )
        self.assertLess(low_coverage[0], 0.20)
        self.assertGreater(high_coverage[0], 0.90)

    def test_bucketed_calibration_softens_probabilities(self) -> None:
        softened = apply_temperature_1d([0.90, 0.10], temperature=1.6)
        sharpened = apply_temperature_1d([0.90, 0.10], temperature=0.8)
        self.assertAlmostEqual(sum(softened), 1.0, places=6)
        self.assertLess(softened[0], 0.90)
        self.assertGreater(sharpened[0], 0.90)

    def test_confidence_ensemble_uses_model_prior_without_observations(self) -> None:
        mixer = ConfidenceEnsembleMixer()
        posterior = build_cell_posterior(
            counts=[0, 0],
            model_prior=[0.15, 0.85],
            posterior_mixer=mixer,
        )
        self.assertEqual(posterior, [0.15, 0.85])

    def test_confidence_ensemble_moves_toward_empirical_as_coverage_grows(self) -> None:
        mixer = ConfidenceEnsembleMixer(
            bucket_empirical_weights=(0.0, 0.30, 0.60, 0.85),
            empirical_confidence_bonus=0.0,
            disagreement_penalty=0.0,
            agreement_bonus=0.0,
            min_model_weight=0.10,
        )
        model_prior = [0.10, 0.90]
        low = build_cell_posterior(
            counts=[1, 0],
            model_prior=model_prior,
            posterior_mixer=mixer,
        )
        high = build_cell_posterior(
            counts=[4, 0],
            model_prior=model_prior,
            posterior_mixer=mixer,
        )
        self.assertGreater(low[1], low[0])
        self.assertGreater(high[0], low[0])


if __name__ == "__main__":
    unittest.main()
