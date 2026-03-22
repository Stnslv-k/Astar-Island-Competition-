import unittest

from astar_island.gbdt import (
    HistoricalRun,
    StateFeatureContext,
    _feature_matrices_for_run,
    _observations_for_run,
    apply_temperature,
    available_training_backends,
    recent_historical_round_ids,
    resolve_training_backend,
)


class GBDTTests(unittest.TestCase):
    def test_feature_vector_has_stable_shape(self) -> None:
        state = {
            "grid": [
                [10, 11, 4],
                [1, 2, 5],
                [3, 4, 11],
            ],
            "settlements": [{"x": 1, "y": 1, "alive": True, "has_port": False}],
        }
        context = StateFeatureContext(state)
        left = context.features_for_cell(0, 0)
        center = context.features_for_cell(1, 1)
        self.assertEqual(len(left), len(center))
        self.assertNotEqual(left, center)

    def test_feature_version_3_preserves_legacy_shape(self) -> None:
        state = {
            "grid": [
                [10, 11, 4],
                [1, 2, 5],
                [3, 4, 11],
            ],
            "settlements": [{"x": 1, "y": 1, "alive": True, "has_port": False}],
        }
        legacy = StateFeatureContext(state, feature_version=3).features_for_cell(0, 0)
        current = StateFeatureContext(state, feature_version=4).features_for_cell(0, 0)
        self.assertEqual(len(legacy), 71)
        self.assertEqual(len(current), 124)

    def test_temperature_scaling_preserves_normalization(self) -> None:
        probabilities = apply_temperature([[0.70, 0.20, 0.10]], temperature=1.5)
        self.assertAlmostEqual(sum(probabilities[0]), 1.0, places=6)
        self.assertLess(probabilities[0][0], 0.70)

    def test_backend_availability_includes_known_backends(self) -> None:
        availability = available_training_backends()
        self.assertIn("sklearn-hgbt", availability)
        self.assertIn("lightgbm", availability)
        self.assertIn("xgboost", availability)
        self.assertIn("catboost", availability)
        self.assertTrue(availability["sklearn-hgbt"]["available"])

    def test_auto_backend_resolves_without_optional_packages(self) -> None:
        availability = available_training_backends()
        resolved = resolve_training_backend("auto")
        self.assertIn(resolved, availability)
        self.assertTrue(availability[resolved]["available"])

    def test_ocean_proximity_changes_features_for_same_center_class(self) -> None:
        state = {
            "grid": [
                [10, 10, 10, 10, 10],
                [10, 11, 11, 11, 10],
                [10, 11, 11, 11, 10],
                [10, 11, 11, 11, 10],
                [10, 10, 10, 10, 10],
            ],
            "settlements": [],
        }
        context = StateFeatureContext(state)
        near = context.features_for_cell(1, 1)
        center = context.features_for_cell(2, 2)
        self.assertEqual(len(near), len(center))
        self.assertNotEqual(near, center)

    def test_historical_run_feature_matrices_are_cached_by_version(self) -> None:
        run = HistoricalRun(
            round_id="r1",
            round_number=1,
            initial_states=[
                {
                    "grid": [
                        [10, 11],
                        [4, 1],
                    ],
                    "settlements": [],
                }
            ],
            payloads=[],
            run_dir=".",
        )
        first = _feature_matrices_for_run(run, 3)
        second = _feature_matrices_for_run(run, 3)
        current = _feature_matrices_for_run(run, 4)
        self.assertIs(first, second)
        self.assertEqual(first[0].shape, (4, 71))
        self.assertEqual(current[0].shape, (4, 124))

    def test_historical_run_observations_are_cached(self) -> None:
        run = HistoricalRun(
            round_id="r1",
            round_number=1,
            initial_states=[
                {
                    "grid": [
                        [10, 11],
                        [4, 1],
                    ],
                    "settlements": [],
                }
            ],
            payloads=[
                {
                    "query": {"seed_index": 0},
                    "viewport": {"x": 0, "y": 0},
                    "grid": [[10, 11], [4, 1]],
                }
            ],
            run_dir=".",
        )
        first = _observations_for_run(run)
        second = _observations_for_run(run)
        self.assertIs(first, second)
        self.assertIn((0, 0), first[0])
        self.assertEqual(len(first[0][(0, 0)]), 1)

    def test_recent_historical_round_ids_uses_round_number_order(self) -> None:
        history = [
            HistoricalRun(round_id="r10", round_number=10, initial_states=[], payloads=[], run_dir="."),
            HistoricalRun(round_id="r8", round_number=8, initial_states=[], payloads=[], run_dir="."),
            HistoricalRun(round_id="r12", round_number=12, initial_states=[], payloads=[], run_dir="."),
            HistoricalRun(round_id="r11", round_number=11, initial_states=[], payloads=[], run_dir="."),
        ]
        self.assertEqual(
            recent_historical_round_ids(history, 2),
            ["r11", "r12"],
        )


if __name__ == "__main__":
    unittest.main()
