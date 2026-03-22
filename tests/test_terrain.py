import unittest

from astar_island.terrain import build_initial_priors, terrain_value_to_submission_class


class TerrainTests(unittest.TestCase):
    def test_empty_like_terrain_values_collapse_to_submission_class_zero(self) -> None:
        self.assertEqual(terrain_value_to_submission_class(0), 0)
        self.assertEqual(terrain_value_to_submission_class(10), 0)
        self.assertEqual(terrain_value_to_submission_class(11), 0)

    def test_initial_settlement_overrides_cell_prior(self) -> None:
        priors = build_initial_priors(
            {
                "grid": [[11, 11], [11, 11]],
                "settlements": [{"x": 1, "y": 0, "has_port": True, "alive": True}],
            }
        )
        self.assertGreater(priors[0][1][2], priors[0][1][0])
        self.assertAlmostEqual(sum(priors[0][1]), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
