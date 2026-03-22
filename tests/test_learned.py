import unittest

from astar_island.learned import LearnedPriorModel, build_augmented_priors


class LearnedPriorTests(unittest.TestCase):
    def test_augmented_priors_use_learned_signal(self) -> None:
        initial_state = {
            "grid": [
                [1, 1],
                [1, 1],
            ],
            "settlements": [],
        }
        model = LearnedPriorModel(
            full_stats={
                (1, 1, 4, 0, 0): [0, 8, 0, 0, 0, 0],
                (1, 1, 4, 1, 0): [0, 0, 7, 0, 0, 0],
            },
            coarse_stats={(1, 1, 0): [0, 6, 4, 0, 0, 0]},
            center_stats={(1, 0): [0, 10, 6, 0, 0, 0]},
            total_examples=16,
        )

        priors, diagnostics = build_augmented_priors(initial_state, model, learned_blend=0.50)

        self.assertEqual(diagnostics["enabled"], 1)
        self.assertEqual(diagnostics["training_examples"], 16)
        self.assertEqual(len(priors), 2)
        self.assertEqual(len(priors[0]), 2)
        for row in priors:
            for cell in row:
                self.assertAlmostEqual(sum(cell), 1.0, places=6)
        self.assertGreater(priors[0][0][1], 0.58)


if __name__ == "__main__":
    unittest.main()
