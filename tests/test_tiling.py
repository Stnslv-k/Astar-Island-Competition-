import unittest

from astar_island.predict import ObservationAccumulator
from astar_island.tiling import choose_adaptive_refinement_queries


class TilingTests(unittest.TestCase):
    def test_adaptive_refinement_targets_uncertain_window(self) -> None:
        accumulator = ObservationAccumulator(width=20, height=20)
        accumulator.add_patch(
            viewport={"x": 0, "y": 0, "w": 15, "h": 15},
            grid=[[1] * 15 for _ in range(15)],
        )
        accumulator.add_patch(
            viewport={"x": 5, "y": 5, "w": 15, "h": 15},
            grid=[[1] * 15 for _ in range(15)],
        )
        accumulator.add_patch(
            viewport={"x": 5, "y": 5, "w": 1, "h": 1},
            grid=[[2]],
        )

        detail = {
            "map_width": 20,
            "map_height": 20,
            "seeds_count": 1,
            "initial_states": [
                {
                    "grid": [[11] * 20 for _ in range(20)],
                    "settlements": [],
                }
            ],
        }

        queries = choose_adaptive_refinement_queries(
            round_id="round-1",
            detail=detail,
            accumulators=[accumulator],
            budget=1,
            window=15,
            stride=5,
            prior_strength=0.10,
        )

        self.assertEqual(len(queries), 1)
        self.assertEqual(queries[0].seed_index, 0)
        self.assertLessEqual(queries[0].viewport_x, 5)
        self.assertGreater(queries[0].viewport_x + queries[0].viewport_w, 5)
        self.assertLessEqual(queries[0].viewport_y, 5)
        self.assertGreater(queries[0].viewport_y + queries[0].viewport_h, 5)


if __name__ == "__main__":
    unittest.main()
